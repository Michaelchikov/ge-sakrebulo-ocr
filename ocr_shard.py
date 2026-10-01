"""OCR one shard of public Tbilisi City Council (Sakrebulo) PDFs.

Each runner takes every SHARDS-th line of urls.tsv, downloads the PDF from
tbsakrebulo.gov.ge (robots.txt allows all), and OCRs every page with the same settings
as the local archive pipeline: 300 dpi grayscale (huge pages scaled to 7000 px),
Tesseract `kat+eng`, the bundled traineddata, OMP_THREAD_LIMIT=1.

Downloads are paced so that all shards together stay at or below MAX_TOTAL_RPS.

    python3 ocr_shard.py <shard> <shards>
"""
import gzip
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SHARD, SHARDS = int(sys.argv[1]), int(sys.argv[2])
MAX_TOTAL_RPS = 2.0
MIN_INTERVAL_S = SHARDS / MAX_TOTAL_RPS
UA = "GeLegalArchive/1.0 (research archive; OCR of public Tbilisi Sakrebulo acts)"
OCR_DPI = 300
MAX_OCR_LONG_SIDE_PX = 7000
OCR_LANG = "kat+eng"
TESSDATA = os.path.abspath("tessdata")
CPUS = os.cpu_count() or 4
PREFETCH = 6
RETRIES = 4
OUT = os.path.join("results", f"shard-{SHARD:02d}.jsonl")
ENV = {**os.environ, "OMP_THREAD_LIMIT": "1"}

rows = [line.rstrip("\n").split("\t") for line in open("urls.tsv", encoding="utf-8") if line.strip()]
mine = [(url, int(size)) for i, (url, size) in enumerate(rows) if i % SHARDS == SHARD]


def fetch(url: str) -> bytes:
    delay = 30
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == RETRIES - 1:
                raise
            print(f"retry {url}: {exc}", flush=True)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def downloader(q: queue.Queue) -> None:
    last = 0.0
    for url, size in mine:
        wait = MIN_INTERVAL_S - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        last = time.time()
        try:
            data = fetch(url)
            q.put((url, size, data if len(data) == size else None,
                   None if len(data) == size else f"size {len(data)} != {size}"))
        except Exception as exc:  # report and continue with the next file
            q.put((url, size, None, f"download: {exc}"))
    q.put(None)


def ocr_pdf(data: bytes, pool: ThreadPoolExecutor) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(data)
        tmp.flush()
        info = subprocess.run(["pdfinfo", "-f", "1", "-l", "100000", tmp.name],
                              capture_output=True, text=True).stdout
        m = re.search(r"^Pages:\s+(\d+)", info, re.M)
        n = int(m.group(1)) if m else 0
        sizes = {int(p): (float(w), float(h)) for p, w, h in
                 re.findall(r"^Page\s+(\d+)\s+size:\s+([\d.]+)\s+x\s+([\d.]+)", info, re.M)}

        def one(page: int):
            size = sizes.get(page)
            scale = (size and max(size) * OCR_DPI / 72 > MAX_OCR_LONG_SIDE_PX)
            args = ["-scale-to", str(MAX_OCR_LONG_SIDE_PX)] if scale else ["-r", str(OCR_DPI)]
            try:
                img = subprocess.run(["pdftoppm", *args, "-gray", "-f", str(page), "-l", str(page), tmp.name],
                                     capture_output=True, timeout=180).stdout
                if not img:
                    return page, None, "pdftoppm:empty"
                out = subprocess.run(["tesseract", "stdin", "stdout", "-l", OCR_LANG, "--tessdata-dir", TESSDATA],
                                     input=img, capture_output=True, timeout=300, env=ENV)
                if out.returncode != 0:
                    return page, None, f"tesseract:exit {out.returncode}"
                return page, out.stdout.decode("utf-8", "replace"), None
            except subprocess.TimeoutExpired:
                return page, None, "timeout"

        results = list(pool.map(one, range(1, n + 1)))
    return {"pages_total": n,
            "pages": {str(p): t for p, t, e in results if t is not None},
            "errors": {str(p): e for p, t, e in results if e}}


def main() -> None:
    os.makedirs("results", exist_ok=True)
    version = subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout.split("\n")[0]
    engine = f"github-actions {version} {OCR_LANG} {OCR_DPI}dpi"
    q: queue.Queue = queue.Queue(maxsize=PREFETCH)
    threading.Thread(target=downloader, args=(q,), daemon=True).start()
    done = failed = 0
    started = time.time()
    with ThreadPoolExecutor(CPUS) as pool, open(OUT, "w", encoding="utf-8") as out:
        while (item := q.get()) is not None:
            url, size, data, error = item
            record = {"url": url, "size": size, "engine": engine}
            if data is None:
                record["error"] = error
                failed += 1
            else:
                try:
                    record.update(ocr_pdf(data, pool))
                    done += 1
                except Exception as exc:
                    record["error"] = f"ocr: {exc}"
                    failed += 1
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if (done + failed) % 10 == 0:
                rate = (done + failed) / (time.time() - started)
                print(f"[shard {SHARD}] {done + failed}/{len(mine)} ok={done} fail={failed} "
                      f"rate={rate:.2f}/s eta={(len(mine) - done - failed) / max(rate, 1e-9) / 60:.0f}min", flush=True)
    with open(OUT, "rb") as src, gzip.open(OUT + ".gz", "wb") as dst:
        dst.write(src.read())
    os.remove(OUT)
    print(f"[shard {SHARD}] finished ok={done} fail={failed}", flush=True)


if __name__ == "__main__":
    main()
