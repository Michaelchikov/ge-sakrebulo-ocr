# ge-sakrebulo-ocr

One-off OCR of public, scanned acts of the Tbilisi City Council (Sakrebulo), published at https://tbsakrebulo.gov.ge. The text feeds a private legal-research archive.

- `urls.tsv` lists the public PDF URLs and their expected sizes.
- `ocr_shard.py` downloads its share of the PDFs (all shards together make at most 2 requests a second) and OCRs every page with Tesseract `kat+eng` at 300 dpi.
- `tessdata/` holds the Tesseract models (Apache-2.0, from the tesseract-ocr project).
- Results are uploaded as workflow artifacts that expire after 3 days.
