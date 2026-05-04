"""
OCR PDFs in data/ using LlamaParse API.
Saves extracted markdown to data/<filename>.md

Usage:
    python ocr_pdfs.py

Requires: pip install llama-parse  (or uses raw REST API as fallback)
"""

import os
import sys
import glob
import time

# ── Config ────────────────────────────────────────────────
API_KEY = os.getenv(
    "LLAMA_CLOUD_API_KEY",
    ""
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def ocr_with_sdk(pdf_path: str) -> str:
    """Use llama-parse SDK (preferred)."""
    from llama_parse import LlamaParse

    parser = LlamaParse(
        api_key=API_KEY,
        result_type="markdown",
        language="vi",
        verbose=True,
    )
    docs = parser.load_data(pdf_path)
    return "\n\n".join(doc.text for doc in docs if doc.text.strip())


def ocr_with_rest(pdf_path: str) -> str:
    """
    Fallback: call LlamaParse REST API directly using only stdlib.
    Uploads the file, polls for completion, downloads result.
    """
    import urllib.request
    import urllib.parse
    import json
    import mimetypes

    base_url = "https://api.cloud.llamaindex.ai/api/parsing"
    headers_json = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
    }

    filename = os.path.basename(pdf_path)
    print(f"  [REST] Uploading {filename} ...")

    # ── 1. Upload file ────────────────────────────────────
    boundary = "----LlamaParseFormBoundary"
    with open(pdf_path, "rb") as f:
        file_data = f.read()

    body_parts = []
    # result_type field
    body_parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="result_type"\r\n\r\n'
        f"markdown\r\n"
    )
    # language field
    body_parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="language"\r\n\r\n'
        f"vi\r\n"
    )
    # file field
    body_parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    )

    body = b"".join(p.encode() if isinstance(p, str) else p for p in body_parts)
    body += file_data
    body += f"\r\n--{boundary}--\r\n".encode()

    upload_req = urllib.request.Request(
        f"{base_url}/upload",
        data=body,
        headers={
            **headers_json,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )

    with urllib.request.urlopen(upload_req) as resp:
        upload_result = json.loads(resp.read().decode())

    job_id = upload_result.get("id")
    if not job_id:
        raise RuntimeError(f"Upload failed: {upload_result}")
    print(f"  [REST] Job ID: {job_id}")

    # ── 2. Poll for completion ────────────────────────────
    status_url = f"{base_url}/job/{job_id}"
    for attempt in range(60):  # max 5 minutes
        time.sleep(5)
        status_req = urllib.request.Request(
            status_url,
            headers=headers_json,
            method="GET",
        )
        with urllib.request.urlopen(status_req) as resp:
            status = json.loads(resp.read().decode())

        job_status = status.get("status", "")
        print(f"  [REST] Status: {job_status} (attempt {attempt + 1})")

        if job_status == "SUCCESS":
            break
        elif job_status in ("ERROR", "CANCELLED"):
            raise RuntimeError(f"Job failed: {status}")
    else:
        raise TimeoutError("LlamaParse job timed out after 5 minutes")

    # ── 3. Download result ────────────────────────────────
    result_url = f"{base_url}/job/{job_id}/result/markdown"
    result_req = urllib.request.Request(
        result_url,
        headers=headers_json,
        method="GET",
    )
    with urllib.request.urlopen(result_req) as resp:
        result = json.loads(resp.read().decode())

    # Result is a list of pages with "md" field
    pages = result.get("pages", [])
    if pages:
        return "\n\n".join(p.get("md", "") for p in pages if p.get("md", "").strip())

    # Some versions return "markdown" directly
    return result.get("markdown", "")


def ocr_pdf(pdf_path: str) -> str:
    """Try SDK first, fall back to REST API."""
    try:
        return ocr_with_sdk(pdf_path)
    except ImportError:
        print("  [INFO] llama-parse SDK not ready, using REST API fallback...")
        return ocr_with_rest(pdf_path)


def main():
    pdf_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}")
        sys.exit(0)

    print(f"Found {len(pdf_files)} PDF(s) to OCR:\n")
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        out_path = pdf_path.replace(".pdf", "_ocr.md")

        # Skip if already done
        if os.path.exists(out_path):
            print(f"  [SKIP] {filename} → already OCR'd ({out_path})")
            continue

        print(f"  Processing: {filename}")
        try:
            text = ocr_pdf(pdf_path)
            if text.strip():
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"  [OK] Saved → {os.path.basename(out_path)} ({len(text):,} chars)\n")
            else:
                print(f"  [WARN] No text extracted from {filename}\n")
        except Exception as e:
            print(f"  [ERROR] {filename}: {e}\n")

    print("Done.")


if __name__ == "__main__":
    main()
