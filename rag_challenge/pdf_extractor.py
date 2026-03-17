import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import List

import fitz  # pymupdf

from config import PDF_DIR, CACHE_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    pdf_sha1: str
    page_index: int
    chunk_index: int
    text: str


def _split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def extract_pdf(pdf_path: str) -> List[Chunk]:
    sha1 = os.path.splitext(os.path.basename(pdf_path))[0]
    chunks: List[Chunk] = []

    try:
        doc = fitz.open(pdf_path)
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text("text").strip()
            if not text:
                try:
                    tp = page.get_textpage_ocr(flags=0, full=True, language="eng")
                    text = page.get_text(textpage=tp).strip()
                except Exception:
                    pass
            if not text:
                continue

            for ci, chunk_text in enumerate(_split_text(text)):
                chunk_text = chunk_text.strip()
                if len(chunk_text) > 30:
                    chunks.append(Chunk(
                        pdf_sha1=sha1,
                        page_index=page_idx,
                        chunk_index=ci,
                        text=chunk_text,
                    ))
        doc.close()
    except Exception as e:
        logger.error(f"Failed to extract {pdf_path}: {e}")

    return chunks


def extract_all_pdfs(pdf_dir: str = PDF_DIR, cache_dir: str = CACHE_DIR) -> List[Chunk]:
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "chunks.json")

    if os.path.exists(cache_file):
        logger.info("Loading chunks from cache...")
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Chunk(**d) for d in data]

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    logger.info(f"Extracting text from {len(pdf_files)} PDFs...")

    all_chunks: List[Chunk] = []
    for i, fname in enumerate(pdf_files, 1):
        chunks = extract_pdf(os.path.join(pdf_dir, fname))
        all_chunks.extend(chunks)
        if i % 10 == 0:
            logger.info(f"  {i}/{len(pdf_files)} processed, {len(all_chunks)} chunks so far")

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in all_chunks], f, ensure_ascii=False)

    logger.info(f"Extracted {len(all_chunks)} chunks from {len(pdf_files)} PDFs")
    return all_chunks
