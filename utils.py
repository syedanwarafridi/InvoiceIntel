from pathlib import Path
from typing import List
import warnings, logging
import numpy as np
import pdfplumber
from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError,
    PDFPopplerTimeoutError,
)
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InvoiceIntel-OCR")


# -------------------- OCR back‑end factories (with error capture) -------------
def _get_easyocr_reader(languages: List[str]):
    try:
        import easyocr
    except ModuleNotFoundError as e:
        raise RuntimeError("EasyOCR not installed.") from e
    return easyocr.Reader(lang_list=languages, gpu=False)


def _get_paddleocr_reader(languages: List[str]):
    try:
        from paddleocr import PaddleOCR
    except ModuleNotFoundError as e:
        raise RuntimeError("PaddleOCR not installed.") from e
    lang = "en" if "en" in languages else languages[0]
    return PaddleOCR(lang=lang, use_angle_cls=True, show_log=False)


# --------------------------- Helper utilities --------------------------------
def _pdfplumber_text(pdf_path: str) -> str:
    """Return embedded text or '' if pdfplumber fails."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(
                (page.extract_text() or "").strip() for page in pdf.pages
            ).strip()
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e!s}")
        return ""


def _pdf_to_images(pdf_path: str, dpi: int = 300):
    """Convert PDF to list of PIL images; may raise pdf2image exceptions."""
    try:
        return convert_from_path(pdf_path, dpi=dpi)
    except (
        PDFInfoNotInstalledError,
        PDFPageCountError,
        PDFSyntaxError,
        PDFPopplerTimeoutError,
    ) as known:
        logger.warning(f"pdf2image conversion problem: {known!s}")
        raise
    except Exception as e:
        logger.warning(f"Unexpected pdf2image failure: {e!s}")
        raise


def _ocr_easy(images, languages):
    try:
        reader = _get_easyocr_reader(languages)
        return "\n".join(
            " ".join([res[1] for res in reader.readtext(np.array(img))])
            for img in images
        )
    except Exception as e:
        logger.warning(f"EasyOCR failed: {e!s}")
        raise


def _ocr_paddle(images, languages):
    try:
        ocr = _get_paddleocr_reader(languages)
        return "\n".join(
            " ".join([line[1][0] for line in ocr.ocr(np.array(img), cls=True)])
            for img in images
        )
    except Exception as e:
        logger.warning(f"PaddleOCR failed: {e!s}")
        raise


# ----------------------- Public orchestration function -----------------------
def extract_text_auto(
    file_path: str,
    ocr_backend: str = "easyocr",
    languages: List[str] = ["en"],
) -> str:
    """
    • PDFs → try pdfplumber → OCR fallback if needed
    • Raster images (.png/.jpg/.jpeg/.tiff/.bmp) → OCR directly
    Raises RuntimeError if all extraction steps fail.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)

    ext = path.suffix.lower()
    img_exts = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

    try:
        # ---------- Branch 1: PDF ----------
        if ext == ".pdf":
            text = _pdfplumber_text(str(path))
            if len(text.split()) > 20:
                return text

            warnings.warn("PDF seems scanned — switching to OCR.")
            images = _pdf_to_images(str(path))

        # ---------- Branch 2: Image ----------
        elif ext in img_exts:
            images = [Image.open(path)]

        else:
            raise ValueError(f"Unsupported file type: {ext}")

        if ocr_backend == "easyocr":
            return _ocr_easy(images, languages)
        elif ocr_backend == "paddle":
            return _ocr_paddle(images, languages)
        else:
            raise ValueError("ocr_backend must be 'easyocr' or 'paddle'")

    except Exception as e:
        raise RuntimeError(f"Text extraction failed: {e!s}") from e
