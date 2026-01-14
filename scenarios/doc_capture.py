"""
Document Capture Handler for the TimeLine Content Capture Pipeline.
Handles PDF text extraction and OCR for images/scanned documents.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import hashlib

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import easyocr
    HAS_EASYOCR = True
    _ocr_reader = None  # Lazy initialization
except ImportError:
    HAS_EASYOCR = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    get_db, CaptureSession, CapturedText, CapturedMedia,
    SessionType, MediaType, MEDIA_DOCS_DIR
)


def _get_ocr_reader():
    """Lazy initialize EasyOCR reader (it's slow to load)."""
    global _ocr_reader
    if _ocr_reader is None and HAS_EASYOCR:
        print("🔄 Initializing EasyOCR (first time may take a while)...")
        _ocr_reader = easyocr.Reader(['en'], gpu=True)
    return _ocr_reader


class DocCaptureHandler:
    """
    Handles document capture:
    - PDF text extraction using PyMuPDF
    - OCR for images/scanned PDFs using EasyOCR
    - Stores original file and extracted text in database
    """
    
    SUPPORTED_PDF_EXTENSIONS = ['.pdf']
    SUPPORTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp']
    
    def __init__(self):
        pass
    
    def capture(self, file_path: str, session_id: str, original_filename: str = None) -> Dict[str, Any]:
        """
        Capture content from a document file.
        
        Args:
            file_path: Path to the uploaded/local file
            session_id: The capture session ID
            original_filename: Original filename if uploaded
            
        Returns:
            Dict with status, extracted content info, and any errors
        """
        result = {
            "success": False,
            "session_id": session_id,
            "file_path": file_path,
            "original_filename": original_filename,
            "text_length": 0,
            "page_count": 0,
            "method": None,  # "pdf_extract" or "ocr"
            "errors": []
        }
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            result["errors"].append(f"File not found: {file_path}")
            return result
        
        ext = file_path.suffix.lower()
        
        try:
            # Determine processing method based on file type
            if ext in self.SUPPORTED_PDF_EXTENSIONS:
                extracted_text, page_count, method = self._process_pdf(file_path)
                result["page_count"] = page_count
                result["method"] = method
            elif ext in self.SUPPORTED_IMAGE_EXTENSIONS:
                extracted_text = self._process_image(file_path)
                result["method"] = "ocr"
            else:
                result["errors"].append(f"Unsupported file type: {ext}")
                return result
            
            result["text_length"] = len(extracted_text) if extracted_text else 0
            
            # Copy file to media directory
            stored_path = self._store_file(file_path, session_id, original_filename)
            
            # Store in database
            db = get_db()
            try:
                # Update session
                session = db.query(CaptureSession).filter_by(id=session_id).first()
                if session:
                    session.title = original_filename or file_path.name
                    session.source_path = str(stored_path)
                
                # Store extracted text
                if extracted_text:
                    text_record = CapturedText(
                        session_id=session_id,
                        content=extracted_text,
                        source_url_or_path=str(stored_path),
                        content_type=result["method"]
                    )
                    db.add(text_record)
                
                # Store file reference
                media_record = CapturedMedia(
                    session_id=session_id,
                    file_path=str(stored_path),
                    media_type=MediaType.IMAGE if ext in self.SUPPORTED_IMAGE_EXTENSIONS else MediaType.IMAGE
                )
                db.add(media_record)
                
                db.commit()
                result["success"] = True
                result["stored_path"] = str(stored_path)
                print(f"✅ Document capture complete: {result['text_length']} chars via {result['method']}")
                
            except Exception as e:
                db.rollback()
                result["errors"].append(f"Database error: {str(e)}")
            finally:
                db.close()
                
        except Exception as e:
            result["errors"].append(f"Processing failed: {str(e)}")
        
        return result
    
    def _process_pdf(self, file_path: Path) -> tuple[str, int, str]:
        """
        Process a PDF file.
        First tries direct text extraction, falls back to OCR if needed.
        
        Returns:
            Tuple of (extracted_text, page_count, method)
        """
        if not HAS_PYMUPDF:
            raise RuntimeError("PyMuPDF not installed. Cannot process PDF files.")
        
        print(f"📄 Processing PDF: {file_path.name}")
        
        doc = fitz.open(file_path)
        page_count = len(doc)
        
        # Try direct text extraction first
        all_text = []
        has_text = False
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                has_text = True
                all_text.append(f"--- Page {page_num + 1} ---\n{text}")
        
        doc.close()
        
        # If we got significant text, use it
        if has_text:
            extracted = "\n\n".join(all_text)
            # Check if it's actually meaningful text (not just headers/footers)
            if len(extracted.strip()) > 100:
                print(f"  📖 Extracted text from {page_count} pages")
                return extracted, page_count, "pdf_extract"
        
        # Fall back to OCR
        print(f"  🔍 PDF appears scanned, using OCR...")
        return self._ocr_pdf(file_path), page_count, "ocr"
    
    def _ocr_pdf(self, file_path: Path) -> str:
        """OCR a PDF by converting pages to images."""
        if not HAS_PYMUPDF:
            raise RuntimeError("PyMuPDF not installed")
        if not HAS_EASYOCR:
            raise RuntimeError("EasyOCR not installed. Cannot perform OCR.")
        
        reader = _get_ocr_reader()
        doc = fitz.open(file_path)
        all_text = []
        
        for page_num, page in enumerate(doc):
            # Render page to image
            mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to bytes
            img_bytes = pix.tobytes("png")
            
            # OCR the image
            result = reader.readtext(img_bytes)
            text = " ".join([detection[1] for detection in result])
            
            if text.strip():
                all_text.append(f"--- Page {page_num + 1} ---\n{text}")
            
            print(f"  🔍 OCR page {page_num + 1}/{len(doc)}")
        
        doc.close()
        return "\n\n".join(all_text)
    
    def _process_image(self, file_path: Path) -> str:
        """Process an image file using OCR."""
        if not HAS_EASYOCR:
            raise RuntimeError("EasyOCR not installed. Cannot perform OCR.")
        
        print(f"🖼️ Processing image with OCR: {file_path.name}")
        
        reader = _get_ocr_reader()
        result = reader.readtext(str(file_path))
        
        # Extract text from results
        text = " ".join([detection[1] for detection in result])
        
        print(f"  📝 OCR extracted {len(text)} characters")
        return text
    
    def _store_file(self, source_path: Path, session_id: str, original_filename: str = None) -> Path:
        """Copy file to media directory."""
        # Create session directory
        session_dir = MEDIA_DOCS_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine filename
        filename = original_filename or source_path.name
        
        # Add hash to avoid overwrites
        file_hash = hashlib.md5(open(source_path, 'rb').read()[:4096]).hexdigest()[:6]
        stem = Path(filename).stem
        ext = Path(filename).suffix
        final_filename = f"{stem}_{file_hash}{ext}"
        
        dest_path = session_dir / final_filename
        
        # Copy file
        shutil.copy2(source_path, dest_path)
        print(f"  💾 Stored file: {dest_path}")
        
        return dest_path
    
    @staticmethod
    def get_capabilities() -> Dict[str, bool]:
        """Return available capabilities."""
        return {
            "pdf_extract": HAS_PYMUPDF,
            "ocr": HAS_EASYOCR,
            "supported_extensions": DocCaptureHandler.SUPPORTED_PDF_EXTENSIONS + DocCaptureHandler.SUPPORTED_IMAGE_EXTENSIONS
        }


if __name__ == "__main__":
    # Quick test
    handler = DocCaptureHandler()
    print(f"Capabilities: {handler.get_capabilities()}")
