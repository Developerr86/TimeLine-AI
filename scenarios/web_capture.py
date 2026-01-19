"""
Web Capture Handler for the TimeLine Content Capture Pipeline.
Handles extraction of article content and images from web URLs.
"""

import requests
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin, urlparse
import hashlib

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    get_db, CaptureSession, CapturedText, CapturedMedia,
    SessionType, MediaType, MEDIA_WEB_DIR
)


class WebCaptureHandler:
    """
    Handles web article capture:
    - Fetches URL content
    - Extracts main text using trafilatura (fallback: BeautifulSoup)
    - Downloads images from the page
    - Stores everything in the database
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.timeout = 30
    
    def capture(self, url: str, session_id: str) -> Dict[str, Any]:
        """
        Capture content from a web URL.
        
        Args:
            url: The URL to capture
            session_id: The capture session ID
            
        Returns:
            Dict with status, captured content info, and any errors
        """
        result = {
            "success": False,
            "session_id": session_id,
            "url": url,
            "title": None,
            "text_length": 0,
            "images_downloaded": 0,
            "errors": []
        }
        
        try:
            # Fetch the page
            print(f"🌐 Fetching URL: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            html_content = response.text
            
            # Extract main content
            title, main_text = self._extract_content(html_content, url)
            result["title"] = title or "Untitled"
            
            if not main_text:
                result["errors"].append("Could not extract main content")
                main_text = ""
            
            result["text_length"] = len(main_text)
            
            # Extract and download images
            image_urls = self._extract_image_urls(html_content, url)
            downloaded_images = self._download_images(image_urls, session_id)
            result["images_downloaded"] = len(downloaded_images)
            
            # Store in database
            db = get_db()
            try:
                # Update session with title and URL
                session = db.query(CaptureSession).filter_by(id=session_id).first()
                if session:
                    session.title = title
                    session.source_url = url
                
                # Store extracted text
                if main_text:
                    text_record = CapturedText(
                        session_id=session_id,
                        content=main_text,
                        source_url_or_path=url,
                        content_type="article"
                    )
                    db.add(text_record)
                
                # Store image records
                for img_path, img_url in downloaded_images:
                    media_record = CapturedMedia(
                        session_id=session_id,
                        file_path=str(img_path),
                        media_type=MediaType.IMAGE,
                        original_url=img_url
                    )
                    db.add(media_record)
                
                db.commit()
                result["success"] = True
                
                # Save to transcript.txt for frontend compatibility
                if main_text:
                    self._save_transcript(session_id, main_text)
                
                print(f"✅ Web capture complete: {len(main_text)} chars, {len(downloaded_images)} images")
                
            except Exception as e:
                db.rollback()
                result["errors"].append(f"Database error: {str(e)}")
            finally:
                db.close()
                
        except requests.RequestException as e:
            result["errors"].append(f"Request failed: {str(e)}")
        except Exception as e:
            result["errors"].append(f"Capture failed: {str(e)}")
        
        return result
    
    def process_snapshot(self, snapshot_path: str, session_id: str, url: str = None) -> Dict[str, Any]:
        """
        Process a saved HTML snapshot instead of fetching from network.
        
        Args:
            snapshot_path: Path to the saved HTML file
            session_id: The capture session ID
            url: Original URL (for metadata)
            
        Returns:
            Dict with status, captured content info, and any errors
        """
        result = {
            "success": False,
            "session_id": session_id,
            "url": url,
            "title": None,
            "text_length": 0,
            "images_downloaded": 0,
            "method": "snapshot",
            "errors": []
        }
        
        snapshot_path = Path(snapshot_path)
        
        if not snapshot_path.exists():
            result["errors"].append(f"Snapshot file not found: {snapshot_path}")
            return result
        
        try:
            # Read HTML from snapshot
            print(f"📖 Reading snapshot: {snapshot_path}")
            with open(snapshot_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Extract main content (same as capture method)
            title, main_text = self._extract_content(html_content, url or '')
            result["title"] = title or "Untitled"
            
            if not main_text:
                result["errors"].append("Could not extract main content from snapshot")
                main_text = ""
            
            result["text_length"] = len(main_text)
            
            # Extract and download images (optional, might skip for offline)
            if url:
                image_urls = self._extract_image_urls(html_content, url)
                downloaded_images = self._download_images(image_urls, session_id)
                result["images_downloaded"] = len(downloaded_images)
            else:
                downloaded_images = []
            
            # Store in database
            db = get_db()
            try:
                # Update session with title and URL
                session = db.query(CaptureSession).filter_by(id=session_id).first()
                if session:
                    session.title = title
                    if url:
                        session.source_url = url
                
                # Store extracted text
                if main_text:
                    text_record = CapturedText(
                        session_id=session_id,
                        content=main_text,
                        source_url_or_path=url or str(snapshot_path),
                        content_type="article"
                    )
                    db.add(text_record)
                
                # Store image records
                for img_path, img_url in downloaded_images:
                    media_record = CapturedMedia(
                        session_id=session_id,
                        file_path=str(img_path),
                        media_type=MediaType.IMAGE,
                        original_url=img_url
                    )
                    db.add(media_record)
                
                # Move snapshot to permanent session folder
                session_dir = MEDIA_WEB_DIR / session_id
                session_dir.mkdir(parents=True, exist_ok=True)
                permanent_snapshot = session_dir / "snapshot.html"
                
                import shutil
                shutil.move(str(snapshot_path), str(permanent_snapshot))
                print(f"  💾 Moved snapshot to: {permanent_snapshot}")
                
                db.commit()
                result["success"] = True
                result["snapshot_moved"] = str(permanent_snapshot)
                
                # Save to transcript.txt for frontend compatibility
                self._save_transcript(session_id, main_text, session_dir)
                
                print(f"✅ Web snapshot processed: {len(main_text)} chars, {len(downloaded_images)} images")
                
            except Exception as e:
                db.rollback()
                result["errors"].append(f"Database error: {str(e)}")
            finally:
                db.close()
                
        except Exception as e:
            result["errors"].append(f"Snapshot processing failed: {str(e)}")
            print(f"❌ Snapshot processing failed: {e}")
        
        return result
    
    def _save_transcript(self, session_id: str, text: str, session_dir: Path = None):
        """Save extracted text to transcript.txt for frontend compatibility."""
        try:
            if session_dir is None:
                session_dir = MEDIA_WEB_DIR / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            
            transcript_path = session_dir / "transcript.txt"
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"  📝 Saved transcript to: {transcript_path}")
        except Exception as e:
            print(f"  ⚠️ Failed to save transcript: {e}")
    
    def _extract_content(self, html: str, url: str) -> tuple[Optional[str], Optional[str]]:
        """Extract title and main text content from HTML."""
        title = None
        text = None
        
        # Try trafilatura first (better quality)
        if HAS_TRAFILATURA:
            try:
                text = trafilatura.extract(
                    html,
                    include_comments=False,
                    include_tables=True,
                    no_fallback=False
                )
                # Try to get title from metadata
                metadata = trafilatura.extract_metadata(html)
                if metadata:
                    title = metadata.title
            except Exception as e:
                print(f"⚠️ Trafilatura extraction failed: {e}")
        
        # Fallback to BeautifulSoup
        if not text and HAS_BS4:
            try:
                soup = BeautifulSoup(html, 'html.parser')
                
                # Get title
                title_tag = soup.find('title')
                if title_tag:
                    title = title_tag.get_text().strip()
                
                # Remove script and style elements
                for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    element.decompose()
                
                # Try to find main content
                main_content = soup.find('main') or soup.find('article') or soup.find('body')
                if main_content:
                    text = main_content.get_text(separator='\n', strip=True)
                    
            except Exception as e:
                print(f"⚠️ BeautifulSoup extraction failed: {e}")
        
        return title, text
    
    def _extract_image_urls(self, html: str, base_url: str) -> List[str]:
        """Extract image URLs from HTML."""
        image_urls = []
        
        if not HAS_BS4:
            return image_urls
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            for img in soup.find_all('img'):
                src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                if src:
                    # Convert relative URLs to absolute
                    absolute_url = urljoin(base_url, src)
                    
                    # Filter out tracking pixels, icons, etc.
                    if self._is_valid_image_url(absolute_url):
                        image_urls.append(absolute_url)
                        
        except Exception as e:
            print(f"⚠️ Image URL extraction failed: {e}")
        
        return list(set(image_urls))[:20]  # Limit to 20 images
    
    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL is likely a valid content image."""
        # Skip data URLs, tracking pixels, etc.
        if url.startswith('data:'):
            return False
        
        # Check for common image extensions
        parsed = urlparse(url)
        path_lower = parsed.path.lower()
        
        # Skip small icons and tracking pixels
        skip_patterns = ['1x1', 'pixel', 'tracking', 'beacon', 'spacer', 'blank', 'icon']
        if any(pattern in path_lower for pattern in skip_patterns):
            return False
        
        # Prefer actual image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.avif']
        has_image_ext = any(path_lower.endswith(ext) for ext in image_extensions)
        
        return has_image_ext or 'image' in url.lower()
    
    def _download_images(self, image_urls: List[str], session_id: str) -> List[tuple[Path, str]]:
        """Download images and return list of (local_path, original_url) tuples."""
        downloaded = []
        
        # Create session directory
        session_dir = MEDIA_WEB_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        for i, img_url in enumerate(image_urls):
            try:
                response = self.session.get(img_url, timeout=10)
                response.raise_for_status()
                
                # Determine file extension from content type or URL
                content_type = response.headers.get('content-type', '')
                ext = self._get_image_extension(img_url, content_type)
                
                # Create filename from hash of URL
                url_hash = hashlib.md5(img_url.encode()).hexdigest()[:8]
                filename = f"img_{i:03d}_{url_hash}{ext}"
                filepath = session_dir / filename
                
                # Save image
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                downloaded.append((filepath, img_url))
                print(f"  📷 Downloaded: {filename}")
                
            except Exception as e:
                print(f"  ⚠️ Failed to download {img_url[:50]}...: {e}")
        
        return downloaded
    
    def _get_image_extension(self, url: str, content_type: str) -> str:
        """Determine image file extension."""
        # Try content type first
        type_map = {
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'image/webp': '.webp',
            'image/avif': '.avif',
        }
        
        for mime, ext in type_map.items():
            if mime in content_type:
                return ext
        
        # Try URL
        parsed = urlparse(url)
        path = parsed.path.lower()
        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.avif']:
            if path.endswith(ext):
                return ext if ext != '.jpeg' else '.jpg'
        
        return '.jpg'  # Default


if __name__ == "__main__":
    # Quick test
    handler = WebCaptureHandler()
    print(f"Trafilatura available: {HAS_TRAFILATURA}")
    print(f"BeautifulSoup available: {HAS_BS4}")
