"""
Contact Sheet Generator for Video Frames

Creates 4x4 grid contact sheets from video frames for batch analysis.
Each contact sheet contains 16 thumbnails (300x300 each = 1200x1200 total).
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️ OpenCV not available - contact sheet creation disabled")


THUMBNAIL_SIZE = 300  # 300x300 pixels per thumbnail
GRID_SIZE = 4  # 4x4 grid
CONTACT_SHEET_SIZE = THUMBNAIL_SIZE * GRID_SIZE  # 1200x1200


def create_contact_sheet(frames: List[np.ndarray], output_path: Path) -> Optional[np.ndarray]:
    """
    Create a single 4x4 contact sheet from up to 16 frames.
    
    Args:
        frames: List of frame images (numpy arrays)
        output_path: Path to save the contact sheet
        
    Returns:
        The contact sheet image as numpy array, or None if failed
    """
    if not HAS_CV2:
        return None
    
    # Create black canvas
    contact_sheet = np.zeros((CONTACT_SHEET_SIZE, CONTACT_SHEET_SIZE, 3), dtype=np.uint8)
    
    # Fill with black in case of fewer than 16 frames
    # (already black, but being explicit)
    
    # Place each frame in the grid
    for idx, frame in enumerate(frames):
        if idx >= 16:
            break
            
        row = idx // 4
        col = idx % 4
        
        # Resize frame to thumbnail size
        try:
            resized = cv2.resize(frame, (THUMBNAIL_SIZE, THUMBNAIL_SIZE))
            
            # Convert grayscale to RGB if needed
            if len(resized.shape) == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            
            # Calculate position
            y = row * THUMBNAIL_SIZE
            x = col * THUMBNAIL_SIZE
            
            # Place in contact sheet
            contact_sheet[y:y+THUMBNAIL_SIZE, x:x+THUMBNAIL_SIZE] = resized
        except Exception as e:
            print(f"⚠️ Failed to resize frame {idx}: {e}")
    
    # Save contact sheet
    try:
        cv2.imwrite(str(output_path), contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return contact_sheet
    except Exception as e:
        print(f"⚠️ Failed to save contact sheet: {e}")
        return None


def create_contact_sheets(frames_dir: Path, output_dir: Path, batch_size: int = 16) -> List[Path]:
    """
    Create contact sheets from all frames in a directory.
    
    Args:
        frames_dir: Directory containing frame images
        output_dir: Directory to save contact sheets
        batch_size: Number of frames per contact sheet (default 16)
        
    Returns:
        List of paths to created contact sheets
    """
    if not HAS_CV2:
        print("⚠️ OpenCV not available - cannot create contact sheets")
        return []
    
    if not frames_dir.exists():
        print(f"⚠️ Frames directory does not exist: {frames_dir}")
        return []
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all frame files (jpg, png)
    frame_files = sorted([
        f for f in frames_dir.iterdir() 
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ])
    
    if not frame_files:
        print(f"⚠️ No frame files found in {frames_dir}")
        return []
    
    print(f"📸 Found {len(frame_files)} frames, creating contact sheets...")
    
    contact_sheets = []
    
    # Process in batches
    for batch_idx in range(0, len(frame_files), batch_size):
        batch_files = frame_files[batch_idx:batch_idx + batch_size]
        
        # Load frames
        frames = []
        for frame_file in batch_files:
            try:
                frame = cv2.imread(str(frame_file))
                if frame is not None:
                    frames.append(frame)
            except Exception as e:
                print(f"⚠️ Failed to load frame {frame_file}: {e}")
        
        if not frames:
            continue
        
        # Create contact sheet
        output_path = output_dir / f"contact_sheet_{batch_idx // batch_size:03d}.jpg"
        result = create_contact_sheet(frames, output_path)
        
        if result is not None:
            contact_sheets.append(output_path)
            print(f"  ✓ Contact sheet {len(contact_sheets)} created ({len(frames)} frames)")
    
    print(f"✅ Created {len(contact_sheets)} contact sheets")
    return contact_sheets


def extract_timestamp_from_filename(filename: str) -> Optional[str]:
    """
    Extract timestamp from frame filename.
    Filename format: frame_00001_t120_20240101.jpg -> 00:02:00
    
    Args:
        filename: The frame filename
        
    Returns:
        Formatted timestamp string [HH:MM:SS] or None
    """
    import re
    
    # Try to find _t<seconds> pattern
    match = re.search(r'_t(\d+)_', filename)
    if match:
        seconds = int(match.group(1))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    return None


def get_frame_index_from_filename(filename: str) -> int:
    """
    Extract frame index from filename.
    Filename format: frame_00001_t120_20240101.jpg -> 1
    
    Args:
        filename: The frame filename
        
    Returns:
        Frame index (1-based)
    """
    import re
    
    # Try to find frame_<number> pattern
    match = re.search(r'frame_(\d+)', filename)
    if match:
        return int(match.group(1))
    
    return 0


if __name__ == "__main__":
    # Test contact sheet creation
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python contact_sheet.py <frames_dir> [output_dir]")
        sys.exit(1)
    
    frames_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else frames_dir.parent / "contact_sheets"
    
    contact_sheets = create_contact_sheets(frames_dir, output_dir)
    print(f"\nCreated {len(contact_sheets)} contact sheets in {output_dir}")
