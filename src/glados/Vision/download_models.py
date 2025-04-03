"""Script to download required models for the Vision module."""

import os
import shutil
import sys
from pathlib import Path
from typing import Optional, List

import requests
from loguru import logger
from tqdm import tqdm

from ..utils.resources import resource_path


# YOLOv8n model URLs - updated to use PyTorch (.pt) file
# Primary URL from confirmed working release
YOLOV8N_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
# Fallback URLs in case the primary one fails
FALLBACK_URLS = [
    "https://github.com/ultralytics/ultralytics/releases/download/v0.0.0/yolov8n.pt"
]

# COCO dataset labels
COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def download_file(url: str, destination: Path, desc: Optional[str] = None, fallback_urls: Optional[List[str]] = None) -> bool:
    """Download a file with progress bar.
    
    Args:
        url: URL to download from
        destination: Local path to save the file
        desc: Description for the progress bar
        fallback_urls: List of fallback URLs to try if the primary URL fails
        
    Returns:
        bool: Success flag
    """
    urls_to_try = [url] + (fallback_urls or [])
    
    for current_url in urls_to_try:
        try:
            logger.info(f"Attempting to download from {current_url}")
            response = requests.get(current_url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get("content-length", 0))
            block_size = 8192  # 8KB
            
            # Create directory if it doesn't exist
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            desc = desc or f"Downloading {destination.name}"
            
            with open(destination, "wb") as f, tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=desc,
                ascii=True,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            
            # If we got here, download was successful
            logger.success(f"Successfully downloaded from {current_url}")
            return True
            
        except Exception as e:
            logger.warning(f"Error downloading from {current_url}: {e}")
            if destination.exists():
                destination.unlink()
            # Continue to next URL if this one failed
    
    # If we get here, all URLs failed
    logger.error("All download attempts failed")
    return False


def create_labels_file(labels: list[str], destination: Path) -> bool:
    """Create a labels file.
    
    Args:
        labels: List of label strings
        destination: Path to save the labels file
        
    Returns:
        bool: Success flag
    """
    try:
        # Create directory if it doesn't exist
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with open(destination, "w") as f:
            for label in labels:
                f.write(f"{label}\n")
                
        return True
    except Exception as e:
        logger.error(f"Error creating labels file: {e}")
        return False


def download_vision_models() -> bool:
    """Download the required models for the Vision module.
    
    Returns:
        bool: Success flag
    """
    models_dir = resource_path("models/vision")
    model_path = models_dir / "yolov8n.pt"  # Updated to .pt extension
    labels_path = models_dir / "coco_labels.txt"
    
    # Create directory
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if model already exists
    if model_path.exists():
        logger.info(f"Model already exists at {model_path}")
    else:
        logger.info(f"Downloading YOLOv8n model to {model_path}")
        if not download_file(YOLOV8N_URL, model_path, "Downloading YOLOv8n model", FALLBACK_URLS):
            return False
            
    # Check if labels file already exists
    if labels_path.exists():
        logger.info(f"Labels file already exists at {labels_path}")
    else:
        logger.info(f"Creating COCO labels file at {labels_path}")
        if not create_labels_file(COCO_LABELS, labels_path):
            return False
            
    return True


def main():
    """Main entry point for the model download script."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    logger.info("Downloading required models for GLaDOS Vision module")
    
    if download_vision_models():
        logger.success("All models downloaded successfully")
    else:
        logger.error("Failed to download all models")
        sys.exit(1)


if __name__ == "__main__":
    main() 