"""Demo script for the Vision module."""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from .camera import Camera
from .detector import ObjectDetector
from .processor import VisionProcessor


def main():
    """Run the Vision demo."""
    parser = argparse.ArgumentParser(description="GLaDOS Vision Demo")
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera ID to use (default: 0)"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Path to custom PyTorch model (.pt)"
    )
    parser.add_argument(
        "--labels", type=str, default=None, help="Path to custom labels file"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Detection confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--tracking", action="store_true", help="Enable object tracking"
    )
    parser.add_argument(
        "--save-frames", type=str, default=None, help="Directory to save frames (if provided)"
    )
    args = parser.parse_args()

    # Set up logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Initialize camera
    logger.info(f"Initializing camera with ID {args.camera}")
    camera = Camera(camera_id=args.camera)

    # Initialize detector with custom model if provided
    detector_args = {}
    if args.model:
        detector_args["model_path"] = args.model
    if args.labels:
        detector_args["labels_path"] = args.labels
    if args.confidence:
        detector_args["confidence_threshold"] = args.confidence

    logger.info("Initializing object detector")
    detector = ObjectDetector(**detector_args)

    # Initialize vision processor
    logger.info("Starting vision processor")
    processor = VisionProcessor(
        camera=camera,
        detector=detector,
        enable_display=True,
        enable_tracking=args.tracking
    )

    # Set up callbacks
    def on_new_detection(detection):
        logger.info(f"New detection: {detection.label} ({detection.confidence:.2f})")

    def on_object_lost(tracked_obj):
        logger.info(f"Lost object: {tracked_obj.detection.label} (visible for {tracked_obj.frames_visible} frames)")

    processor.add_new_detection_callback(on_new_detection)
    processor.add_object_lost_callback(on_object_lost)

    # Create save directory if needed
    if args.save_frames:
        save_dir = Path(args.save_frames)
        save_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Frames will be saved to {save_dir}")

    # Start processor
    if not processor.start():
        logger.error("Failed to start vision processor")
        return

    logger.info("Press ESC in the display window to exit")
    
    try:
        # Main loop
        last_description_time = 0
        last_save_time = 0
        frame_count = 0
        
        while processor.is_running:
            # Print scene description every 5 seconds
            current_time = time.time()
            if current_time - last_description_time >= 5.0:
                description = processor.get_scene_description()
                logger.info(f"Scene: {description}")
                last_description_time = current_time
                
            # Save frames if requested (every 2 seconds)
            if args.save_frames and current_time - last_save_time >= 2.0:
                frame = processor.get_current_frame()
                if frame is not None:
                    frame_path = os.path.join(args.save_frames, f"frame_{frame_count:04d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_count += 1
                    last_save_time = current_time
                    
            # Sleep to reduce CPU usage
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Clean up
        processor.stop()
        logger.info("Vision processor stopped")


if __name__ == "__main__":
    main() 