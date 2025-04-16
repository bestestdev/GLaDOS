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
        "--device", type=str, default=None, help="Direct camera device path (e.g., /dev/video0 or /dev/media0)"
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
    parser.add_argument(
        "--enable-display", action="store_true", help="Enable GUI display (not recommended for headless systems)"
    )
    parser.add_argument(
        "--rpi-camera", action="store_true", help="Force use of Picamera2 for Raspberry Pi camera"
    )
    args = parser.parse_args()

    # Set up logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Check for direct device access
    if args.device:
        # Try to determine camera ID from device name if possible
        camera_id = args.camera
        if args.device.startswith("/dev/video"):
            try:
                camera_id = int(args.device[len("/dev/video"):])
                logger.info(f"Using camera ID {camera_id} from device path {args.device}")
            except ValueError:
                logger.warning(f"Could not extract camera ID from {args.device}, using default {args.camera}")
        
        # Make sure the device exists
        if not os.path.exists(args.device):
            logger.error(f"Device {args.device} does not exist")
            return
            
        logger.info(f"Initializing camera with device {args.device} (ID: {camera_id})")
        camera = Camera(camera_id=camera_id, use_picamera=args.rpi_camera)
    else:
        # Standard camera initialization
        logger.info(f"Initializing camera with ID {args.camera}")
        camera = Camera(camera_id=args.camera, use_picamera=args.rpi_camera)

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

    # Initialize vision processor - disable display by default (headless mode)
    logger.info("Starting vision processor")
    processor = VisionProcessor(
        camera=camera,
        detector=detector,
        enable_display=args.enable_display,
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

    if args.enable_display:
        logger.info("Press ESC in the display window to exit")
    else:
        logger.info("Running in headless mode - press Ctrl+C to exit")
    
    try:
        # Main loop
        last_description_time = 0
        last_save_time = 0
        frame_count = 0
        
        while processor.is_running:
            # Get current frame and describe the scene
            frame = processor.get_current_frame()
            if frame is not None:
                # Print scene description for every frame
                description = processor.get_scene_description()
                logger.info(f"Scene: {description}")
                frame_count += 1
                
                # Save frames if requested
                if args.save_frames and time.time() - last_save_time >= 2.0:
                    frame_path = os.path.join(args.save_frames, f"frame_{frame_count:04d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    last_save_time = time.time()
            
            # Sleep to reduce CPU usage but remain responsive
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Clean up
        processor.stop()
        logger.info("Vision processor stopped")


if __name__ == "__main__":
    main() 