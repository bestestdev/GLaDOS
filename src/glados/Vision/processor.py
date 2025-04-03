"""Vision processor for GLaDOS."""

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from ..utils.resources import resource_path
from .camera import Camera
from .detector import Detection, ObjectDetector


@dataclass
class TrackedObject:
    """Class representing a tracked object across frames."""
    detection: Detection
    center: Tuple[int, int]  # (x, y) center point
    frames_visible: int  # Number of consecutive frames object has been visible
    last_seen: float  # Timestamp when last seen


class VisionProcessor:
    """Vision processor for GLaDOS.
    
    This class integrates camera input and object detection to provide
    vision capabilities for GLaDOS. It can track objects, detect faces,
    and analyze the environment for GLaDOS to interact with.
    """
    
    # Default tracking parameters
    DEFAULT_TRACKING_CONFIDENCE = 0.3  # Lower threshold for tracking than initial detection
    DEFAULT_MAX_FRAMES_MISSING = 10  # How many frames an object can be missing before removing tracking
    
    def __init__(
        self,
        camera: Optional[Camera] = None,
        detector: Optional[ObjectDetector] = None,
        enable_display: bool = False,
        enable_tracking: bool = True,
        tracking_confidence: float = DEFAULT_TRACKING_CONFIDENCE,
        object_classes_to_track: Optional[List[str]] = None,
    ) -> None:
        """Initialize the vision processor.
        
        Args:
            camera: Camera instance for capturing frames
            detector: ObjectDetector instance for object detection
            enable_display: Whether to display the camera feed with detections
            enable_tracking: Whether to enable object tracking
            tracking_confidence: Confidence threshold for tracking objects
            object_classes_to_track: List of class names to track, None for all classes
        """
        # Initialize camera if not provided
        self.camera = camera or Camera()
        
        # Initialize detector if not provided
        self.detector = detector or ObjectDetector()
        
        self.enable_display = enable_display
        self.enable_tracking = enable_tracking
        self.tracking_confidence = tracking_confidence
        self.object_classes_to_track = object_classes_to_track
        
        # Tracking state
        self.tracked_objects: Dict[int, TrackedObject] = {}  # ID -> TrackedObject
        self.next_track_id = 0
        self.tracking_lock = threading.Lock()
        
        # Processing state
        self.processing_thread: Optional[threading.Thread] = None
        self.running = False
        self.current_frame: Optional[NDArray[np.uint8]] = None
        self.current_detections: List[Detection] = []
        self.frame_lock = threading.Lock()
        
        # Callbacks
        self.on_new_detection_callbacks: List[Callable[[Detection], None]] = []
        self.on_object_lost_callbacks: List[Callable[[TrackedObject], None]] = []
        
        # Display window name
        self.window_name = "GLaDOS Vision"
    
    def start(self) -> bool:
        """Start the vision processor.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.running:
            logger.warning("Vision processor is already running")
            return True
            
        # Start camera
        if not self.camera.start():
            logger.error("Failed to start camera")
            return False
            
        # Start processing thread
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        if self.enable_display:
            # Create display window
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            
        logger.success("Vision processor started")
        return True
    
    def stop(self) -> None:
        """Stop the vision processor."""
        self.running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
            
        self.camera.stop()
        
        if self.enable_display:
            cv2.destroyWindow(self.window_name)
            
        logger.info("Vision processor stopped")
    
    def _processing_loop(self) -> None:
        """Main processing loop that runs in a separate thread."""
        last_fps_time = time.time()
        frame_count = 0
        
        while self.running:
            try:
                # Get frame from camera
                frame = self.camera.get_frame(timeout=0.1)
                if frame is None:
                    time.sleep(0.01)
                    continue
                    
                # Detect objects
                detections = self.detector.detect(frame)
                
                # Filter detections by class if specified
                if self.object_classes_to_track:
                    detections = [
                        det for det in detections 
                        if det.label in self.object_classes_to_track
                    ]
                
                # Update tracking
                if self.enable_tracking:
                    self._update_tracking(detections, frame)
                
                # Generate annotated frame
                annotated_frame = self._generate_annotated_frame(frame, detections)
                
                # Update current frame and detections
                with self.frame_lock:
                    self.current_frame = frame.copy()
                    self.current_detections = detections.copy()
                
                # Display frame if enabled
                if self.enable_display:
                    # Calculate FPS
                    frame_count += 1
                    current_time = time.time()
                    if current_time - last_fps_time >= 1.0:
                        fps = frame_count / (current_time - last_fps_time)
                        cv2.putText(
                            annotated_frame,
                            f"FPS: {fps:.1f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA
                        )
                        frame_count = 0
                        last_fps_time = current_time
                    
                    cv2.imshow(self.window_name, annotated_frame)
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC key
                        self.running = False
                
            except Exception as e:
                logger.error(f"Error in vision processing loop: {e}")
                time.sleep(0.1)
    
    def _update_tracking(self, detections: List[Detection], frame: NDArray[np.uint8]) -> None:
        """Update object tracking with new detections.
        
        Args:
            detections: List of new detections
            frame: Current frame
        """
        current_time = time.time()
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate centers for all detections
        detection_centers = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            detection_centers.append((center_x, center_y))
        
        with self.tracking_lock:
            # Match existing tracked objects with new detections
            matched_indices = set()
            
            # For each existing tracked object
            for track_id, tracked_obj in list(self.tracked_objects.items()):
                best_match_idx = -1
                best_match_distance = float('inf')
                track_center = tracked_obj.center
                
                # Find closest detection to this tracked object
                for i, (det, center) in enumerate(zip(detections, detection_centers)):
                    if i in matched_indices:
                        continue
                        
                    # Check if same class
                    if det.class_id != tracked_obj.detection.class_id:
                        continue
                        
                    # Calculate Euclidean distance between centers
                    distance = np.sqrt(
                        (center[0] - track_center[0])**2 +
                        (center[1] - track_center[1])**2
                    )
                    
                    # Use a threshold based on object size
                    x1, y1, x2, y2 = tracked_obj.detection.bbox
                    obj_size = max(x2 - x1, y2 - y1)
                    distance_threshold = obj_size * 0.5  # Threshold is 50% of object size
                    
                    if distance < distance_threshold and distance < best_match_distance:
                        best_match_distance = distance
                        best_match_idx = i
                
                # If match found, update tracked object
                if best_match_idx >= 0:
                    matched_det = detections[best_match_idx]
                    matched_center = detection_centers[best_match_idx]
                    matched_indices.add(best_match_idx)
                    
                    # Update tracked object
                    self.tracked_objects[track_id] = TrackedObject(
                        detection=matched_det,
                        center=matched_center,
                        frames_visible=tracked_obj.frames_visible + 1,
                        last_seen=current_time
                    )
                else:
                    # No match found, mark as missing
                    frames_missing = int((current_time - tracked_obj.last_seen) * 30)  # Estimate missing frames
                    
                    # Remove if missing for too long
                    if frames_missing > self.DEFAULT_MAX_FRAMES_MISSING:
                        # Notify object lost
                        for callback in self.on_object_lost_callbacks:
                            try:
                                callback(tracked_obj)
                            except Exception as e:
                                logger.error(f"Error in object lost callback: {e}")
                        
                        # Remove from tracking
                        del self.tracked_objects[track_id]
            
            # Add new tracked objects for unmatched detections
            for i, (det, center) in enumerate(zip(detections, detection_centers)):
                if i not in matched_indices:
                    # Use more strict confidence for new objects
                    if det.confidence >= self.tracking_confidence:
                        track_id = self.next_track_id
                        self.next_track_id += 1
                        
                        # Create new tracked object
                        tracked_obj = TrackedObject(
                            detection=det,
                            center=center,
                            frames_visible=1,
                            last_seen=current_time
                        )
                        
                        self.tracked_objects[track_id] = tracked_obj
                        
                        # Notify new detection
                        for callback in self.on_new_detection_callbacks:
                            try:
                                callback(det)
                            except Exception as e:
                                logger.error(f"Error in new detection callback: {e}")
    
    def _generate_annotated_frame(
        self, 
        frame: NDArray[np.uint8], 
        detections: List[Detection]
    ) -> NDArray[np.uint8]:
        """Generate an annotated frame with detections and tracking info.
        
        Args:
            frame: Original frame
            detections: List of detections
            
        Returns:
            NDArray[np.uint8]: Annotated frame
        """
        # Start with drawing standard detections
        result = self.detector.draw_detections(frame, detections)
        
        # Add tracking info if enabled
        if self.enable_tracking:
            with self.tracking_lock:
                for track_id, tracked_obj in self.tracked_objects.items():
                    x1, y1, x2, y2 = tracked_obj.detection.bbox
                    center_x, center_y = tracked_obj.center
                    
                    # Draw track ID
                    cv2.putText(
                        result,
                        f"ID: {track_id}",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA
                    )
                    
                    # Draw center point
                    cv2.circle(result, (center_x, center_y), 5, (0, 255, 255), -1)
        
        return result
    
    def get_current_frame(self) -> Optional[NDArray[np.uint8]]:
        """Get the current frame.
        
        Returns:
            Optional[NDArray[np.uint8]]: Current frame or None if not available
        """
        with self.frame_lock:
            if self.current_frame is None:
                return None
            return self.current_frame.copy()
    
    def get_current_detections(self) -> List[Detection]:
        """Get the current detections.
        
        Returns:
            List[Detection]: List of current detections
        """
        with self.frame_lock:
            return self.current_detections.copy()
    
    def get_object_in_center(
        self, 
        frame: Optional[NDArray[np.uint8]] = None, 
        center_threshold: float = 0.2
    ) -> Optional[Detection]:
        """Get the object closest to the center of the frame.
        
        Args:
            frame: Frame to use, if None uses current frame
            center_threshold: Threshold for considering an object in center (as fraction of frame size)
            
        Returns:
            Optional[Detection]: Detection of center object or None if none found
        """
        # Get frame if not provided
        if frame is None:
            frame = self.get_current_frame()
            if frame is None:
                return None
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        frame_center_x, frame_center_y = frame_width // 2, frame_height // 2
        
        # Get detections
        detections = self.get_current_detections()
        if not detections:
            return None
        
        # Find object closest to center
        closest_det = None
        min_distance = float('inf')
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            obj_center_x = (x1 + x2) // 2
            obj_center_y = (y1 + y2) // 2
            
            # Calculate distance to center
            distance = np.sqrt(
                (obj_center_x - frame_center_x)**2 +
                (obj_center_y - frame_center_y)**2
            )
            
            # Normalize by frame size
            normalized_distance = distance / np.sqrt(frame_width**2 + frame_height**2)
            
            # Check if within threshold and closer than previous best
            if normalized_distance < center_threshold and normalized_distance < min_distance:
                min_distance = normalized_distance
                closest_det = det
        
        return closest_det
    
    def find_objects(self, class_name: str) -> List[Detection]:
        """Find objects of a specific class in the current frame.
        
        Args:
            class_name: Class name to search for
            
        Returns:
            List[Detection]: List of detections matching the class
        """
        detections = self.get_current_detections()
        return [det for det in detections if det.label.lower() == class_name.lower()]
    
    def analyze_scene(self) -> Dict[str, int]:
        """Analyze the current scene to generate a summary of detected objects.
        
        Returns:
            Dict[str, int]: Dictionary of class names to counts
        """
        detections = self.get_current_detections()
        
        # Count objects by class
        class_counts: Dict[str, int] = {}
        for det in detections:
            class_counts[det.label] = class_counts.get(det.label, 0) + 1
            
        return class_counts
    
    def get_scene_description(self) -> str:
        """Generate a human-readable description of the current scene.
        
        Returns:
            str: Description of what's visible in the scene
        """
        class_counts = self.analyze_scene()
        
        if not class_counts:
            return "I don't see anything recognizable right now."
            
        # Format description
        descriptions = []
        for label, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            if count == 1:
                descriptions.append(f"a {label}")
            else:
                descriptions.append(f"{count} {label}s")
                
        if len(descriptions) == 1:
            return f"I can see {descriptions[0]}."
        elif len(descriptions) == 2:
            return f"I can see {descriptions[0]} and {descriptions[1]}."
        else:
            return f"I can see {', '.join(descriptions[:-1])}, and {descriptions[-1]}."
    
    def save_current_frame(self, file_path: Union[str, Path]) -> bool:
        """Save the current frame to a file.
        
        Args:
            file_path: Path where to save the image
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        frame = self.get_current_frame()
        if frame is None:
            logger.warning("No frame available to save")
            return False
            
        try:
            cv2.imwrite(str(file_path), frame)
            logger.success(f"Frame saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving frame: {e}")
            return False
    
    def add_new_detection_callback(self, callback: Callable[[Detection], None]) -> None:
        """Add a callback to be called when a new object is detected.
        
        Args:
            callback: Function to call with the new detection
        """
        self.on_new_detection_callbacks.append(callback)
        
    def add_object_lost_callback(self, callback: Callable[[TrackedObject], None]) -> None:
        """Add a callback to be called when a tracked object is lost.
        
        Args:
            callback: Function to call with the lost object
        """
        self.on_object_lost_callbacks.append(callback)
    
    @property
    def is_running(self) -> bool:
        """Check if the vision processor is running.
        
        Returns:
            bool: True if running, False otherwise
        """
        return self.running and self.camera.is_running 