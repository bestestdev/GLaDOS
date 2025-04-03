"""Camera interface for capturing frames."""

import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from ..utils.resources import resource_path


class Camera:
    """Camera interface for capturing frames from a connected camera device.
    
    This class provides methods to start, stop, and retrieve frames from a camera.
    It uses OpenCV to handle the camera interface and provides both synchronous
    and asynchronous access to camera frames.
    """
    
    def __init__(
        self, 
        camera_id: int = 0, 
        width: int = 640, 
        height: int = 480, 
        fps: int = 30
    ) -> None:
        """Initialize the camera interface.
        
        Args:
            camera_id: ID of the camera device (default: 0, typically the first connected camera)
            width: Width of the captured frames in pixels
            height: Height of the captured frames in pixels
            fps: Target frames per second
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self._capture: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_frame: Optional[NDArray[np.uint8]] = None
        self._frame_lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._frame_callbacks: list[Callable[[NDArray[np.uint8]], None]] = []
        
    def start(self) -> bool:
        """Start the camera capture process.
        
        Returns:
            bool: True if camera started successfully, False otherwise
        """
        if self._running:
            logger.warning("Camera is already running")
            return True
            
        try:
            self._capture = cv2.VideoCapture(self.camera_id)
            if not self._capture.isOpened():
                logger.error(f"Failed to open camera with ID {self.camera_id}")
                return False
                
            # Set camera properties
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._capture.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Start capture thread
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            logger.success(f"Camera started with ID {self.camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
    
    def _capture_loop(self) -> None:
        """Main capture loop that runs in a separate thread."""
        if not self._capture:
            return
            
        while self._running:
            try:
                ret, frame = self._capture.read()
                if not ret:
                    logger.warning("Failed to capture frame, retrying...")
                    time.sleep(0.1)
                    continue
                    
                with self._frame_lock:
                    self._current_frame = frame
                    self._frame_ready.set()
                    
                # Notify callbacks
                for callback in self._frame_callbacks:
                    try:
                        callback(frame.copy())
                    except Exception as e:
                        logger.error(f"Error in frame callback: {e}")
                        
                # Small delay to control CPU usage
                time.sleep(1.0 / self.fps)
                
            except Exception as e:
                logger.error(f"Error in camera capture loop: {e}")
                time.sleep(0.1)
    
    def stop(self) -> None:
        """Stop the camera capture process."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            
        if self._capture:
            self._capture.release()
            self._capture = None
            
        self._current_frame = None
        logger.info("Camera stopped")
    
    def get_frame(self, timeout: float = 1.0) -> Optional[NDArray[np.uint8]]:
        """Get the latest frame from the camera.
        
        Args:
            timeout: Maximum time to wait for a frame in seconds
            
        Returns:
            Optional[NDArray[np.uint8]]: The captured frame as a numpy array, or None if not available
        """
        if not self._running:
            logger.warning("Camera is not running")
            return None
            
        # Wait for a frame to be ready
        if not self._frame_ready.wait(timeout):
            logger.warning("Timeout waiting for camera frame")
            return None
            
        # Get the current frame
        with self._frame_lock:
            self._frame_ready.clear()
            if self._current_frame is None:
                return None
            return self._current_frame.copy()
    
    def add_frame_callback(self, callback: Callable[[NDArray[np.uint8]], None]) -> None:
        """Add a callback function that will be called for each new frame.
        
        Args:
            callback: Function that takes a frame as a parameter
        """
        self._frame_callbacks.append(callback)
        
    def remove_frame_callback(self, callback: Callable[[NDArray[np.uint8]], None]) -> None:
        """Remove a previously added frame callback.
        
        Args:
            callback: The callback function to remove
        """
        if callback in self._frame_callbacks:
            self._frame_callbacks.remove(callback)
    
    @property
    def is_running(self) -> bool:
        """Check if the camera is currently running.
        
        Returns:
            bool: True if camera is running, False otherwise
        """
        return self._running and (self._thread is not None and self._thread.is_alive())
    
    @staticmethod
    def get_available_cameras(max_to_check: int = 5) -> list[int]:
        """Get a list of available camera IDs.
        
        Args:
            max_to_check: Maximum number of camera IDs to check
            
        Returns:
            list[int]: List of available camera IDs
        """
        available_cameras = []
        for i in range(max_to_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras
    
    @staticmethod
    def capture_test_image(camera_id: int = 0, save_path: Optional[str] = None) -> Tuple[bool, Optional[NDArray[np.uint8]]]:
        """Capture a single test image from the camera.
        
        Args:
            camera_id: ID of the camera to use
            save_path: If provided, save the image to this path
            
        Returns:
            Tuple[bool, Optional[NDArray[np.uint8]]]: Success flag and the captured image
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Could not open camera {camera_id}")
            return False, None
            
        # Allow camera to initialize
        time.sleep(0.5)
        
        # Capture frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logger.error("Failed to capture test image")
            return False, None
            
        # Save image if requested
        if save_path:
            try:
                cv2.imwrite(save_path, frame)
                logger.success(f"Test image saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save test image: {e}")
                
        return True, frame 