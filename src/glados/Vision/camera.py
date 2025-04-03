"""Camera interface for capturing frames."""

import threading
import time
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from ..utils.resources import resource_path

# Conditionally import picamera2 if available
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False


class Camera:
    """Camera interface for capturing frames from a connected camera device.
    
    This class provides methods to start, stop, and retrieve frames from a camera.
    It uses OpenCV to handle the camera interface and provides both synchronous
    and asynchronous access to camera frames. On Raspberry Pi, it can also use
    the picamera2 library for better compatibility with Raspberry Pi cameras.
    """
    
    def __init__(
        self, 
        camera_id: int = 0, 
        width: int = 640, 
        height: int = 480, 
        fps: int = 30,
        use_picamera: Optional[bool] = None
    ) -> None:
        """Initialize the camera interface.
        
        Args:
            camera_id: ID of the camera device (default: 0, typically the first connected camera)
            width: Width of the captured frames in pixels
            height: Height of the captured frames in pixels
            fps: Target frames per second
            use_picamera: Whether to use picamera2 on Raspberry Pi (auto-detect if None)
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
        
        # For libcamera process-based approach
        self._libcamera_process: Optional[subprocess.Popen] = None
        self._libcamera_temp_dir: Optional[str] = None
        
        # For picamera2 approach
        self._picamera: Optional[Any] = None
        
        # Check if we're on a Raspberry Pi
        self._is_raspberry_pi = self._check_if_raspberry_pi()
        
        # Determine if we should use picamera2
        if use_picamera is None:
            self._use_picamera = self._is_raspberry_pi and PICAMERA2_AVAILABLE
        else:
            self._use_picamera = use_picamera and PICAMERA2_AVAILABLE
            
        if self._use_picamera:
            logger.info("Using picamera2 for Raspberry Pi camera")
        
    def _check_if_raspberry_pi(self) -> bool:
        """Check if we're running on a Raspberry Pi."""
        try:
            if os.path.exists('/proc/device-tree/model'):
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read()
                return 'Raspberry Pi' in model
            return False
        except:
            return False
    
    def _start_picamera(self) -> bool:
        """Start capture using the picamera2 library."""
        if not PICAMERA2_AVAILABLE:
            logger.warning("picamera2 is not available")
            return False
            
        try:
            # Initialize Picamera2
            self._picamera = Picamera2()
            
            # Configure camera
            config = self._picamera.create_still_configuration(
                main={"size": (self.width, self.height)},
                lores={"size": (640, 480)},
                display="lores"
            )
            self._picamera.configure(config)
            
            # Start camera
            self._picamera.start()
            
            logger.success("Started picamera2 for Raspberry Pi camera")
            
            # Start capture thread
            self._running = True
            self._thread = threading.Thread(target=self._picamera_capture_loop, daemon=True)
            self._thread.start()
            
            return True
        except Exception as e:
            logger.error(f"Failed to start picamera2: {e}")
            if self._picamera:
                try:
                    self._picamera.close()
                except:
                    pass
                self._picamera = None
            return False
    
    def _picamera_capture_loop(self) -> None:
        """Capture loop for picamera2."""
        if not self._picamera:
            logger.error("Picamera2 not initialized")
            return
            
        capture_interval = 1.0 / self.fps
        last_capture_time = 0
        
        while self._running:
            try:
                current_time = time.time()
                
                # Check if it's time to capture a new frame
                if current_time - last_capture_time >= capture_interval:
                    # Capture frame
                    frame = self._picamera.capture_array("main")
                    
                    # Convert to BGR (OpenCV format) if needed
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        # Check if we need to convert from RGB to BGR
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Update current frame
                    with self._frame_lock:
                        self._current_frame = frame
                        self._frame_ready.set()
                        
                    # Notify callbacks
                    for callback in self._frame_callbacks:
                        try:
                            callback(frame.copy())
                        except Exception as e:
                            logger.error(f"Error in frame callback: {e}")
                    
                    # Update last capture time
                    last_capture_time = current_time
                
                # Sleep to reduce CPU usage
                sleep_time = max(0.001, capture_interval / 2)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in picamera2 capture loop: {e}")
                time.sleep(0.1)
                
    def _start_libcamera_direct(self) -> bool:
        """Start the camera using direct libcamera for Raspberry Pi."""
        try:
            # Create a temporary directory for capturing frames
            self._libcamera_temp_dir = tempfile.mkdtemp(prefix="glados_vision_")
            logger.info(f"Created temporary directory for libcamera frames: {self._libcamera_temp_dir}")
            
            # Check which media devices are available
            self._media_devices = []
            try:
                # Get list of media devices
                for entry in os.listdir('/dev'):
                    if entry.startswith('media'):
                        self._media_devices.append(f"/dev/{entry}")
                
                if self._media_devices:
                    logger.info(f"Found media devices: {self._media_devices}")
            except Exception as e:
                logger.warning(f"Error checking media devices: {e}")
            
            # Check if libcamera-still is available
            result = subprocess.run(['which', 'libcamera-still'], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
            if result.returncode != 0:
                logger.warning("libcamera-still not found, cannot use direct libcamera")
                return False
            
            # Start capture thread
            self._running = True
            self._thread = threading.Thread(target=self._libcamera_still_capture_loop, daemon=True)
            self._thread.start()
            
            logger.success(f"Started libcamera-still direct capture with camera ID {self.camera_id}")
            return True
        except Exception as e:
            logger.error(f"Error starting libcamera direct capture: {e}")
            if self._libcamera_temp_dir and os.path.exists(self._libcamera_temp_dir):
                try:
                    os.rmdir(self._libcamera_temp_dir)
                except:
                    pass
                self._libcamera_temp_dir = None
            return False
    
    def _libcamera_still_capture_loop(self) -> None:
        """Capture loop using libcamera-still to directly capture frames."""
        if not self._libcamera_temp_dir:
            logger.error("Temporary directory for libcamera not set")
            return
        
        # Set up frame capture parameters
        frame_interval = 1.0 / min(self.fps, 2.5)  # Max 2.5 fps for still captures, adjust as needed
        last_capture_time = 0
        last_log_time = 0  # Track when we last logged a successful frame
        frame_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        try:
            while self._running:
                current_time = time.time()
                
                # Check if it's time to capture a new frame based on our target frame rate
                if current_time - last_capture_time >= frame_interval:
                    output_path = os.path.join(self._libcamera_temp_dir, f"frame_{frame_count:04d}.jpg")
                    
                    # Build the libcamera-still command
                    cmd = [
                        "libcamera-still",
                        "-n",                        # No preview
                        "-t", "100",                 # Short timeout (ms)
                        "--width", str(self.width),
                        "--height", str(self.height),
                        "--nopreview",               # Disable preview
                        "--immediate",               # Capture immediately
                        "--camera", "0",             # Use camera 0
                        "--output", output_path      # Output file
                    ]
                    
                    # Run the command with a timeout
                    try:
                        logger.debug(f"Capturing frame with: {' '.join(cmd)}")
                        result = subprocess.run(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=1.0  # Timeout after 1 second
                        )
                        
                        # Check if the capture was successful
                        if result.returncode == 0 and os.path.exists(output_path):
                            # Read the captured image
                            frame = cv2.imread(output_path)
                            
                            if frame is not None:
                                # Set the frame_ready event before updating current_frame
                                # This ensures that waiters are notified after the frame is updated
                                with self._frame_lock:
                                    self._current_frame = frame
                                    self._frame_ready.set()
                                    
                                # Notify callbacks
                                for callback in self._frame_callbacks:
                                    try:
                                        callback(frame.copy())
                                    except Exception as e:
                                        logger.error(f"Error in frame callback: {e}")
                                
                                # Clean up the file
                                try:
                                    os.unlink(output_path)
                                except Exception as e:
                                    logger.warning(f"Failed to delete temporary file {output_path}: {e}")
                                
                                # Update counter and time
                                frame_count += 1
                                last_capture_time = current_time
                                consecutive_failures = 0  # Reset failure counter
                                
                                # Log success only once every 60 seconds
                                if current_time - last_log_time >= 60.0:
                                    logger.success(f"Successfully captured frame {frame_count}")
                                    last_log_time = current_time
                            else:
                                logger.warning(f"Failed to read captured frame from {output_path}")
                                consecutive_failures += 1
                        else:
                            # Capture failed
                            err_output = result.stderr.decode('utf-8', errors='ignore')
                            if err_output:
                                logger.warning(f"libcamera-still failed with error: {err_output}")
                            else:
                                logger.warning(f"libcamera-still failed with return code {result.returncode}")
                            consecutive_failures += 1
                            
                    except subprocess.TimeoutExpired:
                        logger.warning("libcamera-still command timed out")
                        consecutive_failures += 1
                    except Exception as e:
                        logger.error(f"Error running libcamera-still: {e}")
                        consecutive_failures += 1
                    
                    # Check if we've had too many consecutive failures
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"Too many consecutive failures ({consecutive_failures}), stopping capture")
                        break
                
                # Calculate how long to sleep to maintain the desired frame rate
                elapsed = time.time() - current_time
                sleep_time = max(0.001, frame_interval - elapsed)
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Error in libcamera capture loop: {e}")

    def _start_libcamera_still(self) -> bool:
        """Start the camera using libcamera-still for Raspberry Pi."""
        logger.warning("libcamera-still approach is deprecated and disabled")
        return False
    
    def _try_libcamera_still_media(self, output_path: str, media_device_path: str) -> bool:
        """Try capturing with libcamera-still specifying the media device."""
        logger.warning("libcamera-still approach is deprecated and disabled")
        return False
            
    def _try_libcamera_still(self, output_path: str) -> bool:
        """Try capturing with standard libcamera-still."""
        logger.warning("libcamera-still approach is deprecated and disabled")
        return False
            
    def _try_raspistill(self, output_path: str) -> bool:
        """Try capturing with raspistill as a fallback."""
        logger.warning("raspistill approach is deprecated and disabled")
        return False
    
    def start(self) -> bool:
        """Start the camera capture process.
        
        Returns:
            bool: True if camera started successfully, False otherwise
        """
        if self._running:
            logger.warning("Camera is already running")
            return True
        
        # If picamera2 is available and we should use it, try that first
        if self._use_picamera:
            logger.info("Attempting to use picamera2 for Raspberry Pi camera")
            if self._start_picamera():
                return True
            else:
                logger.warning("picamera2 failed, falling back to alternatives")
        
        # For Raspberry Pi, try libcamera-vid approach first
        if self._is_raspberry_pi:
            logger.info("Attempting to use libcamera-vid for Raspberry Pi camera")
            if self._start_libcamera_direct():
                return True
            else:
                logger.warning("libcamera-vid approach failed, falling back to V4L2")
        
        # For Raspberry Pi, try multiple approaches starting with the best ones
        if self._is_raspberry_pi:
            # Check if media devices are available (newer RPi camera systems)
            media_devices_exist = False
            try:
                media_devices = [f for f in os.listdir('/dev') if f.startswith('media')]
                media_devices_exist = len(media_devices) > 0
                if media_devices_exist:
                    logger.info(f"Media devices found: {media_devices}")
            except:
                pass
            
            # Try different camera IDs for V4L2 approach on RPi
            logger.info(f"Attempting to use V4L2 with OpenCV for Raspberry Pi camera")
            
            # For RPi, priority camera IDs are often 0-7 when media devices exist
            camera_ids_to_try = [self.camera_id]
            if self.camera_id not in [0, 2, 4, 6] and media_devices_exist:
                camera_ids_to_try.extend([0, 2, 4, 6])
            
            for cam_id in camera_ids_to_try:
                device_path = f"/dev/video{cam_id}"
                if os.path.exists(device_path):
                    logger.info(f"Trying camera device: {device_path}")
                    
                    # Try a few different configurations for this device
                    for api_pref in [cv2.CAP_V4L2, cv2.CAP_ANY]:
                        try:
                            if api_pref == cv2.CAP_ANY:
                                self._capture = cv2.VideoCapture(cam_id)
                            else:
                                self._capture = cv2.VideoCapture(cam_id, api_pref)
                                
                            if not self._capture.isOpened():
                                logger.warning(f"Failed to open camera ID {cam_id} with API {api_pref}")
                                self._capture = None
                                continue
                            
                            # Try different formats - YUV420 then MJPG
                            formats_to_try = [
                                (ord('Y'), ord('U'), ord('1'), ord('2')),  # YUV420 format (YU12)
                                (ord('Y'), ord('U'), ord('Y'), ord('V')),  # YUYV format
                                (ord('M'), ord('J'), ord('P'), ord('G'))   # MJPG format
                            ]
                            
                            for format_chars in formats_to_try:
                                logger.info(f"Trying format: {chr(format_chars[0])}{chr(format_chars[1])}{chr(format_chars[2])}{chr(format_chars[3])}")
                                self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*format_chars))
                                
                                # Set camera properties
                                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                                self._capture.set(cv2.CAP_PROP_FPS, self.fps)
                                
                                # Try to read a test frame
                                for attempt in range(3):
                                    ret, frame = self._capture.read()
                                    if ret and frame is not None and frame.size > 0:
                                        # Found a working configuration
                                        self.camera_id = cam_id
                                        fourcc_str = ''.join([chr(c) for c in format_chars])
                                        logger.success(f"Successfully initialized camera ID {cam_id} with API {api_pref} using format {fourcc_str}")
                                        
                                        # Start capture thread
                                        self._running = True
                                        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
                                        self._thread.start()
                                        return True
                                    time.sleep(0.1)
                        
                            # If we got here, this configuration didn't work
                            if self._capture:
                                self._capture.release()
                                self._capture = None
                                
                        except Exception as e:
                            logger.warning(f"Error trying camera ID {cam_id} with API {api_pref}: {e}")
                            if self._capture:
                                self._capture.release()
                                self._capture = None
        
        # Fallback to standard OpenCV approach
        try:
            logger.info(f"Using standard OpenCV camera interface with ID {self.camera_id}")
            self._capture = cv2.VideoCapture(self.camera_id)
                
            if not self._capture.isOpened():
                # Try a few alternative camera IDs if the specified one didn't work
                alternative_ids = [0, 1, 2, 4] if self.camera_id not in [0, 1, 2, 4] else [5, 6, 7, 8]
                for alt_id in alternative_ids:
                    logger.info(f"Trying alternative camera ID: {alt_id}")
                    self._capture = cv2.VideoCapture(alt_id)
                    if self._capture.isOpened():
                        self.camera_id = alt_id
                        logger.success(f"Successfully opened camera with alternative ID {alt_id}")
                        break
                
            if not self._capture.isOpened():
                logger.error(f"Failed to open camera with ID {self.camera_id} and alternatives")
                return False
            
            # Try to set YUV420 format with proper fourcc
            try:
                self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', '1', '2'))
            except Exception as e:
                logger.warning(f"Failed to set YUV420 format, using default: {e}")
                
            # Set camera properties
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._capture.set(cv2.CAP_PROP_FPS, self.fps)
            
            # For some camera interfaces, we may need to read a few frames to initialize
            for _ in range(5):
                ret, _ = self._capture.read()
                if ret:
                    break
                time.sleep(0.1)
            
            # Start capture thread
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            logger.success(f"Camera started with ID {self.camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
    
    def _check_libcamera_available(self) -> bool:
        """Check if libcamera tools are available."""
        try:
            result = subprocess.run(['which', 'libcamera-vid'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
            return result.returncode == 0
        except:
            return False
    
    def _capture_loop(self) -> None:
        """Main capture loop that runs in a separate thread."""
        if not self._capture:
            return
        
        consecutive_failures = 0
        max_consecutive_failures = 50  # After this many failures, we'll try to reinitialize
            
        while self._running:
            try:
                ret, frame = self._capture.read()
                if not ret:
                    logger.warning("Failed to capture frame, retrying...")
                    consecutive_failures += 1
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"Too many consecutive failures ({consecutive_failures}), attempting to reinitialize camera")
                        # Release and recreate capture
                        self._capture.release()
                        time.sleep(0.5)
                        self._capture = cv2.VideoCapture(self.camera_id)
                        
                        if not self._capture.isOpened():
                            logger.error("Failed to reinitialize camera")
                            time.sleep(1.0)
                        else:
                            # Set camera properties again
                            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                            self._capture.set(cv2.CAP_PROP_FPS, self.fps)
                            logger.info("Camera reinitialized")
                            consecutive_failures = 0
                            
                    time.sleep(0.1)
                    continue
                
                # Reset failure counter on success
                consecutive_failures = 0
                    
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
                consecutive_failures += 1
                time.sleep(0.1)
    
    def stop(self) -> None:
        """Stop the camera capture process."""
        self._running = False
        
        # Stop thread
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        
        # Clean up picamera2 if used
        if self._picamera:
            try:
                self._picamera.close()
            except Exception as e:
                logger.error(f"Error closing picamera2: {e}")
            self._picamera = None
        
        # Clean up OpenCV capture if used
        if self._capture:
            self._capture.release()
            self._capture = None
            
        # Clean up libcamera temp directory if used
        if self._libcamera_temp_dir and os.path.exists(self._libcamera_temp_dir):
            try:
                # Remove any remaining files
                for file in os.listdir(self._libcamera_temp_dir):
                    file_path = os.path.join(self._libcamera_temp_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        logger.warning(f"Error removing file {file_path}: {e}")
                        
                # Remove the directory
                os.rmdir(self._libcamera_temp_dir)
                logger.info(f"Removed temporary directory {self._libcamera_temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up libcamera temporary directory: {e}")
            self._libcamera_temp_dir = None
            
        self._current_frame = None
        logger.info("Camera stopped")
    
    def get_frame(self, timeout: float = 5.0) -> Optional[NDArray[np.uint8]]:
        """Get the latest frame from the camera.
        
        Args:
            timeout: Maximum time to wait for a frame in seconds
            
        Returns:
            Optional[NDArray[np.uint8]]: The captured frame as a numpy array, or None if not available
        """
        if not self._running:
            logger.warning("Camera is not running")
            return None
            
        # For libcamera approach, give it more time for the first frame
        if self._is_raspberry_pi and timeout == 1.0:
            timeout = 5.0  # Use a longer timeout on Raspberry Pi
            
        start_time = time.time()
        
        # Wait for a frame to be ready
        if not self._frame_ready.wait(timeout):
            # Only log a warning if no frame has been captured yet
            if time.time() - start_time >= timeout:
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
    def get_available_cameras(max_to_check: int = 10) -> list[int]:
        """Get a list of available camera IDs.
        
        Args:
            max_to_check: Maximum number of camera IDs to check
            
        Returns:
            list[int]: List of available camera IDs
        """
        available_cameras = []
        
        # First check if we're on a Raspberry Pi
        is_raspberry_pi = False
        try:
            if os.path.exists('/proc/device-tree/model'):
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read()
                is_raspberry_pi = 'Raspberry Pi' in model
        except:
            pass
            
        # Check for Picamera2 on Raspberry Pi
        if is_raspberry_pi and PICAMERA2_AVAILABLE:
            try:
                # Get list of cameras from picamera2
                cameras = Picamera2.global_camera_info()
                if cameras:
                    return list(range(len(cameras)))
            except Exception as e:
                logger.warning(f"Failed to get camera info from picamera2: {e}")
            
        # On Raspberry Pi, try to use v4l2-ctl to find camera devices
        if is_raspberry_pi:
            try:
                # Try to use v4l2-ctl to list devices
                result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE,
                                       text=True)
                if result.returncode == 0:
                    # Parse the output to find camera devices
                    current_device = None
                    for line in result.stdout.splitlines():
                        if ':' in line:  # This is a device category line
                            current_device = line.strip()
                        elif '/dev/video' in line and current_device:
                            # Extract video device number
                            try:
                                video_dev = line.strip()
                                device_id = int(video_dev.replace('/dev/video', ''))
                                # Prioritize CSI/camera devices (like rp1-cfe)
                                if 'csi' in current_device.lower() or 'cam' in current_device.lower():
                                    # Put these at the beginning
                                    available_cameras.insert(0, device_id)
                                else:
                                    available_cameras.append(device_id)
                            except ValueError:
                                pass
                    
                    if available_cameras:
                        # Return a reasonably sized subset if we found many devices
                        return available_cameras[:max_to_check]
            except Exception as e:
                logger.warning(f"Failed to get camera devices using v4l2-ctl: {e}")
            
            # Try using libcamera to list cameras
            try:
                result = subprocess.run(['libcamera-still', '--list-cameras'], 
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE,
                                        text=True)
                if result.returncode == 0:
                    # Parse the output to find camera IDs
                    cameras = []
                    for line in result.stdout.splitlines():
                        if "camera" in line.lower() and ":" in line:
                            try:
                                # Extract camera ID
                                camera_id = int(line.split(":")[0].strip().split()[-1])
                                cameras.append(camera_id)
                            except:
                                pass
                    if cameras:
                        return cameras
            except:
                pass
                
            # Fall back to checking all device files
            # Check for /dev/videoX devices
            video_devices = []
            try:
                # Get a list of all video devices
                for entry in os.listdir('/dev'):
                    if entry.startswith('video'):
                        try:
                            video_id = int(entry[5:])
                            video_devices.append(video_id)
                        except ValueError:
                            pass
                
                # Try to prioritize likely camera devices
                # For RPi with media device layout, video0-video7 are often the main cameras
                main_cameras = [id for id in video_devices if 0 <= id <= 7]
                other_cameras = [id for id in video_devices if id > 7]
                
                # Sort by priority: main cameras first, then others
                available_cameras = main_cameras + other_cameras
                
                if available_cameras:
                    return available_cameras[:max_to_check]
            except Exception as e:
                logger.warning(f"Error enumerating video devices: {e}")
            
            # Also check if /dev/media* exists (for newer RPI systems)
            try:
                media_devices = []
                for entry in os.listdir('/dev'):
                    if entry.startswith('media'):
                        try:
                            # If there are media devices, we should prioritize video devices 0-7
                            media_id = int(entry[5:])
                            media_devices.append(media_id)
                        except ValueError:
                            pass
                
                if media_devices and not available_cameras:
                    # If we found media devices but no video devices yet, 
                    # return the first few video device IDs as a best guess
                    return list(range(min(8, max_to_check)))
            except:
                pass
                
        # Standard OpenCV camera check (as last resort)
        if not available_cameras:
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
        # Check if we're on a Raspberry Pi
        is_raspberry_pi = False
        try:
            if os.path.exists('/proc/device-tree/model'):
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read()
                is_raspberry_pi = 'Raspberry Pi' in model
        except:
            pass
            
        # Try using picamera2 first if on Raspberry Pi
        if is_raspberry_pi and PICAMERA2_AVAILABLE:
            try:
                # Initialize camera
                picam2 = Picamera2()
                
                # Configure camera
                config = picam2.create_still_configuration()
                picam2.configure(config)
                
                # Start camera
                picam2.start()
                
                # Wait for camera to initialize
                time.sleep(0.5)
                
                # Capture image
                frame = picam2.capture_array()
                
                # Convert to BGR (OpenCV format) if needed
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Save image if requested
                if save_path:
                    cv2.imwrite(save_path, frame)
                    logger.success(f"Test image saved to {save_path} using picamera2")
                
                # Clean up
                picam2.close()
                
                return True, frame
            except Exception as e:
                logger.warning(f"Failed to capture test image with picamera2: {e}")

        # Try using libcamera-jpeg for Raspberry Pi cameras
        if is_raspberry_pi:
            try:
                # Check if libcamera-jpeg is available
                check_result = subprocess.run(['which', 'libcamera-jpeg'], 
                                             stdout=subprocess.PIPE, 
                                             stderr=subprocess.PIPE)
                if check_result.returncode == 0:
                    # Use a temporary file if no save path provided
                    temp_file = None
                    output_path = save_path
                    if not output_path:
                        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                        output_path = temp_file.name
                        temp_file.close()
                    
                    # Capture image with libcamera-jpeg
                    cmd = [
                        "libcamera-jpeg",
                        "-n",                # No preview
                        "-t", "1000",        # Timeout (ms)
                        "--width", "640",
                        "--height", "480",
                        "-o", output_path    # Output file
                    ]
                    
                    result = subprocess.run(cmd, timeout=5)
                    if result.returncode == 0 and os.path.exists(output_path):
                        # Read the image
                        frame = cv2.imread(output_path)
                        
                        # Remove temporary file if we created one
                        if temp_file and not save_path:
                            try:
                                os.unlink(output_path)
                            except:
                                pass
                            
                        if frame is not None:
                            return True, frame
            except Exception as e:
                logger.warning(f"Failed to capture test image with libcamera-jpeg: {e}")
            
        # Fall back to OpenCV approach
        # First try with V4L2 backend for Raspberry Pi
        cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        if not cap.isOpened():
            # Fall back to default backend
            cap = cv2.VideoCapture(camera_id)
            
        if not cap.isOpened():
            logger.error(f"Could not open camera {camera_id}")
            return False, None
        
        # Try to set YUV420 format first
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', '1', '2'))
        except Exception as e:
            logger.warning(f"Failed to set YUV420 format, using default: {e}")
            
        # Allow camera to initialize
        time.sleep(0.5)
        
        # Try reading a few frames to ensure we get a good one
        ret = False
        frame = None
        for _ in range(5):
            ret, frame = cap.read()
            if ret:
                break
            time.sleep(0.1)
            
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

    def _libcamera_capture_loop(self) -> None:
        """Capture loop using libcamera-still to directly capture frames."""
        logger.warning("libcamera capture loop is deprecated and disabled")
        # Just return since this method should not be in use
        return 