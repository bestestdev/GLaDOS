"""Object detection and recognition using YOLOv8 models."""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
# Import torch and ultralytics for PyTorch model support
import torch
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    
# Keep ONNX runtime for backward compatibility
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    
from loguru import logger
from numpy.typing import NDArray

from ..utils.resources import resource_path


@dataclass
class Detection:
    """Class representing a detected object."""
    class_id: int
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)


class ObjectDetector:
    """Object detection using YOLOv8 models.
    
    This class provides methods to detect objects in images using pre-trained
    models in either PyTorch (.pt) or ONNX format. It supports YOLOv8 architecture
    and can be used for both detection and tracking.
    """
    
    # Default parameters
    DEFAULT_MODEL_PATH = resource_path("models/vision/yolov8n.pt")  # Updated to .pt
    DEFAULT_LABELS_PATH = resource_path("models/vision/coco_labels.txt")
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    DEFAULT_NMS_THRESHOLD = 0.45
    DEFAULT_INPUT_SIZE = (640, 640)
    
    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        labels_path: Union[str, Path] = DEFAULT_LABELS_PATH,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
    ) -> None:
        """Initialize the object detector.
        
        Args:
            model_path: Path to the model file (.pt or .onnx)
            labels_path: Path to the text file containing class labels
            confidence_threshold: Minimum confidence score for detections
            nms_threshold: Non-maximum suppression threshold for filtering overlapping boxes
            input_size: Size of input images expected by the model (width, height)
        """
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.is_pt_model = str(self.model_path).endswith('.pt')
        
        # Ensure required files exist
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            # Initialize model based on file type
            if self.is_pt_model:
                if not ULTRALYTICS_AVAILABLE:
                    raise ImportError(
                        "Ultralytics package not found. "
                        "Install it with: pip install ultralytics"
                    )
                # Load PyTorch model using Ultralytics YOLO
                self.model = YOLO(str(self.model_path))
                logger.success(f"Loaded PyTorch model: {self.model_path}")
            else:
                if not ONNX_AVAILABLE:
                    raise ImportError(
                        "ONNX Runtime not found. "
                        "Install it with: pip install onnxruntime"
                    )
                # Initialize ONNX Runtime session
                self.session = ort.InferenceSession(
                    str(self.model_path), 
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                
                # Get model metadata
                self.input_name = self.session.get_inputs()[0].name
                self.output_names = [output.name for output in self.session.get_outputs()]
                logger.success(f"Loaded ONNX model: {self.model_path}")
            
            # Load class labels
            self.class_names = self._load_class_names()
            logger.success(f"Loaded {len(self.class_names)} class labels")
            
        except Exception as e:
            logger.error(f"Error initializing object detector: {e}")
            raise
    
    def _load_class_names(self) -> List[str]:
        """Load class names from a text file.
        
        Returns:
            List[str]: List of class names
        """
        if not os.path.exists(self.labels_path):
            logger.warning(f"Labels file not found: {self.labels_path}, using default labels")
            # Default to COCO dataset classes if file not found
            return [f"class_{i}" for i in range(80)]
        
        with open(self.labels_path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    
    def _preprocess(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Preprocess image for model input.
        
        Args:
            image: Input image in BGR format (OpenCV default)
            
        Returns:
            NDArray[np.float32]: Preprocessed image ready for model input
        """
        # For PyTorch models, preprocessing is handled by the YOLO class
        if self.is_pt_model:
            return image  # Return as is, YOLO class will handle preprocessing
            
        # ONNX model preprocessing
        # Resize image
        input_img = cv2.resize(image, self.input_size)
        
        # Convert to RGB and normalize
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.astype(np.float32) / 255.0
        
        # Change dimensions from HWC to NCHW (batch, channels, height, width)
        input_img = input_img.transpose(2, 0, 1)
        input_img = np.expand_dims(input_img, 0)
        
        return input_img
    
    def _postprocess_onnx(
        self, 
        outputs: Union[List[NDArray], Dict[str, NDArray]], 
        original_shape: Tuple[int, int]
    ) -> List[Detection]:
        """Process ONNX model output to get detections.
        
        Args:
            outputs: Raw model outputs
            original_shape: Original image shape (height, width)
            
        Returns:
            List[Detection]: List of Detection objects
        """
        # Convert dict output to list if necessary
        if isinstance(outputs, dict):
            outputs = list(outputs.values())
        
        # Get the first output (YOLOv8 has a single output)
        predictions = outputs[0]
        
        # For YOLOv8, the output shape is (batch, num_detections, num_classes + 4)
        # The first 4 columns are the bounding box coordinates (x, y, w, h)
        # The remaining columns are class probabilities
        
        detections = []
        orig_h, orig_w = original_shape
        input_h, input_w = self.input_size
        
        # Scale factors for converting normalized coordinates to original image coordinates
        scale_h, scale_w = orig_h / input_h, orig_w / input_w
        
        for pred in predictions:
            boxes = pred[:, :4]  # (x, y, w, h) - center_x, center_y, width, height
            scores = pred[:, 4:]  # class probabilities
            
            # Get class with highest probability and its score
            class_ids = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1)
            
            # Filter by confidence threshold
            mask = confidences >= self.confidence_threshold
            boxes, class_ids, confidences = boxes[mask], class_ids[mask], confidences[mask]
            
            if len(boxes) == 0:
                continue
            
            # Convert from (center_x, center_y, width, height) to (x1, y1, x2, y2)
            x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = (x - w/2) * scale_w
            y1 = (y - h/2) * scale_h
            x2 = (x + w/2) * scale_w
            y2 = (y + h/2) * scale_h
            
            # Non-maximum suppression
            indices = cv2.dnn.NMSBoxes(
                np.column_stack((x1, y1, x2-x1, y2-y1)).tolist(), 
                confidences.tolist(), 
                self.confidence_threshold, 
                self.nms_threshold
            )
            
            # Create detection objects
            for i in indices:
                detections.append(Detection(
                    class_id=int(class_ids[i]),
                    label=self.class_names[int(class_ids[i])] if class_ids[i] < len(self.class_names) else f"class_{class_ids[i]}",
                    confidence=float(confidences[i]),
                    bbox=(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]))
                ))
            
        return detections
    
    def _convert_yolo_results_to_detections(self, results, original_shape: Tuple[int, int]) -> List[Detection]:
        """Convert YOLO model results to Detection objects.
        
        Args:
            results: Results from YOLO model prediction
            original_shape: Original image shape (height, width)
            
        Returns:
            List[Detection]: List of Detection objects
        """
        detections = []
        
        # Process results - YOLO models return a Results object
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get box coordinates (already in x1,y1,x2,y2 format and scaled to image)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence and class
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                # Skip if below threshold
                if conf < self.confidence_threshold:
                    continue
                    
                # Get class name
                label = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                
                # Create detection
                detections.append(Detection(
                    class_id=cls_id,
                    label=label,
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2))
                ))
                
        return detections
    
    def detect(self, image: NDArray[np.uint8]) -> List[Detection]:
        """Detect objects in an image.
        
        Args:
            image: Input image in BGR format (OpenCV default)
            
        Returns:
            List[Detection]: List of detected objects
        """
        if image.size == 0:
            logger.warning("Empty image provided to detect()")
            return []
            
        try:
            start_time = time.time()
            
            # Different detection process based on model type
            if self.is_pt_model:
                # PyTorch model inference using YOLO
                results = self.model.predict(
                    image, 
                    conf=self.confidence_threshold,
                    verbose=False
                )
                detections = self._convert_yolo_results_to_detections(results, image.shape[:2])
            else:
                # ONNX model inference
                input_tensor = self._preprocess(image)
                outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
                detections = self._postprocess_onnx(outputs, image.shape[:2])
            
            inference_time = (time.time() - start_time) * 1000  # ms
            logger.debug(f"Detected {len(detections)} objects in {inference_time:.1f}ms")
            return detections
            
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return []
    
    def draw_detections(
        self, 
        image: NDArray[np.uint8], 
        detections: List[Detection], 
        draw_labels: bool = True
    ) -> NDArray[np.uint8]:
        """Draw detection bounding boxes on an image.
        
        Args:
            image: Input image
            detections: List of Detection objects
            draw_labels: Whether to draw class labels and confidence scores
            
        Returns:
            NDArray[np.uint8]: Image with drawn detections
        """
        result = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            # Generate a deterministic color based on class ID
            color = (
                (det.class_id * 50) % 255,
                (det.class_id * 100) % 255,
                (det.class_id * 150) % 255
            )
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw label if requested
            if draw_labels:
                label = f"{det.label}: {det.confidence:.2f}"
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Draw label background
                cv2.rectangle(
                    result, 
                    (x1, y1 - text_size[1] - 10), 
                    (x1 + text_size[0], y1), 
                    color, 
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    result, 
                    label, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1, 
                    cv2.LINE_AA
                )
        
        return result
    
    def detect_and_draw(
        self, 
        image: NDArray[np.uint8], 
        draw_labels: bool = True
    ) -> Tuple[NDArray[np.uint8], List[Detection]]:
        """Detect objects and draw them on the image.
        
        Args:
            image: Input image
            draw_labels: Whether to draw class labels
            
        Returns:
            Tuple[NDArray[np.uint8], List[Detection]]: Annotated image and list of detections
        """
        detections = self.detect(image)
        annotated_image = self.draw_detections(image, detections, draw_labels)
        return annotated_image, detections 