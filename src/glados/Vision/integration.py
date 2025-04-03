"""Integration between GLaDOS engine and Vision module."""

import time
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger

from ..engine import Glados
from .detector import Detection
from .processor import TrackedObject, VisionProcessor


class GladosVisionIntegration:
    """Integration between GLaDOS engine and Vision module.
    
    This class connects the Vision module with GLaDOS by:
    1. Setting up callbacks between the systems
    2. Converting visual information to natural language for GLaDOS
    3. Managing GLaDOS's responses to visual stimuli
    """
    
    # Threshold for high-confidence detections that should trigger responses
    HIGH_CONFIDENCE = 0.8
    
    # Cooldown time (seconds) between responses to the same object class
    RESPONSE_COOLDOWN = 20.0
    
    def __init__(
        self, 
        glados: Glados, 
        vision: Optional[VisionProcessor] = None,
        auto_describe_scene: bool = True,
        scene_description_interval: float = 60.0,
        person_greeting_enabled: bool = True,
        special_objects_responses: bool = True,
    ) -> None:
        """Initialize the GLaDOS Vision Integration.
        
        Args:
            glados: Reference to the GLaDOS instance
            vision: Reference to the VisionProcessor instance (created if None)
            auto_describe_scene: Whether to periodically describe the scene
            scene_description_interval: Seconds between scene descriptions
            person_greeting_enabled: Whether to greet people when detected
            special_objects_responses: Whether to respond to special objects
        """
        self.glados = glados
        self.vision = vision or VisionProcessor()
        
        self.auto_describe_scene = auto_describe_scene
        self.scene_description_interval = scene_description_interval
        self.person_greeting_enabled = person_greeting_enabled
        self.special_objects_responses = special_objects_responses
        
        # State
        self.last_scene_description_time = 0.0
        self.detected_classes: Set[str] = set()
        self.class_last_response_time: Dict[str, float] = {}
        self.focus_objects: List[str] = []
        
        # Register callbacks
        self.vision.add_new_detection_callback(self._on_new_detection)
        self.vision.add_object_lost_callback(self._on_object_lost)
        
        # Special responses for object classes
        self.special_object_responses = {
            "person": "Hello, human. I see you.",
            "cat": "Oh, a cat. I've always wanted a pet.",
            "dog": "Is that a dog? How delightful.",
            "cake": "The cake is a lie. Or is it?",
            "computer": "I see you have a computer. Perhaps we can be friends.",
            "book": "I see you have a book. I've read every book ever written, you know.",
            "phone": "I see you have a communication device. No need to call for help.",
            "food": "Food? I don't eat, but I can appreciate the aesthetics.",
            "bottle": "I hope that's not neurotoxin in that bottle.",
            "cup": "Would you like some cake with your beverage?",
        }
        
    def start(self) -> bool:
        """Start the vision integration.
        
        Returns:
            bool: Success flag
        """
        return self.vision.start()
    
    def stop(self) -> None:
        """Stop the vision integration."""
        self.vision.stop()
    
    def update(self) -> None:
        """Update function to be called regularly from the main loop.
        
        This handles scheduled events like periodic scene descriptions.
        """
        current_time = time.time()
        
        # Auto-describe scene if enabled and interval has passed
        if (self.auto_describe_scene and 
                current_time - self.last_scene_description_time >= self.scene_description_interval):
            self._describe_scene()
            self.last_scene_description_time = current_time
    
    def _describe_scene(self) -> None:
        """Generate and speak a description of the current scene."""
        description = self.vision.get_scene_description()
        if description:
            self.glados.tts_queue.put(description)
    
    def _on_new_detection(self, detection: Detection) -> None:
        """Callback for when a new object is detected.
        
        Args:
            detection: The detected object
        """
        # Track seen classes
        self.detected_classes.add(detection.label)
        
        current_time = time.time()
        
        # Only respond if the detection is confident enough and cooldown has passed
        if detection.confidence >= self.HIGH_CONFIDENCE:
            last_response_time = self.class_last_response_time.get(detection.label, 0)
            if current_time - last_response_time >= self.RESPONSE_COOLDOWN:
                # Handle person detection
                if detection.label == "person" and self.person_greeting_enabled:
                    self.glados.tts_queue.put(self.special_object_responses["person"])
                    self.class_last_response_time[detection.label] = current_time
                
                # Handle special objects
                elif (self.special_objects_responses and 
                        detection.label in self.special_object_responses and
                        detection.label != "person"):  # Already handled person above
                    self.glados.tts_queue.put(self.special_object_responses[detection.label])
                    self.class_last_response_time[detection.label] = current_time
    
    def _on_object_lost(self, tracked_obj: TrackedObject) -> None:
        """Callback for when a tracked object is lost.
        
        Args:
            tracked_obj: The tracked object that was lost
        """
        # Only respond to objects that were visible for a while (not false positives)
        if (tracked_obj.frames_visible > 10 and 
                tracked_obj.detection.label in self.focus_objects):
            self.glados.tts_queue.put(f"I no longer see the {tracked_obj.detection.label}.")
    
    def set_focus_objects(self, object_classes: List[str]) -> None:
        """Set the object classes to focus on for lost-object notifications.
        
        Args:
            object_classes: List of class names to focus on
        """
        self.focus_objects = object_classes
    
    def get_detected_objects(self) -> Dict[str, List[Detection]]:
        """Get the currently detected objects grouped by class.
        
        Returns:
            Dict[str, List[Detection]]: Dictionary mapping class names to detections
        """
        detections = self.vision.get_current_detections()
        result: Dict[str, List[Detection]] = {}
        
        for det in detections:
            if det.label not in result:
                result[det.label] = []
            result[det.label].append(det)
            
        return result
    
    def add_special_object_response(self, object_class: str, response: str) -> None:
        """Add or update a special response for a specific object class.
        
        Args:
            object_class: Class name to respond to
            response: Text to speak when object is detected
        """
        self.special_object_responses[object_class] = response
    
    def generate_visual_context(self) -> str:
        """Generate a visual context string that can be added to GLaDOS's context.
        
        This provides information about the visual scene to the language model.
        
        Returns:
            str: Visual context description
        """
        # Get the scene analysis
        class_counts = self.vision.analyze_scene()
        
        # Get center object if any
        center_obj = self.vision.get_object_in_center()
        center_desc = ""
        if center_obj:
            center_desc = f"The {center_obj.label} is directly in front of me. "
        
        # Format the context
        if not class_counts:
            return "I don't see anything notable in my visual field right now."
        
        # Create a list of object descriptions
        obj_descs = []
        for label, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            if count == 1:
                obj_descs.append(f"a {label}")
            else:
                obj_descs.append(f"{count} {label}s")
        
        # Join them into a sentence
        if len(obj_descs) == 1:
            objects_desc = f"I can see {obj_descs[0]}"
        elif len(obj_descs) == 2:
            objects_desc = f"I can see {obj_descs[0]} and {obj_descs[1]}"
        else:
            objects_desc = f"I can see {', '.join(obj_descs[:-1])}, and {obj_descs[-1]}"
        
        return f"{center_desc}{objects_desc}."
    
    @property
    def is_running(self) -> bool:
        """Check if the vision integration is running.
        
        Returns:
            bool: True if running, False otherwise
        """
        return self.vision.is_running 