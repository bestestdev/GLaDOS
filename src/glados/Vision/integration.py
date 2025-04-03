"""Integration between GLaDOS engine and Vision module."""

import time
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Any

from loguru import logger

# Remove direct import of Glados to break circular dependency
# from ..engine import Glados

from .detector import Detection
from .processor import TrackedObject, VisionProcessor

# Use TYPE_CHECKING to allow type hints without runtime imports
if TYPE_CHECKING:
    from ..engine import Glados


class GladosVisionIntegration:
    """Integration between GLaDOS engine and Vision module.
    
    This class connects the Vision module with GLaDOS by:
    1. Setting up callbacks between the systems
    2. Converting visual information to natural language for GLaDOS
    3. Managing GLaDOS's responses to visual stimuli
    """
    
    # Cooldown time (seconds) between responses to the same object class
    RESPONSE_COOLDOWN = 20.0
    
    def __init__(
        self, 
        glados: "Glados", 
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
        
        # Special objects to respond to - these are just categories we'll generate responses for
        self.special_object_categories = [
            "person", "cat", "dog", "cake", "computer", "book", 
            "phone", "food", "bottle", "cup"
        ]
        
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
        scene_description = self.vision.get_scene_description()
        if scene_description:
            # Instead of directly speaking the scene description,
            # we'll ask the LLM to comment on what it sees
            self._send_llm_prompt(f"Announce what you currently see: {scene_description}")
    
    def _on_new_detection(self, detection: Detection) -> None:
        """Callback for when a new object is detected.
        
        Args:
            detection: The detected object
        """
        # Track seen classes
        self.detected_classes.add(detection.label)
        
        # Log the detection
        logger.debug(f"INTEGRATION: Detection callback for {detection.label}")
        
        current_time = time.time()
        last_response_time = self.class_last_response_time.get(detection.label, 0)
        
        if current_time - last_response_time >= self.RESPONSE_COOLDOWN:
            logger.info(f"INTEGRATION: Response cooldown passed for {detection.label}, proceeding with response")
            
            # Handle person detection
            if detection.label == "person" and self.person_greeting_enabled:
                logger.success(f"INTEGRATION: Person detected! Sending greeting prompt")
                self._send_llm_prompt(f"You see a person. Make a brief greeting.")
                self.class_last_response_time[detection.label] = current_time
            
            # Handle special objects
            elif (self.special_objects_responses and 
                  detection.label in self.special_object_categories and
                  detection.label != "person"):  # Already handled person above
                logger.success(f"INTEGRATION: Special object {detection.label} detected! Sending comment prompt")
                self._send_llm_prompt(f"You just noticed a {detection.label}. Make a brief comment about it.")
                self.class_last_response_time[detection.label] = current_time
        else:
            logger.debug(f"INTEGRATION: Response cooldown not passed for {detection.label} - {current_time - last_response_time:.1f}s elapsed of {self.RESPONSE_COOLDOWN}s")
    
    def _on_object_lost(self, tracked_obj: TrackedObject) -> None:
        """Callback for when a tracked object is lost.
        
        Args:
            tracked_obj: The tracked object that was lost
        """
        # Only respond to objects that were visible for a while (not false positives)
        if (tracked_obj.frames_visible > 10 and 
                tracked_obj.detection.label in self.focus_objects):
            self._send_llm_prompt(f"You no longer see the {tracked_obj.detection.label} that was there before. Make a brief comment about this.")
    
    def _send_llm_prompt(self, prompt: str) -> None:
        """Send a prompt to the LLM to generate a response.
        
        Instead of using predefined responses, this method asks the LLM to generate
        a contextual response based on the visual event.
        
        Args:
            prompt: Instruction for the LLM about what to respond to
        """
        # Add our prompt as a system message
        self.glados.messages.append({"role": "system", "content": prompt})
        
        # Use the existing processing flag to prevent interruptions during processing
        self.glados.processing = True
        
        # Use the existing LLM/TTS pipeline by adding an "empty" user message
        # This will trigger the standard LLM processing with our system prompt included
        self.glados.llm_queue.put("Respond to the visual context.")
        
        # Log what we're doing
        logger.info(f"Sent visual event prompt to LLM: {prompt}")
    
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
    
    def add_special_object_category(self, object_class: str) -> None:
        """Add a special object category to respond to.
        
        Args:
            object_class: Class name to respond to
        """
        if object_class not in self.special_object_categories:
            self.special_object_categories.append(object_class)
    
    def generate_visual_context(self) -> str:
        """Generate a visual context string that can be added to GLaDOS's context.
        
        This provides information about the visual scene to the language model.
        
        Returns:
            str: Visual context description
        """
        # Check if vision processor is running
        if not self.vision.is_running:
            logger.warning("INTEGRATION: Vision processor is not running when requesting visual context")
            return "Vision system is currently offline."
        
        # Debug the vision processor state
        logger.debug(f"INTEGRATION: Vision processor running: {self.vision.is_running}")
        
        # Get the scene analysis
        logger.debug("INTEGRATION: Calling analyze_scene to get current objects")
        class_counts = self.vision.analyze_scene()
        logger.debug(f"INTEGRATION: Scene analysis result: {class_counts}")
        
        # Get center object if any
        center_obj = self.vision.get_object_in_center()
        center_desc = ""
        if center_obj:
            center_desc = f"The {center_obj.label} is directly in front of me. "
            logger.debug(f"INTEGRATION: Center object detected: {center_obj.label}")
        else:
            logger.debug("INTEGRATION: No object detected in center of frame")
        
        # Format the context
        if not class_counts:
            logger.warning("INTEGRATION: No objects detected in visual context")
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
        
        final_context = f"{center_desc}{objects_desc}."
        logger.debug(f"INTEGRATION: Generated visual context: {final_context}")
        return final_context
    
    @property
    def is_running(self) -> bool:
        """Check if the vision integration is running.
        
        Returns:
            bool: True if running, False otherwise
        """
        return self.vision.is_running 