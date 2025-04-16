# GLaDOS Vision Module

This module provides vision capabilities for GLaDOS, allowing her to see and analyze her surroundings, track objects and people, and interact with the physical world.

## Components

The Vision module consists of three main components:

1. **Camera**: Provides an interface to access camera devices, capture frames, and manage the video stream.
2. **ObjectDetector**: Implements object detection and recognition using PyTorch models (YOLOv8).
3. **VisionProcessor**: Integrates the camera and detector, provides tracking capabilities, and handles scene analysis.

## Prerequisites

Before using the Vision module, make sure you have:

1. A working webcam or camera device connected to your system
2. Required dependencies installed (OpenCV, NumPy, PyTorch, Ultralytics)
3. Downloaded the required PyTorch models

## Installing Dependencies

You can install the Vision module and dependencies using the installation script with the `--vision` flag:

```bash
# Install GLaDOS with vision support
python scripts/install.py --vision
```

This will:
1. Install all necessary vision dependencies
2. Download the required PyTorch models automatically

If you prefer to install manually, you can use the vision extra:

```bash
# Manual installation with vision support
pip install -e ".[vision]"
```

## Model Setup

This module uses YOLOv8 PyTorch models for object detection. The default model is expected to be located at `models/vision/yolov8n.pt`, with class labels at `models/vision/coco_labels.txt`.

### Automatic Model Download

The easiest way to get the required models is to use the `--vision` flag with the install script:

```bash
# Install vision support and download models
python scripts/install.py --vision
```

If you've already installed the dependencies manually, you can run the download script directly:

```bash
# From the project root directory, after installing dependencies
python src/glados/Vision/download_models.py
```

### Manual Setup

If you prefer to set up the models manually:

1. Download a YOLOv8 PyTorch model (e.g., YOLOv8n) from the [Ultralytics GitHub repository](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)
2. Create a `models/vision/` directory in the resources path
3. Place the model file as `yolov8n.pt` and create a text file named `coco_labels.txt` containing the COCO class names

## Basic Usage

Here's a simple example of how to use the Vision module:

```python
from glados.Vision import Camera, ObjectDetector, VisionProcessor

# Initialize the vision processor with default settings
vision = VisionProcessor(enable_display=True)

# Start the vision processor
vision.start()

# Get the current scene description
description = vision.get_scene_description()
print(f"GLaDOS sees: {description}")

# Check if a specific object is visible
persons = vision.find_objects("person")
if persons:
    print(f"I see {len(persons)} people")

# Get the object at the center of the camera view
center_object = vision.get_object_in_center()
if center_object:
    print(f"There's a {center_object.label} in front of me")

# Clean up when done
vision.stop()
```

## Running the Demo

To test the Vision module, you can run the included demo script:

```bash
# Basic demo with default camera
python -m glados.Vision.demo

# Specify a different camera
python -m glados.Vision.demo --camera 1

# Enable object tracking
python -m glados.Vision.demo --tracking

# Save frames to a directory
python -m glados.Vision.demo --save-frames output/frames/

# Specify a custom model and labels
python -m glados.Vision.demo --model path/to/model.pt --labels path/to/labels.txt
```

## Integration with GLaDOS

The Vision module is designed to be integrated with the main GLaDOS system. It provides callbacks for detecting new objects or tracking object movement, which can be used to trigger appropriate responses from GLaDOS.

### Using the GladosVisionIntegration

The easiest way to integrate vision with GLaDOS is to use the provided integration class:

```python
from glados.engine import Glados
from glados.Vision import GladosVisionIntegration

# Initialize GLaDOS
glados = Glados.from_yaml("configs/glados_config.yaml")

# Create and configure the vision integration
vision_integration = GladosVisionIntegration(
    glados=glados,
    auto_describe_scene=True,
    scene_description_interval=60.0,  # Describe scene every 60 seconds
    person_greeting_enabled=True
)

# Start the integration
vision_integration.start()

# Set objects to focus on for lost-object notifications
vision_integration.set_focus_objects(["person", "cat", "dog"])

# Add a custom response for a specific object
vision_integration.add_special_object_response(
    "laptop", "I see you're using a laptop. What are you working on?"
)

# Start GLaDOS
glados.start_listen_event_loop()
```

### Manual Integration

You can also manually integrate with GLaDOS by setting up your own callbacks:

```python
from glados.engine import Glados
from glados.Vision import VisionProcessor

# Initialize GLaDOS
glados = Glados.from_yaml("configs/glados_config.yaml")

# Initialize vision
vision = VisionProcessor()

# Define callback for new detections
def on_new_object(detection):
    if detection.label == "person" and detection.confidence > 0.7:
        glados.tts_queue.put(f"I see a human. Hello there.")
    elif detection.label == "cat" and detection.confidence > 0.7:
        glados.tts_queue.put(f"I see a cat. I love cats.")

# Register callback
vision.add_new_detection_callback(on_new_object)

# Start both systems
vision.start()
glados.start_listen_event_loop()
```

## Advanced Features

### Object Tracking

The Vision module includes multi-object tracking capabilities. When tracking is enabled:

- Objects are assigned unique IDs that persist across frames
- The system can track objects even during brief occlusions
- You can receive callbacks when objects appear or disappear

### Saving Frames

You can save the current camera frame to disk:

```python
# Save the current frame
vision.save_current_frame("glados_view.jpg")
```

### Scene Analysis

Get a summary of all detected objects in the current view:

```python
# Get object counts by class
objects = vision.analyze_scene()
for class_name, count in objects.items():
    print(f"I see {count} {class_name}(s)")
```

## Dependencies

- OpenCV (cv2)
- NumPy
- PyTorch
- Ultralytics (for YOLOv8 models)
- Loguru (for logging)
- tqdm (for download progress)

## Troubleshooting

- If no camera is detected, try specifying a different camera ID
- If detections are inaccurate, consider using a different model or adjusting the confidence threshold
- For performance issues, try using a smaller model or lower resolution
- Make sure you have the correct version of PyTorch for your platform (CPU/CUDA) 