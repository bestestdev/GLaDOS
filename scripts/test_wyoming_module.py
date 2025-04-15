#!/usr/bin/env python3
"""Test script for Wyoming Whisper module implementation."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
import numpy as np
from loguru import logger

# Import our Wyoming implementation
from glados.Whisper.wyoming_whisper import WyomingTranscriber
from glados.utils.resources import resource_path


def main():
    """Test the Wyoming Whisper module."""
    
    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    
    # Load config
    with open("configs/alfred_config.yaml", "r") as f:
        config = yaml.safe_load(f)["Glados"]
    
    wyoming_url = config.get("wyoming_whisper_url")
    if not wyoming_url:
        logger.error("No wyoming_whisper_url found in config")
        return
    
    logger.info(f"Using Wyoming Whisper at {wyoming_url}")
    
    try:
        # Initialize transcriber
        transcriber = WyomingTranscriber(
            uri=f"tcp://{wyoming_url}", 
            language="en"
        )
        
        # Test with a sample audio file
        test_audio = resource_path("data/0.wav")
        logger.info(f"Transcribing {test_audio}")
        
        # Transcribe the file
        text = transcriber.transcribe_file(str(test_audio))
        
        # Print results
        logger.success(f"Transcription: {text}")
        
    except Exception as e:
        logger.error(f"Error testing Wyoming Whisper: {e}")


if __name__ == "__main__":
    main() 