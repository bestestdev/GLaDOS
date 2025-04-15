#!/usr/bin/env python3
"""Test script for Wyoming Whisper implementation."""

import sys
import os
from pathlib import Path
import asyncio
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
import numpy as np
import soundfile as sf
from loguru import logger

# Wyoming imports
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.info import Describe


async def test_wyoming_whisper():
    """Test Wyoming Whisper with the config file."""
    
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
        # Connect to Wyoming server
        logger.debug(f"Connecting to Wyoming server at tcp://{wyoming_url}")
        client = await AsyncClient.connect(f"tcp://{wyoming_url}")
        
        if client is None:
            logger.error("Failed to create client")
            return
            
        logger.debug("Client connected successfully")
        
        # Get server info
        logger.debug("Sending Describe event")
        await client.write_event(Describe().event())
        
        logger.debug("Waiting for server info")
        event = await asyncio.wait_for(client.read_event(), timeout=5.0)
        
        if not event:
            logger.error(f"Failed to connect to Wyoming server at {wyoming_url}")
            return
        
        logger.info("Successfully connected to Wyoming server")
        
        # Test audio file
        test_audio_path = "src/glados/utils/resources/data/0.wav"
        if not os.path.exists(test_audio_path):
            logger.error(f"Test audio file not found: {test_audio_path}")
            return
            
        logger.debug(f"Loading audio file: {test_audio_path}")
        # Load audio file
        audio, sample_rate = sf.read(test_audio_path, dtype="float32")
        logger.debug(f"Audio loaded: {len(audio)} samples, sample rate: {sample_rate}")
        
        # Convert audio to bytes (16-bit PCM)
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        logger.debug(f"Audio converted to bytes: {len(audio_bytes)} bytes")
        
        # Begin audio session
        logger.debug("Sending AudioStart event")
        audio_start = AudioStart(
            rate=sample_rate, width=2, channels=1
        ).event()
        await client.write_event(audio_start)
        
        # Set language for transcription
        logger.debug("Sending Transcribe event with language=en")
        transcribe = Transcribe(language="en").event()
        await client.write_event(transcribe)
        
        # Send audio data (chunked to avoid excessive memory usage)
        chunk_size = 1024 * 16  # 16 KB chunks
        chunks_sent = 0
        logger.debug(f"Sending audio data in chunks of {chunk_size} bytes")
        
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            audio_chunk = AudioChunk(
                rate=sample_rate, width=2, channels=1, audio=chunk
            ).event()
            await client.write_event(audio_chunk)
            chunks_sent += 1
            
        logger.debug(f"Sent {chunks_sent} audio chunks")
            
        # End audio session
        logger.debug("Sending AudioStop event")
        audio_stop = AudioStop().event()
        await client.write_event(audio_stop)
        
        # Wait for transcription result
        logger.info("Waiting for transcription...")
        start_time = time.time()
        timeout = 10  # 10 seconds timeout
        
        while True:
            if time.time() - start_time > timeout:
                logger.error("Timed out waiting for transcription")
                break
                
            try:
                logger.debug("Waiting for response event")
                event = await asyncio.wait_for(client.read_event(), timeout=2.0)
                
                if not event:
                    logger.error("No event received")
                    break
                    
                logger.debug(f"Received event type: {event.type}")
                
                if Transcript.is_type(event.type):
                    transcript = Transcript.from_event(event)
                    logger.success(f"Transcription: {transcript.text}")
                    break
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for response")
                break
            
        # Disconnect from server
        logger.debug("Disconnecting from server")
        await client.disconnect()
        
    except Exception as e:
        import traceback
        logger.error(f"Error testing Wyoming Whisper: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(test_wyoming_whisper()) 