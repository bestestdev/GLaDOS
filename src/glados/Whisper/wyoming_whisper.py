"""Wyoming Whisper ASR implementation."""

import asyncio
import socket
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import soundfile as sf  # type: ignore
from loguru import logger

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient, AsyncTcpClient
from wyoming.info import Describe
from wyoming.event import async_read_event, Event


class WyomingTranscriber:
    """A transcriber that uses a remote Wyoming Faster Whisper server."""

    def __init__(self, uri: str, language: str = "en") -> None:
        """
        Initialize a WyomingTranscriber with a remote Wyoming server.

        Parameters:
            uri (str): URI of the Wyoming server (e.g., 'tcp://192.168.1.100:10300')
            language (str, optional): Language code for transcription. Defaults to 'en'.
        """
        self.uri = uri
        self.language = language
        self._client: Optional[AsyncTcpClient] = None
        self._initialize_lock = asyncio.Lock()
        self._event_loop = asyncio.new_event_loop()
        
        # Patch AsyncClient if needed
        self._patch_async_client()
        
        # Initialize logging
        self._setup_logging()
        
    def _patch_async_client(self) -> None:
        """Patch the AsyncClient class to add is_connected method if it doesn't exist."""
        if not hasattr(AsyncClient, 'is_connected'):
            def is_connected(self) -> bool:
                """Check if the client is connected."""
                if not hasattr(self, '_writer') or self._writer is None:
                    return False
                return not self._writer.is_closing()
                
            # Add the method to the class
            setattr(AsyncClient, 'is_connected', is_connected)
            logger.debug("Patched AsyncClient with is_connected method")
            
    def _setup_logging(self) -> None:
        """Set up logging for the Wyoming client."""
        # Configure Wyoming logger to be less verbose
        logging.getLogger("wyoming").setLevel(logging.WARNING)
        
    async def _connect_with_timeout(self, host: str, port: int, timeout: float = 5.0) -> Optional[AsyncTcpClient]:
        """
        Connect to a Wyoming server with timeout.
        
        Parameters:
            host (str): Host name or IP address
            port (int): Port number
            timeout (float): Connection timeout in seconds
            
        Returns:
            Optional[AsyncTcpClient]: Client if connected, None if failed
        """
        try:
            # First check if the server is reachable
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            
            logger.debug(f"Attempting to connect to {host}:{port}")
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result != 0:
                logger.error(f"Failed to connect to {host}:{port}: {result} - {os.strerror(result) if result < 100 else 'Unknown error'}")
                return None
                
            # Create the actual Wyoming client
            logger.debug(f"Creating Wyoming client for {host}:{port}")
            client = AsyncTcpClient(host, port)
            
            # Ensure the client can actually communicate with the server
            try:
                reader_writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=timeout
                )
                writer = reader_writer[1]
                writer.close()
                await writer.wait_closed()
                logger.debug(f"Successfully opened test connection to {host}:{port}")
            except Exception as e:
                logger.error(f"Failed to establish test connection to {host}:{port}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return None
                
            return client
            
        except Exception as e:
            logger.error(f"Error connecting to {host}:{port}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    async def _initialize_client(self) -> bool:
        """
        Initialize the Wyoming client if not already initialized.
        
        Returns:
            bool: True if successful, False otherwise
        """
        async with self._initialize_lock:
            # If client exists, check if it's still connected
            if self._client is not None:
                try:
                    if not hasattr(self._client, "is_connected") or not self._client.is_connected():
                        logger.warning("Client exists but is not connected, recreating...")
                        self._client = None
                    else:
                        logger.debug("Client already initialized and connected")
                        return True
                except Exception as e:
                    logger.warning(f"Error checking client connection: {e}, recreating client")
                    self._client = None

            # Parse the URI
            if not self.uri.startswith("tcp://"):
                logger.error(f"Unsupported URI scheme: {self.uri}, must start with tcp://")
                return False
            
            try:
                # Parse host and port from URI
                host_port = self.uri.replace("tcp://", "")
                if ":" not in host_port:
                    logger.error(f"Invalid URI format: {self.uri}, expected tcp://host:port")
                    return False
                
                host, port_str = host_port.split(":")
                try:
                    port = int(port_str)
                except ValueError:
                    logger.error(f"Invalid port number: {port_str}")
                    return False
                
                logger.debug(f"Connecting to Wyoming server at {host}:{port}")
                
                # Test basic connectivity first
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(3.0)
                    result = sock.connect_ex((host, port))
                    sock.close()
                    
                    if result != 0:
                        logger.error(f"Cannot connect to {host}:{port}: {os.strerror(result) if result < 100 else 'Network error'}")
                        return False
                    
                    logger.debug(f"Basic TCP connection to {host}:{port} successful")
                except Exception as e:
                    logger.error(f"Socket connection failed: {e}")
                    return False
                
                # Create the Wyoming client
                client = AsyncClient.from_uri(self.uri)
                
                # Connect to the server with explicit timeout
                try:
                    logger.debug("Connecting to Wyoming server...")
                    await asyncio.wait_for(client.connect(), timeout=5.0)
                    logger.debug("Connection established")
                except asyncio.TimeoutError:
                    logger.error("Timeout connecting to Wyoming server")
                    return False
                except Exception as e:
                    logger.error(f"Failed to connect: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    return False
                
                # Store the client
                self._client = client
                
                # Test connection by getting server info
                try:
                    logger.debug("Testing connection with Describe event")
                    await asyncio.wait_for(self._client.write_event(Describe().event()), timeout=3.0)
                    
                    logger.debug("Waiting for server response")
                    event = await asyncio.wait_for(self._client.read_event(), timeout=3.0)
                    
                    if not event:
                        logger.error("No response from Wyoming server")
                        self._client = None
                        return False
                    
                    logger.debug(f"Server responded successfully: {event.type}")
                    return True
                    
                except asyncio.TimeoutError:
                    logger.error("Timeout waiting for server response to Describe event")
                    self._client = None
                    return False
                except Exception as e:
                    logger.error(f"Error testing server connection: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    self._client = None
                    return False
                
            except Exception as e:
                logger.error(f"Error during client initialization: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                self._client = None
                return False

    async def _transcribe_audio(self, audio: NDArray[np.float32], sample_rate: int = 16000) -> str:
        """
        Transcribe audio data using the Wyoming server.

        Parameters:
            audio (NDArray[np.float32]): Audio data as a numpy float32 array
            sample_rate (int, optional): Sample rate of the audio. Defaults to 16000.

        Returns:
            str: Transcribed text

        Raises:
            ConnectionError: If connection to Wyoming server fails
        """
        # Initialize client if needed
        success = await self._initialize_client()
        if not success:
            logger.error("Failed to initialize Wyoming client")
            return ""
            
        assert self._client is not None

        try:
            # Check if there's actual audio content to transcribe
            # Calculate energy and detect if there's likely speech
            audio_energy = np.mean(np.abs(audio))
            max_amplitude = np.max(np.abs(audio))
            
            if len(audio) < 1000 or audio_energy < 0.005 or max_amplitude < 0.05:
                logger.debug(f"Audio appears to be mostly silence or too short: len={len(audio)}, energy={audio_energy:.5f}, max={max_amplitude:.5f}")
                return ""

            # Log the audio characteristics
            logger.debug(f"Transcribing audio: {len(audio)} samples, energy: {audio_energy:.5f}, max amplitude: {max_amplitude:.5f}")

            # Convert audio to bytes (16-bit PCM)
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            logger.debug(f"Audio converted to bytes: {len(audio_bytes)} bytes")

            # Begin audio session
            logger.debug("Sending AudioStart event")
            audio_start = AudioStart(
                rate=sample_rate, width=2, channels=1
            ).event()
            await self._client.write_event(audio_start)

            # Set language for transcription
            logger.debug(f"Sending Transcribe event with language={self.language}")
            transcribe = Transcribe(language=self.language).event()
            await self._client.write_event(transcribe)

            # Send audio data (chunked to avoid excessive memory usage)
            chunk_size = 1024 * 16  # 16 KB chunks
            chunks_sent = 0
            logger.debug(f"Sending audio data in chunks of {chunk_size} bytes")
            
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                audio_chunk = AudioChunk(
                    rate=sample_rate, width=2, channels=1, audio=chunk
                ).event()
                await self._client.write_event(audio_chunk)
                chunks_sent += 1
                
            logger.debug(f"Sent {chunks_sent} audio chunks")

            # End audio session
            logger.debug("Sending AudioStop event")
            audio_stop = AudioStop().event()
            await self._client.write_event(audio_stop)
            
            # Check if client is still connected
            if not self._client or not self._client.is_connected():
                logger.error("Client disconnected during transcription")
                self._client = None
                return ""

            # Wait for transcription result with timeout
            try:
                logger.debug("Waiting for transcription result...")
                start_time = asyncio.get_event_loop().time()
                timeout = 6.0  # 6 seconds timeout - reduced from 10 seconds
                
                while True:
                    current_time = asyncio.get_event_loop().time()
                    remaining_time = timeout - (current_time - start_time)
                    
                    if remaining_time <= 0:
                        logger.error(f"Transcription timed out after {timeout} seconds")
                        return ""
                    
                    try:
                        logger.debug(f"Waiting for response event (timeout: {remaining_time:.1f}s)")
                        event = await asyncio.wait_for(self._client.read_event(), timeout=min(1.0, remaining_time))
                        
                        if not event:
                            logger.error("Received empty event")
                            break

                        logger.debug(f"Received event of type: {event.type}")
                        
                        if Transcript.is_type(event.type):
                            transcript = Transcript.from_event(event)
                            text = transcript.text.strip()
                            if text:
                                logger.success(f"Transcription result: '{text}'")
                                return text
                            else:
                                logger.warning("Received empty transcript text")
                                return ""
                            
                    except asyncio.TimeoutError:
                        logger.warning(f"No response received within timeout window, {remaining_time:.1f}s remaining")
                        # Continue the loop to try again if there's time left
                
                logger.error("No transcription received within timeout period")
                return ""
                    
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for transcription result")
                return ""

        except Exception as e:
            import traceback
            logger.error(f"Error during transcription: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Reset client on error
            self._client = None
            
        return ""  # Return empty string if no transcription is received

    def transcribe(self, audio: NDArray[np.float32]) -> str:
        """
        Transcribe an audio signal to text using the Wyoming server.

        Parameters:
            audio (NDArray[np.float32]): Input audio signal as a numpy float32 array.

        Returns:
            str: Transcribed text representation of the input audio.
        """
        try:
            # Check if audio has meaningful content before attempting transcription
            if len(audio) < 100 or np.max(np.abs(audio)) < 0.01:
                logger.debug("Audio appears to be mostly silence, skipping transcription")
                return ""
                
            # Force a clean disconnect after each transcription to avoid stuck connections
            if self._client is not None:
                try:
                    logger.debug("Disconnecting existing client before transcription")
                    self._event_loop.run_until_complete(self._client.disconnect())
                    self._client = None
                    # Small delay to allow for proper cleanup
                    self._event_loop.run_until_complete(asyncio.sleep(0.1))
                except Exception as e:
                    logger.warning(f"Error while disconnecting client: {e}")
                    self._client = None
            
            # Perform the transcription
            result = self._event_loop.run_until_complete(self._transcribe_audio(audio))
            
            # Always clean up the client after transcription to avoid connection issues
            if self._client is not None:
                try:
                    logger.debug("Disconnecting client after transcription")
                    self._event_loop.run_until_complete(self._client.disconnect())
                    self._client = None
                except Exception:
                    self._client = None
                    
            return result
        except Exception as e:
            import traceback
            logger.error(f"Error in Wyoming transcription: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._client = None
            return ""

    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribe an audio file to text.

        Parameters:
            audio_path (str): Path to the audio file to be transcribed.

        Returns:
            str: The transcribed text content of the audio file.

        Raises:
            FileNotFoundError: If the specified audio file does not exist.
            ValueError: If the audio file cannot be read or processed.
        """
        # Load audio
        audio, sr = sf.read(audio_path, dtype="float32")
        return self.transcribe(audio)
        
    def disconnect(self) -> None:
        """
        Explicitly disconnect from the Wyoming server and clean up resources.
        Should be called when the transcriber is no longer needed.
        """
        if self._client is not None:
            try:
                logger.debug("Manually disconnecting Wyoming client")
                self._event_loop.run_until_complete(self._client.disconnect())
            except Exception as e:
                logger.warning(f"Error disconnecting client: {e}")
            finally:
                self._client = None
                
    def __del__(self) -> None:
        """Clean up client and event loop on garbage collection."""
        # Try to disconnect the client if it exists
        if self._client is not None:
            try:
                if hasattr(self._event_loop, 'is_running') and not self._event_loop.is_running():
                    logger.debug("Cleaning up Wyoming client in __del__")
                    self._event_loop.run_until_complete(self._client.disconnect())
            except Exception as e:
                logger.warning(f"Error in __del__ while disconnecting client: {e}")
            finally:
                self._client = None
                
        # Close the event loop
        if self._event_loop is not None:
            try:
                # Stop the loop if it's running
                if hasattr(self._event_loop, 'is_running') and self._event_loop.is_running():
                    self._event_loop.stop()
                    
                # Close the loop
                if hasattr(self._event_loop, 'is_closed') and not self._event_loop.is_closed():
                    self._event_loop.close()
            except Exception as e:
                logger.warning(f"Error in __del__ while closing event loop: {e}") 