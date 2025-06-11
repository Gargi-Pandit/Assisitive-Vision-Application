import speech_recognition as sr
from gtts import gTTS
import pygame
import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioHelper:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        pygame.mixer.init()  # Initialize pygame mixer
        
        # Adjust for ambient noise
        with self.mic as source:
            logger.info("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source)
            logger.info("Ambient noise adjustment complete")
    
    def listen_for_speech(self, timeout=5):
        """
        Listen for user speech and convert to text
        """
        logger.info("Listening for speech...")
        with self.mic as source:
            try:
                audio = self.recognizer.listen(source, timeout=timeout)
                text = self.recognizer.recognize_google(audio)
                logger.info(f"Recognized speech: {text}")
                return text.lower()
            except sr.WaitTimeoutError:
                logger.warning("No speech detected within timeout period")
                return None
            except sr.UnknownValueError:
                logger.warning("Speech was unintelligible")
                return None
            except sr.RequestError as e:
                logger.error(f"Could not request results from speech recognition service: {e}")
                return None

    def speak_text(self, text):
        """
        Convert text to speech and play it using pygame
        """
        logger.info(f"Converting to speech: {text}")
        try:
            # Create temporary file with timestamp to avoid conflicts
            temp_file = f"temp_speech_{int(time.time())}.mp3"
            tts = gTTS(text=text, lang='en')
            tts.save(temp_file)
            
            logger.info("Playing audio...")
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Wait for audio to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Clean up
            pygame.mixer.music.unload()
            os.remove(temp_file)
            logger.info("Audio playback complete")
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            raise 