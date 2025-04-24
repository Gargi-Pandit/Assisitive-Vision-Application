import cv2
import numpy as np
from PIL import Image
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    def __init__(self):
        self.debug_mode = True

    def capture_image(self):
        """
        Capture image from webcam
        """
        logger.info("Initializing camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Failed to open webcam")
            raise IOError("Cannot open webcam")
        
        # Add a small delay to allow camera to adjust
        cv2.waitKey(1000)
        
        logger.info("Capturing image...")
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logger.error("Failed to capture image")
            raise IOError("Cannot capture image")
        
        if self.debug_mode:
            self._display_debug_image(frame, "Captured Image")
        
        logger.info("Image captured successfully")
        return frame
    
    def preprocess_for_caption(self, image, target_size=(384, 384)):
        """
        Preprocess image for caption model
        """
        logger.info("Preprocessing image for captioning...")
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Resize and normalize
        image = image.resize(target_size)
        image = np.array(image)
        image = image / 255.0
        
        if self.debug_mode:
            self._display_debug_image(image, "Preprocessed for Captioning")
        
        return np.expand_dims(image, axis=0)
    
    def preprocess_for_vqa(self, image, target_size=(224, 224)):
        """
        Preprocess image for VQA model
        """
        logger.info("Preprocessing image for VQA...")
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Resize and normalize
        image = image.resize(target_size)
        image = np.array(image)
        image = image / 255.0
        
        if self.debug_mode:
            self._display_debug_image(image, "Preprocessed for VQA")
        
        return np.expand_dims(image, axis=0)
    
    def _display_debug_image(self, image, title):
        """
        Display debug image using matplotlib
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show(block=False)
        plt.pause(1)
        plt.close() 