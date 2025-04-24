import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
from PIL import Image
from utils.audio import AudioHelper
from utils.preprocess import ImagePreprocessor
from transformers import (
    VisionEncoderDecoderModel, 
    ViTImageProcessor, 
    AutoTokenizer,
    ViltProcessor, 
    ViltForQuestionAnswering
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AssistiveVisionApp:
    def __init__(self):
        logger.info("Initializing Assistive Vision App...")
        self.audio_helper = AudioHelper()
        self.image_processor = ImagePreprocessor()
        
        # Load models
        logger.info("Loading models...")
        self.load_models()
        
        # Welcome message
        self.audio_helper.speak_text(
            "Welcome to Assistive Vision. Do you want a description or ask a question?"
        )

    def load_models(self):
        """
        Load the image captioning and VQA models
        """
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load image captioning model
        logger.info("Loading image captioning model...")
        self.caption_model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.caption_processor = ViTImageProcessor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.caption_tokenizer = AutoTokenizer.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.caption_model.to(self.device)
        
        # Load VQA model
        logger.info("Loading VQA model...")
        self.vqa_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.vqa_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.vqa_model.to(self.device)

    def generate_caption(self, image):
        """
        Generate caption for the image
        """
        logger.info("Generating image caption...")
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        pixel_values = self.caption_processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        generated_ids = self.caption_model.generate(
            pixel_values,
            max_length=50,
            num_beams=4,
            return_dict_in_generate=True
        )
        
        caption = self.caption_tokenizer.batch_decode(
            generated_ids.sequences, 
            skip_special_tokens=True
        )[0]
        
        logger.info(f"Generated caption: {caption}")
        return caption

    def answer_question(self, image, question):
        """
        Answer a question about the image
        """
        logger.info(f"Answering question: {question}")
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Prepare inputs
        inputs = self.vqa_processor(image, question, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate answer
        outputs = self.vqa_model(**inputs)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = self.vqa_model.config.id2label[idx]
        
        logger.info(f"Generated answer: {answer}")
        return answer

    def run(self):
        """
        Main application loop
        """
        logger.info("Starting main application loop...")
        while True:
            # Listen for mode selection
            mode = self.audio_helper.listen_for_speech()
            
            if mode is None:
                self.audio_helper.speak_text(
                    "I didn't catch that. Please say either 'description' or 'question'."
                )
                continue
            
            # Capture image
            try:
                image = self.image_processor.capture_image()
            except Exception as e:
                logger.error(f"Error capturing image: {e}")
                self.audio_helper.speak_text(
                    "Failed to capture image. Please try again."
                )
                continue

            if "description" in mode:
                # Generate and speak caption
                try:
                    caption = self.generate_caption(image)
                    self.audio_helper.speak_text(caption)
                except Exception as e:
                    logger.error(f"Error generating caption: {e}")
                    self.audio_helper.speak_text(
                        "Sorry, I had trouble generating a description. Please try again."
                    )
                
            elif "question" in mode:
                # Ask for the question
                self.audio_helper.speak_text(
                    "What would you like to know about the image?"
                )
                question = self.audio_helper.listen_for_speech()
                
                if question is None:
                    self.audio_helper.speak_text(
                        "I didn't catch your question. Please try again."
                    )
                    continue
                
                try:
                    answer = self.answer_question(image, question)
                    self.audio_helper.speak_text(answer)
                except Exception as e:
                    logger.error(f"Error answering question: {e}")
                    self.audio_helper.speak_text(
                        "Sorry, I had trouble answering your question. Please try again."
                    )
            
            else:
                self.audio_helper.speak_text(
                    "Please say either 'description' or 'question'."
                )
                continue
            
            # Ask if user wants to continue
            self.audio_helper.speak_text(
                "Would you like to try another image? Say yes or no."
            )
            response = self.audio_helper.listen_for_speech()
            
            if response and "no" in response:
                self.audio_helper.speak_text(
                    "Thank you for using Assistive Vision. Goodbye!"
                )
                break

if __name__ == "__main__":
    try:
        app = AssistiveVisionApp()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"An error occurred: {e}") 