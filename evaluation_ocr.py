import easyocr
import cv2
import os
import numpy as np
from jiwer import cer, wer
import re
from spellchecker import SpellChecker
from gtts import gTTS

# Paths
frames_folder = r"C:\Users\shravani\OneDrive\Desktop\some projects\Text video to speech\images"  # Ensure this path is correct
final_output_folder = r"C:\Users\shravani\OneDrive\Desktop\some projects\Text video to speech\static"  # Ensure this path is correct
ground_truth_text = "This is a sample text from the dataset."  # Update with the actual ground truth text

# Super-Resolution Function
def apply_super_resolution(image):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("EDSR_x4.pb")  # Use a pre-trained model
    sr.setModel("edsr", 4)
    print("Model loaded successfully")
    return sr.upsample(image)

# Function to preprocess and run OCR
def ocr_and_evaluate(image_path):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist.")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open or find the image {image_path}.")
        return
    
    # Apply Super-Resolution
    image = apply_super_resolution(image)
    
    # Convert to grayscale and preprocess
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.fastNlMeansDenoising(image, None, 30, 7, 21)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Resize and enhance contrast
    height, width = image.shape
    image = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    
    # OCR Processing
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image)
    extracted_text = " ".join([res[1] for res in result])
    
    # Spell Check
    spell = SpellChecker()
    words = extracted_text.split()
    corrected_words = [spell.correction(word) for word in words]
    corrected_text = " ".join(word if word is not None else '' for word in corrected_words)
    
    # Clean text
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", corrected_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    
    # Accuracy Metrics
    cer_value = cer(ground_truth_text, cleaned_text)
    wer_value = wer(ground_truth_text, cleaned_text)
    
    # Save Processed Image
    final_image_path = os.path.join(final_output_folder, "final_image.png")
    cv2.imwrite(final_image_path, image)
    
    # Convert Text to Speech
    tts = gTTS(cleaned_text, lang='en')
    tts.save(os.path.join(final_output_folder, "output_audio.mp3"))
    
    # Print Results
    print("\n--- OCR Accuracy Report ---")
    print(f"Extracted Text: {extracted_text.strip()}")
    print(f"Corrected Text: {corrected_text.strip()}")
    print(f"Cleaned Text: {cleaned_text.strip()}")
    print(f"Ground Truth: {ground_truth_text.strip()}")
    print(f"CER: {cer_value:.2%}")
    print(f"WER: {wer_value:.2%}")
    print(f"Processed image saved at: {final_image_path}")
    print(f"Audio file saved at: {os.path.join(final_output_folder, 'output_audio.mp3')}")

# Process First Frame
frame_path = r"C:\Users\shravani\OneDrive\Desktop\some projects\Text video to speech\images\1.jpg"  # Update with correct file name
ocr_and_evaluate(frame_path)
