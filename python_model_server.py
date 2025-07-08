from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchaudio
import numpy as np
# IMPORTANT: Import Wav2Vec2FeatureExtractor instead of Wav2Vec2Processor
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import librosa
import tempfile
import os
import json

app = Flask(__name__)
CORS(app)

class Wav2Vec2EmotionPredictor:
    def __init__(self, model_path='superb/wav2vec2-base-superb-er'):
        self.model_path = model_path
        self.model = None
        # --- CHANGE 1: Use feature_extractor instead of processor ---
        self.feature_extractor = None
        self.config = None
        self.emotion_labels = None
        self.load_model()
    
    def load_model(self):
        """Load the wav2vec2 emotion recognition model"""
        try:
            # --- CHANGE 2: Load the feature extractor directly ---
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_path)
            
            # Load the model
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()
            
            # Get config and labels from the loaded model
            self.config = self.model.config
            if hasattr(self.config, 'id2label'):
                self.emotion_labels = self.config.id2label
            else:
                print("Warning: 'id2label' not found in model config. Using default labels.")
                self.emotion_labels = {
                    '0': 'angry', '1': 'disgust', '2': 'fear', 
                    '3': 'happy', '4': 'neutral', '5': 'sad', '6': 'surprise'
                }

            print(f"Model loaded successfully!")
            print(f"Emotion labels: {self.emotion_labels}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def preprocess_audio(self, audio_path):
        """Preprocess audio file for wav2vec2 model"""
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=0)
            audio = audio / np.max(np.abs(audio))
            return audio, sr
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            raise e
    
    def predict_emotion(self, audio_path):
        """Predict emotion from audio file"""
        try:
            audio, sr = self.preprocess_audio(audio_path)
            
            # --- CHANGE 3: Use the feature_extractor to process audio ---
            inputs = self.feature_extractor(
                audio, 
                sampling_rate=sr, 
                return_tensors="pt", 
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_id = torch.argmax(probabilities, dim=-1).item()
            
            # Ensure the class ID is a string for dictionary lookup
            predicted_emotion = self.emotion_labels[str(predicted_class_id)]
            confidence = probabilities[0][predicted_class_id].item()
            
            emotion_probabilities = {}
            for i, prob in enumerate(probabilities[0]):
                emotion_name = self.emotion_labels[str(i)]
                emotion_probabilities[emotion_name] = float(prob.item())
            
            return {
                'emotion': predicted_emotion,
                'confidence': float(confidence),
                'probabilities': emotion_probabilities
            }
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            raise e

# --- Using a compatible model ---
MODEL_PATH = "superb/wav2vec2-base-superb-er"
predictor = Wav2Vec2EmotionPredictor(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict_emotion_endpoint(): # Renamed to avoid conflict with class method
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            audio_file.save(tmp_file.name)
            result = predictor.predict_emotion(tmp_file.name)
            os.unlink(tmp_file.name)
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in prediction endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_path': MODEL_PATH,
        'emotions': list(predictor.emotion_labels.values()) if predictor else []
    })

@app.route('/emotions', methods=['GET'])
def get_emotions():
    if predictor and predictor.emotion_labels:
        return jsonify({'emotions': list(predictor.emotion_labels.values())})
    return jsonify({'error': 'Model not loaded'}), 500

if __name__ == '__main__':
    print("Starting Wav2Vec2 Emotion Recognition Server...")
    print(f"Model path: {MODEL_PATH}")
    app.run(debug=True, host='0.0.0.0', port=5000)