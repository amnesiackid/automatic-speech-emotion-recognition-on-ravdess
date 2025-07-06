#!/usr/bin/env python3
"""
Enhanced Local Emotion Recognition Inference Script with Path Debugging
Test your trained Wav2Vec2 emotion model with your own audio files
"""

import os
import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

# You'll need to install these if not already installed:
# pip install transformers torch torchaudio librosa soundfile

from transformers import (
    Wav2Vec2Config,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Processor,
)

# ===== MODEL DEFINITION (Same as training) =====
@dataclass
class SpeechClassifierOutput:
    """Custom output format for speech classification"""
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    def __getitem__(self, key):
        """Make the output subscriptable for HuggingFace Trainer compatibility"""
        if key == 0:
            return self.loss
        elif key == 1:
            return self.logits
        elif key == 2:
            return self.hidden_states
        elif key == 3:
            return self.attentions
        else:
            # Handle slice objects gracefully
            if isinstance(key, slice):
                items = [self.loss, self.logits, self.hidden_states, self.attentions]
                return tuple(items[key])
            raise IndexError(f"Index {key} is out of range")

    def __len__(self):
        """Return the number of elements for tuple-like behavior"""
        return 4

class Wav2Vec2ClassificationHead(nn.Module):
    """Classification head for Wav2Vec2"""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        
    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    """Wav2Vec2 model for speech emotion classification"""
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = getattr(config, 'pooling_mode', 'mean')
        
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)
        
        self.init_weights()
    
    def freeze_feature_extractor(self):
        """Freeze the feature extractor to reduce training parameters"""
        self.wav2vec2.feature_extractor._freeze_parameters()
    
    def _pool_features(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pool features from sequence to fixed-size representation"""
        if self.pooling_mode == "mean":
            return torch.mean(hidden_states, dim=1)
        elif self.pooling_mode == "max":
            return torch.max(hidden_states, dim=1)[0]
        elif self.pooling_mode == "sum":
            return torch.sum(hidden_states, dim=1)
        else:
            raise ValueError(f"Unsupported pooling mode: {self.pooling_mode}")
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SpeechClassifierOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward pass through Wav2Vec2
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Get hidden states and pool
        hidden_states = outputs[0]
        pooled_output = self._pool_features(hidden_states)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# ===== PATH UTILITIES =====
def debug_path(path_str: str, path_type: str = "file") -> Tuple[bool, str]:
    """Debug path issues and provide detailed information"""
    print(f"\n🔍 Debugging {path_type} path: '{path_str}'")
    
    # Convert to Path object
    path = Path(path_str)
    
    # Check if path string is empty
    if not path_str.strip():
        return False, "Empty path provided"
    
    # Print path information
    print(f"   Raw path: {path_str}")
    print(f"   Resolved path: {path.resolve()}")
    print(f"   Is absolute: {path.is_absolute()}")
    print(f"   Current working directory: {os.getcwd()}")
    
    # Check existence
    if not path.exists():
        print(f"   ❌ Path does not exist")
        
        # Check parent directory
        parent = path.parent
        if parent.exists():
            print(f"   📁 Parent directory exists: {parent}")
            print(f"   📋 Contents of parent directory:")
            try:
                for item in parent.iterdir():
                    print(f"      - {item.name}")
            except PermissionError:
                print(f"      ❌ Permission denied to list directory")
        else:
            print(f"   ❌ Parent directory does not exist: {parent}")
        
        return False, f"Path does not exist: {path.resolve()}"
    
    # Check if it's the expected type
    if path_type == "file":
        if not path.is_file():
            if path.is_dir():
                return False, f"Path is a directory, not a file: {path.resolve()}"
            else:
                return False, f"Path exists but is not a file: {path.resolve()}"
    elif path_type == "directory":
        if not path.is_dir():
            if path.is_file():
                return False, f"Path is a file, not a directory: {path.resolve()}"
            else:
                return False, f"Path exists but is not a directory: {path.resolve()}"
    
    print(f"   ✅ Path is valid")
    return True, f"Valid {path_type}: {path.resolve()}"

def find_files_with_extensions(directory: str, extensions: List[str]) -> List[Path]:
    """Find files with specific extensions in directory and subdirectories"""
    directory = Path(directory)
    files = []
    
    print(f"\n🔍 Searching for files with extensions: {extensions}")
    print(f"   In directory: {directory.resolve()}")
    
    try:
        for ext in extensions:
            # Search in current directory
            pattern = f"*{ext}"
            current_files = list(directory.glob(pattern))
            files.extend(current_files)
            print(f"   Found {len(current_files)} files with extension {ext} in current directory")
            
            # Search in subdirectories
            recursive_pattern = f"**/*{ext}"
            recursive_files = list(directory.glob(recursive_pattern))
            # Remove duplicates (files already found in current directory)
            new_files = [f for f in recursive_files if f not in current_files]
            files.extend(new_files)
            print(f"   Found {len(new_files)} additional files with extension {ext} in subdirectories")
        
        # Remove duplicates
        files = list(set(files))
        print(f"   Total unique files found: {len(files)}")
        
        return files
        
    except Exception as e:
        print(f"   ❌ Error searching directory: {e}")
        return []

def get_path_with_autocomplete(prompt: str, path_type: str = "file") -> str:
    """Get path from user with basic autocomplete suggestions"""
    while True:
        path_str = input(f"\n{prompt}: ").strip()
        
        # Remove quotes if present
        if path_str.startswith('"') and path_str.endswith('"'):
            path_str = path_str[1:-1]
        elif path_str.startswith("'") and path_str.endswith("'"):
            path_str = path_str[1:-1]
        
        # Handle special shortcuts
        if path_str == ".":
            path_str = os.getcwd()
        elif path_str.startswith("~/"):
            path_str = os.path.expanduser(path_str)
        
        # Debug the path
        is_valid, message = debug_path(path_str, path_type)
        
        if is_valid:
            return path_str
        else:
            print(f"❌ {message}")
            
            # Offer some suggestions
            path = Path(path_str)
            if path.parent.exists():
                print(f"\n💡 Suggestions from parent directory '{path.parent}':")
                try:
                    items = list(path.parent.iterdir())
                    if path_type == "file":
                        suggestions = [item for item in items if item.is_file()][:10]
                    else:
                        suggestions = [item for item in items if item.is_dir()][:10]
                    
                    for i, item in enumerate(suggestions, 1):
                        print(f"   {i}. {item.name}")
                    
                    if len(suggestions) > 10:
                        print(f"   ... and {len(items) - 10} more")
                        
                except PermissionError:
                    print("   ❌ Permission denied to list directory")
            
            retry = input("\n🔄 Try again? (y/n): ").strip().lower()
            if retry != 'y':
                return ""

# ===== INFERENCE CLASS =====
class EmotionPredictor:
    """Class for predicting emotions from audio files"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the emotion predictor
        
        Args:
            model_path: Path to the trained model directory
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Emotion labels (same as training)
        self.emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
        
        # Load model and processor
        self.processor = None
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the trained model and processor"""
        try:
            print(f"Loading model from: {self.model_path}")
            print(f"Using device: {self.device}")
            
            # Load processor
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_path)
            
            # Load model
            self.model = Wav2Vec2ForSpeechClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ Model loaded successfully!")
            print(f"Model supports {len(self.emotions)} emotions: {self.emotions}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_audio(self, audio_path: str, target_sr: int = 16000) -> np.ndarray:
        """Preprocess audio file for inference"""
        try:
            print(f"   📝 Preprocessing audio: {audio_path}")
            
            # Load audio
            speech, sr = torchaudio.load(audio_path)
            print(f"   📊 Original: {speech.shape} samples at {sr} Hz")
            
            # Convert to mono if stereo
            if speech.shape[0] > 1:
                speech = torch.mean(speech, dim=0, keepdim=True)
                print(f"   🔄 Converted to mono: {speech.shape}")
            
            # Convert to numpy and squeeze
            speech = speech.squeeze().numpy()
            
            # Resample if needed
            if sr != target_sr:
                speech = librosa.resample(
                    y=speech, 
                    orig_sr=sr, 
                    target_sr=target_sr
                )
                print(f"   🔄 Resampled to {target_sr} Hz")
            
            print(f"   ✅ Preprocessing complete: {len(speech)} samples")
            return speech
            
        except Exception as e:
            print(f"❌ Error preprocessing audio {audio_path}: {e}")
            raise
    
    def predict_emotion(self, audio_path: str, return_probabilities: bool = False) -> Union[str, Dict]:
        """
        Predict emotion from audio file
        
        Args:
            audio_path: Path to audio file
            return_probabilities: If True, return probabilities for all emotions
            
        Returns:
            Predicted emotion string or dict with probabilities
        """
        try:
            # Preprocess audio
            audio = self.preprocess_audio(audio_path)
            
            # Process with Wav2Vec2 processor
            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=160000  # 10 seconds
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get probabilities
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_id = torch.argmax(probabilities, dim=-1).item()
                
            # Prepare results
            predicted_emotion = self.emotions[predicted_id]
            confidence = probabilities[0][predicted_id].item()
            
            if return_probabilities:
                prob_dict = {
                    emotion: prob.item() 
                    for emotion, prob in zip(self.emotions, probabilities[0])
                }
                return {
                    'predicted_emotion': predicted_emotion,
                    'confidence': confidence,
                    'all_probabilities': prob_dict
                }
            
            return predicted_emotion
            
        except Exception as e:
            print(f"❌ Error predicting emotion: {e}")
            raise
    
    def predict_batch(self, audio_paths: List[str]) -> List[Dict]:
        """Predict emotions for multiple audio files"""
        results = []
        
        for audio_path in audio_paths:
            try:
                result = self.predict_emotion(audio_path, return_probabilities=True)
                result['audio_path'] = audio_path
                results.append(result)
                
                print(f"✅ {Path(audio_path).name}: {result['predicted_emotion']} "
                      f"(confidence: {result['confidence']:.3f})")
                
            except Exception as e:
                print(f"❌ Error processing {audio_path}: {e}")
                results.append({
                    'audio_path': audio_path,
                    'error': str(e)
                })
        
        return results

# ===== MAIN FUNCTIONS =====
def test_single_audio(model_path: str, audio_path: str):
    """Test a single audio file"""
    print(f"\n{'='*50}")
    print("🎵 EMOTION RECOGNITION - SINGLE FILE TEST")
    print(f"{'='*50}")
    
    # Initialize predictor
    predictor = EmotionPredictor(model_path)
    
    # Predict emotion
    print(f"\n📁 Processing: {audio_path}")
    result = predictor.predict_emotion(audio_path, return_probabilities=True)
    
    # Display results
    print(f"\n🎯 RESULTS:")
    print(f"   Predicted Emotion: {result['predicted_emotion'].upper()}")
    print(f"   Confidence: {result['confidence']:.3f}")
    
    print(f"\n📊 All Probabilities:")
    for emotion, prob in sorted(result['all_probabilities'].items(), 
                               key=lambda x: x[1], reverse=True):
        print(f"   {emotion:8}: {prob:.3f} {'🔥' if prob > 0.5 else '📊' if prob > 0.1 else '▫️'}")

def test_multiple_audios(model_path: str, audio_folder: str):
    """Test multiple audio files from a folder"""
    print(f"\n{'='*50}")
    print("🎵 EMOTION RECOGNITION - BATCH TEST")
    print(f"{'='*50}")
    
    # Find audio files
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    audio_files = find_files_with_extensions(audio_folder, audio_extensions)
    
    if not audio_files:
        print(f"❌ No audio files found in {audio_folder}")
        return
    
    print(f"📁 Found {len(audio_files)} audio files:")
    for i, file in enumerate(audio_files[:10], 1):  # Show first 10
        print(f"   {i}. {file.name}")
    if len(audio_files) > 10:
        print(f"   ... and {len(audio_files) - 10} more files")
    
    # Initialize predictor
    predictor = EmotionPredictor(model_path)
    
    # Process all files
    results = predictor.predict_batch([str(f) for f in audio_files])
    
    # Summary
    print(f"\n📊 SUMMARY:")
    successful = [r for r in results if 'predicted_emotion' in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"   Successfully processed: {len(successful)}")
    print(f"   Failed: {len(failed)}")
    
    if successful:
        emotion_counts = {}
        for result in successful:
            emotion = result['predicted_emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print(f"\n🎭 Emotion Distribution:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {emotion:8}: {count} files")

def main():
    """Main function for testing"""
    print("🎵 Wav2Vec2 Emotion Recognition - Local Testing")
    print("=" * 50)
    
    # Configuration
    MODEL_PATH = "wav2vec2-emotion-model"  # Update this path
    
    # Check if model exists
    print(f"\n🔍 Checking model path...")
    model_valid, model_message = debug_path(MODEL_PATH, "directory")
    
    if not model_valid:
        print(f"❌ {model_message}")
        new_model_path = get_path_with_autocomplete("Enter correct path to model directory", "directory")
        if not new_model_path:
            print("❌ No valid model path provided. Exiting.")
            return
        MODEL_PATH = new_model_path
    
    # Interactive menu
    while True:
        print(f"\n🎛️  Options:")
        print("1. Test single audio file")
        print("2. Test multiple audio files from folder")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            audio_path = get_path_with_autocomplete("Enter path to audio file", "file")
            if audio_path:
                test_single_audio(MODEL_PATH, audio_path)
        
        elif choice == "2":
            folder_path = get_path_with_autocomplete("Enter path to folder with audio files", "directory")
            if folder_path:
                test_multiple_audios(MODEL_PATH, folder_path)
        
        elif choice == "3":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()