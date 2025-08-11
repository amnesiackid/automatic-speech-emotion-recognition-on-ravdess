import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling1D
from keras.regularizers import l2
import librosa
import numpy as np
import json



# --- 1. Model Architecture Definition ---
def create_model_from_summary(input_shape, num_classes):
    model = Sequential([
        # Layer 1
        Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu',
               kernel_regularizer=l2(1e-4),
               input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=5, strides=2, padding='same'),
        Dropout(0.4),
        
        # Layer 2
        Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=5, strides=2, padding='same'),
        Dropout(0.3),
        
        # Layer 3
        Conv1D(32, kernel_size=3, strides=1, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=3, strides=2, padding='same'),
        Dropout(0.3),
        
        # Global Average Pooling instead of Flatten
        GlobalAveragePooling1D(),
        
        # Dense Layers
        Dense(32, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
        
    ])
    return model

# --- 2. Feature Extraction ---
def get_features_for_model(path):
    # Load the audio file
    data, sample_rate = librosa.load(path, duration=5, offset=0.5)
    
    # Trim silence from the beginning and end
    trimmed_data, _ = librosa.effects.trim(data, top_db=60)
    
    # Pad or trim the audio to have a length of 2376
    fixed_length_data = librosa.util.fix_length(trimmed_data, size=2376)
    
    return fixed_length_data

# --- 3. Loading and Prediction Logic ---
try:
    # Load the list of emotion categories
    with open('emotion_categories.json', 'r') as f:
        emotion_categories = json.load(f)
    print("✅ Emotion categories loaded successfully.")

    # Define model input shape and number of classes
    INPUT_SHAPE = (2376, 1)
    NUM_CLASSES = len(emotion_categories)

    # Create the model structure
    model = create_model_from_summary(INPUT_SHAPE, NUM_CLASSES)
    print("✅ Model structure created successfully.")

    # 使用更安全的权重加载方式
    try:
        model.load_weights('SER_model.weights.h5')
        print("✅ Model weights loaded successfully.")
    except Exception as weight_error:
        print(f"❌ Weight loading error: {weight_error}")
        print("🔄 Trying to load with skip_mismatch=True...")
        try:
            model.load_weights('SER_model.weights.h5', by_name=True, skip_mismatch=True)
            print("✅ Model weights loaded successfully (with skip_mismatch).")
        except Exception as e:
            print(f"❌ Failed to load weights even with skip_mismatch: {e}")
            raise

except Exception as e:
    print(f"\n❌ Error during setup: {e}")
    print("👉 Please ensure 'SER_model.weights.h5' and 'emotion_categories.json' are in the same directory.")
    exit()

def predict_emotion(audio_path):
    """
    Loads an audio file, preprocesses it, and predicts its emotion.
    """
    try:
        # Get the feature vector from the audio file
        features = get_features_for_model(audio_path)
        
        # Reshape for model input: (1, 2376, 1)
        features = np.expand_dims(features, axis=0) # Add batch dimension
        features = np.expand_dims(features, axis=2) # Add channel dimension
        
        # Get model prediction
        prediction = model.predict(features)
        
        # Find the index of the highest probability
        predicted_index = np.argmax(prediction, axis=1)[0]
        
        # Get the corresponding emotion label
        predicted_emotion = emotion_categories[predicted_index]
        
        return predicted_emotion

    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        return None

# --- Main Execution Block ---
if __name__ == '__main__':
    # ❗️IMPORTANT: Replace this with the actual path to your audio file
    audio_file_to_test = '03-01-01-01-02-01-24.wav'
    
    print("-" * 40)
    print(f"🎤 Attempting to predict emotion for: {audio_file_to_test}")
    
    emotion = predict_emotion(audio_file_to_test)
    
    if emotion:
        print("\n" + "="*40)
        print(f"🎉 Predicted Emotion: {emotion.upper()} 🎉")
        print("="*40)