"""Architecture of the baseline 1D CNN, identical to ``notebooks/01_baseline_cnn.ipynb``.

Three Conv1D blocks (128 -> 64 -> 32 filters), global average pooling and a
small dense head. Keep this in sync with the notebook: the committed
``SER_model.weights.h5`` only loads if the layer structure matches.
"""

from __future__ import annotations

from features import FEATURE_LENGTH


def build_model(input_length: int = FEATURE_LENGTH, num_classes: int = 8):
    from tensorflow import keras
    from tensorflow.keras import layers as L
    from tensorflow.keras.regularizers import l2

    return keras.Sequential(
        [
            keras.Input(shape=(input_length, 1)),
            L.Conv1D(128, kernel_size=5, strides=1, padding="same", activation="relu",
                     kernel_regularizer=l2(1e-4)),
            L.BatchNormalization(),
            L.MaxPool1D(pool_size=5, strides=2, padding="same"),
            L.Dropout(0.4),

            L.Conv1D(64, kernel_size=5, strides=1, padding="same", activation="relu"),
            L.BatchNormalization(),
            L.MaxPool1D(pool_size=5, strides=2, padding="same"),
            L.Dropout(0.3),

            L.Conv1D(32, kernel_size=3, strides=1, padding="same", activation="relu"),
            L.BatchNormalization(),
            L.MaxPool1D(pool_size=3, strides=2, padding="same"),
            L.Dropout(0.3),

            L.GlobalAveragePooling1D(),

            L.Dense(32, activation="relu", kernel_regularizer=l2(1e-4)),
            L.BatchNormalization(),
            L.Dropout(0.5),
            L.Dense(num_classes, activation="softmax"),
        ]
    )
