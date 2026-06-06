from pathlib import Path

import tensorflow as tf
import keras
from keras import layers


# =========================
# Custom Layer
# =========================
@tf.keras.utils.register_keras_serializable()
class FeatureInteractionLayer(layers.Layer):
    def __init__(self, units=64, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.dense = layers.Dense(units, activation=activation)

    def call(self, inputs):
        squared_features = tf.square(inputs)
        log_features = tf.math.log1p(tf.abs(inputs))

        combined_features = tf.concat([inputs, squared_features, log_features], axis=-1)

        return self.dense(combined_features)

    def get_config(self):
        config = super().get_config()

        config.update({"units": self.units, "activation": self.activation})

        return config


# =========================
# Model Path
# =========================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model_patched.keras"


# =========================
# Load Model
# =========================
model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={"FeatureInteractionLayer": FeatureInteractionLayer},
    compile=False,
)

print("✅ Model Loaded Successfully")
