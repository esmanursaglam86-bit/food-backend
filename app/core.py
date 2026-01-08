import tensorflow as tf
import numpy as np
from PIL import Image

# Model bir kere yüklenecek (çok önemli)
model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=True
)

# ImageNet class label’ları
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

def predict_food(image: Image.Image):
    processed = preprocess_image(image)
    preds = model.predict(processed)

    decoded = decode_predictions(preds, top=1)[0][0]

    return {
        "label": decoded[1],
        "confidence": float(decoded[2])
    }

