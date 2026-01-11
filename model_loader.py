# model_loader.py
import os
import urllib.request

BASE_URL = "https://github.com/Shivanshu19droid/GenePredict/releases/download/v1.0-models"

MODEL_FILES = {
    "trained_model_2.keras": "model/trained_model_2.keras",
    "tokenizer.pkl": "model/tokenizer.pkl",
    "label_encoder.pkl": "model/label_encoder.pkl",
}

def download_models():
    os.makedirs("model", exist_ok=True)

    for filename, filepath in MODEL_FILES.items():
        if not os.path.exists(filepath):
            print(f"⬇️ Downloading {filename}...")
            urllib.request.urlretrieve(
                f"{BASE_URL}/{filename}",
                filepath
            )
