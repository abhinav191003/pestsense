"""
Smart Pest Presence Detection System - Flask Backend
Render-ready Flask backend for PestSense.

Local run:
    python app.py
Then visit:
    http://localhost:5000
"""

import os
import io
import base64

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# TensorFlow is optional at startup. The model is loaded only when a detection
# request arrives, so the web server can start before the heavy model loads.
try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import (
        preprocess_input,
        decode_predictions,
    )
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    MobileNetV2 = None
    preprocess_input = None
    decode_predictions = None
    print("[WARNING] TensorFlow not found - running in DEMO mode.")

UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}
IMG_SIZE = (224, 224)

PEST_DB = {
    "grasshopper": ("Grasshopper", "Medium", "Apply kaolin clay barrier; use neem-based biopesticides."),
    "locust": ("Locust", "Critical", "Apply organophosphate insecticides; contact agriculture dept immediately."),
    "cricket": ("Cricket", "Low", "Remove debris; use insecticide bait around crop borders."),
    "caterpillar": ("Caterpillar", "High", "Apply Bt spray; remove egg masses manually."),
    "moth": ("Moth/Borer", "High", "Use pheromone traps; apply Bt or pyrethrin."),
    "cabbage_butterfly": ("Cabbage Worm", "High", "Apply Bacillus thuringiensis (Bt); use row covers."),
    "monarch": ("Butterfly", "Low", "Monitor only; generally harmless to crops."),
    "admiral": ("Butterfly", "Low", "Monitor only; generally harmless to crops."),
    "ringlet": ("Caterpillar", "High", "Apply Bt spray; remove egg masses manually."),
    "sulphur_butterfly": ("Butterfly Pest", "Medium", "Apply Bt spray; use row covers."),
    "beetle": ("Beetle", "Medium", "Hand-pick adults; apply neem oil or pyrethrin."),
    "weevil": ("Weevil", "High", "Apply diatomaceous earth; use pheromone-baited traps."),
    "leaf_beetle": ("Leaf Beetle", "High", "Apply pyrethrin; use row covers to protect plants."),
    "long_horn": ("Longhorn Beetle", "High", "Remove infested branches; apply systemic insecticide."),
    "rhinoceros_beetle": ("Rhinoceros Beetle", "High", "Apply neem biopesticide; remove larvae from soil."),
    "buprestid": ("Jewel Beetle", "High", "Remove infested wood; apply systemic insecticide."),
    "ladybug": ("Ladybug", "Low", "Beneficial predator - no action needed."),
    "leafhopper": ("Leafhopper", "Medium", "Spray kaolin clay; use reflective mulch."),
    "planthopper": ("Planthopper", "High", "Apply imidacloprid; drain field periodically."),
    "shield_bug": ("Stink Bug", "Medium", "Use kaolin clay or pyrethrin spray."),
    "cicada": ("Cicada", "Medium", "Use netting to protect young trees; apply pyrethrin."),
    "fly": ("Fruit Fly", "High", "Use protein bait traps; remove fallen fruit promptly."),
    "spider": ("Spider Mite", "High", "Apply miticide or neem oil; increase humidity."),
    "tick": ("Mite/Tick", "High", "Apply miticide or neem oil; increase humidity."),
    "ant": ("Ant", "Medium", "Apply diatomaceous earth barrier around plant base."),
    "bee": ("Bee", "Low", "Beneficial pollinator - no action needed."),
    "dragonfly": ("Dragonfly", "Low", "Beneficial predator - no action needed."),
    "mantis": ("Praying Mantis", "Low", "Beneficial predator - no action needed."),
    "slug": ("Slug", "Medium", "Apply iron phosphate bait; create copper barriers."),
    "snail": ("Snail", "Medium", "Apply iron phosphate bait; create copper barriers."),
    "armyworm": ("Armyworm", "Critical", "Apply Bt or pyrethroid at early larval stage."),
    "mealybug": ("Mealybug", "Medium", "Dab with isopropyl alcohol; use systemic insecticides."),
    "whitefly": ("Whitefly", "High", "Use yellow sticky traps; apply pyrethrin."),
    "thrips": ("Thrips", "Medium", "Remove infested parts; apply spinosad spray."),
    "walking_stick": ("Walking Stick", "Low", "Hand-pick; use neem oil if infestation is heavy."),
    "stick_insect": ("Stick Insect", "Low", "Hand-pick from plants; use neem oil spray."),
    "cockroach": ("Grasshopper", "Medium", "Apply kaolin clay barrier; use neem-based biopesticides."),
}

GRASSHOPPER_LOOKALIKES = {
    "ant", "cockroach", "cricket", "walking_stick",
    "stick_insect", "grasshopper", "locust", "mantis",
}

app = Flask(__name__, static_folder="static")
CORS(app)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

_model = None


def get_model():
    """Load MobileNetV2 only when first needed."""
    global _model
