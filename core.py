import numpy as np
import json
import face_recognition
from database import get_conn

def get_embedding_from_image(img):
    """Processes an array image directly without looking up disk file paths."""
    encs = face_recognition.face_encodings(img)
    if not encs:
        raise ValueError("No face detected.")
    return encs[0].tolist()
