from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from supabase import create_client
import numpy as np
import os

# =========================
# LOAD MODEL
# =========================

model = load_model("model/skin_model.h5")

# =========================
# SUPABASE
# =========================

SUPABASE_URL = "https://pzkkvmtwwnxmqzosglni.supabase.co"
SUPABASE_KEY = "sb_publishable_8Gv18DzguBRN5FuPxm3FYQ_PXi57ld5"

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

# =========================
# CLASSES
# =========================

classes = [
    "Eczema",
    "Warts Molluscum and Viral Infections",
    "Melanoma",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma",
    "Melanocytic Nevi",
    "Benign Keratosis",
    "Psoriasis",
    "Seborrheic Keratoses",
    "Tinea Ringworm and Fungal Infections"
]

# =========================
# DISEASE INFO
# =========================

disease_info = {
    "Eczema": "Eczema causes dry, itchy and inflamed skin.",
    "Warts Molluscum and Viral Infections": "Viral infections that cause bumps or lesions.",
    "Melanoma": "A serious type of skin cancer.",
    "Atopic Dermatitis": "A chronic inflammatory skin condition.",
    "Basal Cell Carcinoma": "The most common type of skin cancer.",
    "Melanocytic Nevi": "Commonly known as moles.",
    "Benign Keratosis": "A harmless skin growth.",
    "Psoriasis": "A condition causing red and scaly skin patches.",
    "Seborrheic Keratoses": "Non-cancerous skin growths.",
    "Tinea Ringworm and Fungal Infections": "Fungal skin infection."
}

# =========================
# IMAGE PATH
# =========================

img_path = "test.jpg"

if not os.path.exists(img_path):
    print("Image not found:", img_path)
    exit()

# =========================
# LOAD IMAGE
# =========================

img = image.load_img(
    img_path,
    target_size=(224, 224)
)

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# =========================
# PREDICT
# =========================

prediction = model.predict(img_array)

predicted_index = np.argmax(prediction)

confidence = round(
    float(np.max(prediction) * 100),
    2
)

# =========================
# RESULT
# =========================

if confidence < 75:
    disease = "Unknown Disease / Low Confidence"
    info = "Prediction confidence is low. Please upload a clearer image."
else:
    disease = classes[predicted_index]
    info = disease_info.get(
        disease,
        "No information available."
    )

print("\n==============================")
print("SKIN DISEASE DETECTION RESULT")
print("==============================")
print("Disease :", disease)
print("Confidence :", confidence, "%")
print("Information :", info)
print("==============================")

# =========================
# SAVE TO SUPABASE
# =========================

try:

    supabase.table("predictions").insert({
        "user_email": "test@gmail.com",
        "disease": disease,
        "confidence": confidence,
        "image_name": img_path
    }).execute()

    print("\nPrediction saved successfully!")

except Exception as e:

    print("\nSupabase Error:")
    print(e)