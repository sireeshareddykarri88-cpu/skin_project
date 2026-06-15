from flask import Flask, render_template, request, session, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from supabase import create_client
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "skin_detector_secret"

# ==========================
# SUPABASE CONFIG
# ==========================

SUPABASE_URL = "https://pzkkvmtwwnxmqzosglni.supabase.co"
SUPABASE_KEY = "sb_publishable_8Gv18DzguBRN5FuPxm3FYQ_PXi57ld5"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================
# LOAD MODEL
# ==========================

model = load_model("model/skin_model.h5")

# ==========================
# CLASSES
# ==========================

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
    "Tinea Ringworm"
]

# ==========================
# DISEASE INFO
# ==========================

disease_info = {
    "Eczema": "Eczema causes dry, itchy and inflamed skin.",
    "Warts Molluscum and Viral Infections": "Viral skin infections causing bumps.",
    "Melanoma": "Serious form of skin cancer.",
    "Atopic Dermatitis": "Chronic inflammatory skin condition.",
    "Basal Cell Carcinoma": "Most common skin cancer.",
    "Melanocytic Nevi": "Common moles, usually harmless.",
    "Benign Keratosis": "Non-cancerous skin growth.",
    "Psoriasis": "Red, scaly skin patches.",
    "Seborrheic Keratoses": "Harmless skin growths.",
    "Tinea Ringworm": "Fungal skin infection."
}

# ==========================
# HOME PAGE
# ==========================
@app.route("/")
def index():
    return render_template(
        "home.html",
        name=session.get("name"),
        email=session.get("email"),
        mobile=session.get("mobile")
    )


@app.route("/home")
def homepage():
    return render_template(
        "home.html",
        name=session.get("name"),
        email=session.get("email"),
        mobile=session.get("mobile")
    )

# ==========================
# AUTH PAGES
# ==========================

@app.route("/signup")
def signup():
    return render_template("signup.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/login_user", methods=["POST"])
def login_user():

    email = request.form["email"]
    password = request.form["password"]

    try:
        result = supabase.table("users") \
            .select("*") \
            .eq("email", email) \
            .eq("password", password) \
            .execute()

        if len(result.data) > 0:

            user = result.data[0]

            session["name"] = user["name"]
            session["email"] = user["email"]
            session["mobile"] = user["mobile"]

            return redirect(url_for("homepage"))

        else:
            return "Invalid Login"

    except Exception as e:
        return str(e)

# ==========================
# REGISTER USER
# ==========================

@app.route("/register", methods=["POST"])
def register():

    name = request.form["name"]
    email = request.form["email"]
    mobile = request.form["mobile"]
    password = request.form["password"]

    try:
        supabase.table("users").insert({
            "name": name,
            "email": email,
            "mobile": mobile,
            "password": password
        }).execute()

    except Exception as e:
        print("User Save Error:", e)

    session["name"] = name
    session["email"] = email
    session["mobile"] = mobile

    return redirect(url_for("homepage"))

# ==========================
# DIAGNOSIS PAGE
# ==========================

@app.route("/diagnosis")
def diagnosis():
    return render_template(
        "index.html",
        name=session.get("name"),
        email=session.get("email"),
        mobile=session.get("mobile")
    )

@app.route("/doctors")
def doctors():
    return render_template("doctors.html")

# ==========================
# PREDICT
# ==========================

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    upload_folder = "static/uploads"

    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    index = np.argmax(prediction)

    confidence = round(float(np.max(prediction) * 100), 2)

    if confidence < 75:
        disease = "Unknown Disease / Low Confidence"
        info = "The uploaded image does not match any disease confidently."
    else:
        disease = classes[index]
        info = disease_info.get(disease, "No information available.")

    try:
        supabase.table("predictions").insert({
            "user_email": session.get("email"),
            "disease": disease,
            "confidence": confidence,
            "image_name": file.filename
        }).execute()

    except Exception as e:
        print("Prediction Error:", e)

    return render_template(
        "index.html",
        disease=disease,
        confidence=confidence,
        info=info,
        image="/" + filepath,
        name=session.get("name"),
        email=session.get("email"),
        mobile=session.get("mobile")
    )

# ==========================
# HISTORY
# ==========================

@app.route("/history")
def history():

    try:
        result = supabase.table("predictions") \
            .select("*") \
            .eq("user_email", session.get("email")) \
            .execute()

        records = result.data

    except Exception as e:
        print(e)
        records = []

    return render_template("history.html", records=records)

# ==========================
# APPOINTMENT
# ==========================

@app.route("/appointment", methods=["GET", "POST"])
def appointment():

    if request.method == "POST":

        name = request.form["name"]
        email = request.form["email"]
        doctor = request.form["doctor"]
        appointment_date = request.form["appointment_date"]

        try:
            supabase.table("appointments").insert({
                "name": name,
                "email": email,
                "doctor": doctor,
                "appointment_date": appointment_date
            }).execute()

            return render_template("appointment.html", success=True)

        except Exception as e:
            print("Appointment Error:", e)

    return render_template("appointment.html")

# ==========================
# DASHBOARD
# ==========================

@app.route("/dashboard")
def dashboard():

    email = session.get("email")

    try:
        predictions = supabase.table("predictions") \
            .select("*") \
            .eq("user_email", email) \
            .execute()

        appointments = supabase.table("appointments") \
            .select("*") \
            .eq("email", email) \
            .execute()

        return render_template(
            "dashboard.html",
            user_name=session.get("name"),
            total_predictions=len(predictions.data),
            total_appointments=len(appointments.data),
            predictions=predictions.data,
            appointments=appointments.data
        )

    except Exception as e:

        print("Dashboard Error:", e)

        return render_template(
            "dashboard.html",
            user_name=session.get("name"),
            total_predictions=0,
            total_appointments=0,
            predictions=[],
            appointments=[]
        )

# ==========================
# LOGOUT
# ==========================
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# ==========================
# RUN
# ==========================

if __name__ == "__main__":
    app.run(debug=True)