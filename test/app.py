from flask import Flask, render_template, request, url_for
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
import os
import json

# Inisialisasi Flask
app = Flask(__name__)

# Folder untuk menyimpan gambar yang diunggah
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Ekstensi file gambar yang diperbolehkan
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Memuat model TensorFlow
model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_history.keras')
model = tf.keras.models.load_model(model_path)

# Nama kelas prediksi
class_names = ['Normal', 'Glaukoma']

# Membaca akurasi dari training_history.json
def get_model_accuracy():
    history_path = os.path.join(os.path.dirname(__file__), 'model', 'training_history.json')
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
            val_accuracy = history.get('val_accuracy', [])
            if val_accuracy:
                return f"{int(max(val_accuracy) * 100)}%"
    except Exception as e:
        print(f"Error loading training history: {e}")
    return "N/A"

# Fungsi untuk memprediksi gambar
def predict_image(filepath):
    img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisasi

    pred = model.predict(img_array)
    predicted_class = np.argmax(pred, axis=1)[0]
    confidence = pred[0][predicted_class]
    predicted_label = class_names[predicted_class]

    return predicted_label, confidence

# Route utama
@app.route('/')
def index():
    model_accuracy = get_model_accuracy()
    return render_template('index.html', model_accuracy=model_accuracy)

# Route untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if not allowed_file(file.filename):
        return "Invalid file type. Only images are allowed.", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    predicted_label, confidence = predict_image(filepath)
    image_url = url_for('static', filename=f'uploads/{filename}')
    model_accuracy = get_model_accuracy()

    return render_template('index.html',
                           prediction=predicted_label,
                           confidence=f"{confidence*100:.2f}%",
                           image_path=image_url,
                           model_accuracy=model_accuracy)

if __name__ == '__main__':
    app.run(debug=True)
