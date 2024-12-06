from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Set environment variables for TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize Flask app and secret key
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Database configuration for MySQL on XAMPP
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost:3306/visionguard_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# File upload configuration
UPLOAD_FOLDER = 'static/uploads'
VISUALIZATION_FOLDER = 'static/visualizations'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VISUALIZATION_FOLDER'] = VISUALIZATION_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)

# Load the pre-trained model
model_prediction = load_model("model/best_model_visionguard_pv.h5")

# Define the class labels
CLASS_LABELS = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# User model for database
class User(db.Model):
    __tablename__ = 'user_table'
    
    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    creat_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<User {self.username}>"

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess image for prediction
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to generate Grad-CAM visualization
def generate_grad_cam(img_array, model, layer_name="block5_conv3"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = cv2.resize(heatmap, (150, 150))

    return heatmap

# Function to overlay heatmap on the image
def overlay_heatmap(heatmap, image_path, alpha=0.4, colormap=cv2.COLORMAP_JET):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)

    overlayed_img = cv2.addWeighted(heatmap_colored, alpha, img, 1 - alpha, 0)
    return overlayed_img

# Function to generate Saliency Map visualization
def generate_saliency_map(img_array, model):
    img_tensor = tf.convert_to_tensor(img_array)
    
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, img_tensor)
    saliency = np.max(np.abs(grads), axis=-1)[0]
    return saliency

## Authentication Routes
# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if user exists and password matches
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session['loggedin'] = True
            session['user_id'] = user.user_id
            session['username'] = user.username
            return redirect(url_for('home'))
        else:
            flash('Username atau password salah. Silakan coba lagi.')
    
    return render_template('login.html')

# Register Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Check if the username already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username sudah terdaftar. Silakan pilih username lain.')
            return redirect(url_for('register'))
        
        # Hash password before saving
        if password != confirm_password:
            flash('Password dan Verifikasi Password tidak cocok. Silakan coba lagi.')
            return redirect(url_for('register'))
        
        # Hash password sebelum menyimpannya
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
        # Add new user to the database
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for('login'))
    
    return render_template('register.html')

# Logout Route
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

## Page Routes
# Home Route
@app.route('/home')
def home():
    # Check if session variables are set
    print("Session loggedin:", session.get('loggedin'))
    
    if 'loggedin' in session:
        return render_template('home.html', username=session['username'])
    else:
        return redirect(url_for('login'))

# Route for classification
@app.route('/classification', methods=['GET', 'POST'])
def classification():
    result = None
    img_path = None
    saliency_map_path = None
    grad_cam_path = None
    predicted_probability = None
    interpretation_text_sm = None
    interpretation_text_gradcam = None

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        
        if file and allowed_file(file.filename):
            # Save the file and preprocess it
            final_filename = "classification.png"
            final_filepath = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)
            file.save(final_filepath)
            img_path = f"/static/uploads/{final_filename}".replace("\\", "/")

            # Preprocess the image for prediction
            img_array = preprocess_image(final_filepath)

            # Predict class and calculate probability
            predictions = model_prediction.predict(img_array)
            result = CLASS_LABELS[np.argmax(predictions)]

            predicted_class_index = np.argmax(predictions)
            predicted_probability = predictions[0][predicted_class_index] * 100
            
                        
            # Interpretation texts for Saliency Map and Grad-CAM
            interpretation_texts_sm = {
                'cataract': """
                    Pada visualisasi <strong>Saliency Map</strong> ini, Anda akan melihat area yang sangat terang di bagian tengah atau sekitar lensa mata.
                    Ini menunjukkan area yang dianggap paling penting oleh model saat mengidentifikasi katarak, yaitu kerusakan pada lensa mata.
                    Biasanya, pada katarak, lensa mata mengalami kekeruhan atau perubahan yang terlihat pada gambar retina. 
                    Saliency map ini membantu menunjukkan bagian yang perlu lebih diperhatikan dalam gambar untuk diagnosis.
                """,
                'diabetic_retinopathy': """
                    Pada visualisasi <strong>Saliency Map</strong> ini, model memperhatikan perubahan pada pembuluh darah atau titik-titik yang
                    menunjukkan adanya pendarahan di retina. Ini dapat menunjukkan gejala dari retinopati diabetik, yang umumnya ditandai oleh 
                    pembuluh darah yang rusak atau bengkak akibat diabetes. Warna terang pada bagian retina akan menunjukkan area yang paling berpengaruh
                    dalam keputusan model.
                """,
                'glaucoma': """
                    Pada visualisasi <strong>Saliency Map</strong> ini, area yang paling terang biasanya berada pada bagian retina atau saraf optik.
                    Ini mengindikasikan peningkatan tekanan intraokular yang terkait dengan glaukoma. Saliency map ini memberikan wawasan pada bagian yang
                    dianggap model sebagai tanda peningkatan tekanan pada saraf optik.
                """,
                'normal': """
                    Pada visualisasi <strong>Saliency Map</strong> ini, tidak ada area yang mencolok karena gambar yang tidak menunjukkan kelainan.
                    Hal ini menunjukkan bahwa model tidak menemukan indikasi adanya penyakit atau kelainan pada mata, yang mengarah pada hasil klasifikasi 
                    sebagai normal.
                """
            }
            
            interpretation_texts_gradcam = {
                'cataract': """
                    Pada visualisasi <strong>Grad-CAM</strong> ini, Anda akan melihat area dengan warna terang di sekitar lensa mata, yang menunjukkan bagian
                    yang paling diperhatikan oleh model dalam menentukan adanya katarak. Grad-CAM ini menggambarkan fokus utama model pada kekeruhan atau
                    perubahan pada lensa mata, yang merupakan ciri khas dari katarak. Warna yang lebih terang di sekitar lensa menunjukkan area yang paling
                    mempengaruhi keputusan model dalam mendiagnosis kondisi ini. Grad-CAM memberikan wawasan tentang bagian gambar yang dianggap sebagai
                    tanda utama penyakit pada model.
                """,
    
                'diabetic_retinopathy': """Pada visualisasi <strong>Grad-CAM</strong> ini, model memfokuskan perhatian pada pembuluh darah retina yang rusak
                    atau titik-titik pendarahan, yang merupakan ciri khas dari retinopati diabetik. Area dengan warna terang pada retina menunjukkan tempat-tempat
                    yang dianggap model sebagai indikasi kerusakan pembuluh darah atau pembengkakan akibat diabetes. Grad-CAM membantu menggambarkan area utama
                    yang mendasari keputusan model untuk mendeteksi penyakit ini, memberi gambaran jelas pada kelainan yang perlu diwaspadai.
                """,
                
                'glaucoma': """Pada visualisasi <strong>Grad-CAM</strong> ini, bagian yang paling terang biasanya terfokus pada area retina atau saraf optik, yang
                    menunjukkan peningkatan tekanan intraokular, karakteristik dari glaukoma. Grad-CAM ini memperlihatkan area yang dipilih oleh model yang paling
                    berpengaruh dalam menentukan diagnosa glaukoma, terutama pada saraf optik yang tertekan akibat peningkatan tekanan di dalam mata. Warna terang
                    pada saraf optik memberikan wawasan tentang tanda utama glaukoma dalam gambar.
                """,
                
                'normal': """Pada visualisasi <strong>Grad-CAM</strong> ini, Anda tidak akan melihat area yang terlalu mencolok atau terang, karena gambar ini tidak
                    menunjukkan kelainan atau tanda-tanda penyakit. Hal ini menunjukkan bahwa model tidak menemukan indikasi adanya kelainan pada retina atau struktur
                    mata lainnya. Grad-CAM ini mengonfirmasi bahwa model mengidentifikasi gambar sebagai mata normal, yang tidak menunjukkan adanya kerusakan atau kondisi medis.
                """,
            }
            
            # Get the interpretation text result
            interpretation_text_sm = interpretation_texts_sm.get(result, "")
            interpretation_text_gradcam = interpretation_texts_gradcam.get(result, "")

            # Generate Saliency Map
            saliency_map = generate_saliency_map(img_array, model_prediction)
            plt.imsave(os.path.join(app.config['VISUALIZATION_FOLDER'], "saliency_map.png"), saliency_map, cmap='viridis')
            saliency_map_path = "/static/visualizations/saliency_map.png"

            # Generate Grad-CAM
            heatmap = generate_grad_cam(img_array, model_prediction, layer_name="block5_conv3")
            grad_cam_img = overlay_heatmap(heatmap, final_filepath)
            grad_cam_filename = "grad_cam.png"
            cv2.imwrite(os.path.join(app.config['VISUALIZATION_FOLDER'], grad_cam_filename), grad_cam_img)
            grad_cam_path = f"/static/visualizations/{grad_cam_filename}"

        else:
            flash("File type not allowed. Please upload a valid image (png, jpg, and jpeg).")
            return redirect(request.url)

    return render_template(
        'classification.html',
        result=result,
        probability=predicted_probability,
        interpretation_text_sm_result=interpretation_text_sm,
        interpretation_text_gradcam_result=interpretation_text_gradcam,
        img_path=img_path,
        saliency_map_path=saliency_map_path,
        grad_cam_path=grad_cam_path
    )

# CNN Result Route
@app.route('/cnn_results')
def cnn_results():
    return render_template('cnn_results.html')

# Root Route
@app.route('/')
def index():
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)