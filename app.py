from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from PIL import Image
import matplotlib.pyplot as plt
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show errors

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Database configuration for MySQL on XAMPP
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost:3306/visionguard_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Configuration for file uploads
UPLOAD_FOLDER = 'static/uploads'
VISUALIZATION_FOLDER = 'static/visualizations'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VISUALIZATION_FOLDER'] = VISUALIZATION_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)

# Load the pre-trained model
model_prediction = load_model("model/best_model.h5")

# Define the class labels
CLASS_LABELS = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the User model to map to `user_table` in the database
class User(db.Model):
    __tablename__ = 'user_table'
    
    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    creat_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<User {self.username}>"

# Function to preprocess image for prediction
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((150, 150))  # Resize to model input size
    img = np.array(img) / 255.0   # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to generate Grad-CAM visualization
def generate_grad_cam(img_array, model, layer_name="block5_conv3"):  # Adjust layer_name as per model
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

    # Normalize and resize the heatmap
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)  # Convert to numpy array
    heatmap = cv2.resize(heatmap, (150, 150))

    return heatmap


# Function to overlay heatmap on the image
def overlay_heatmap(heatmap, image_path, alpha=0.4, colormap=cv2.COLORMAP_JET):
    # Load and resize the original image to match the heatmap size
    img = cv2.imread(image_path)
    img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))

    # Convert the heatmap to color (3 channels) if it's grayscale
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)

    # Blend the heatmap with the original image
    overlayed_img = cv2.addWeighted(heatmap_colored, alpha, img, 1 - alpha, 0)
    return overlayed_img

# Function to generate Saliency Map visualization
def generate_saliency_map(img_array, model):
    # Convert the numpy array to a TensorFlow tensor and watch it
    img_tensor = tf.convert_to_tensor(img_array)
    
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)  # Ensure that TensorFlow is tracking gradients for this tensor
        predictions = model(img_tensor)
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, img_tensor)
    saliency = np.max(np.abs(grads), axis=-1)[0]
    return saliency

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    result = None
    img_path = None
    saliency_map_path = None
    grad_cam_path = None

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        
        if file and allowed_file(file.filename):
            now = datetime.now()
            final_filename = now.strftime("%d%m%y-%H%M%S") + ".png"
            final_filepath = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)
            file.save(final_filepath)
            img_path = f"/static/uploads/{final_filename}".replace("\\", "/")

            # Preprocess the image for prediction
            img_array = preprocess_image(final_filepath)

            # Predict the class
            predictions = model_prediction.predict(img_array)
            result = CLASS_LABELS[np.argmax(predictions)]

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
            flash("File type not allowed. Please upload a valid image.")
            return redirect(request.url)
    
    return render_template('classification.html', result=result, img_path=img_path,
                        saliency_map_path=saliency_map_path, grad_cam_path=grad_cam_path)


# Index Route (redirects to login)
@app.route('/')
def index():
    return redirect(url_for('login'))

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Cari user berdasarkan username
        user = User.query.filter_by(username=username).first()
        
        # Cek apakah user ada dan password cocok
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
        
        # Cek jika password dan verifikasi password cocok
        if password != confirm_password:
            flash('Password dan Verifikasi Password tidak cocok. Silakan coba lagi.')
            return redirect(url_for('register'))
        
        # Hash password sebelum menyimpannya
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
        # Tambahkan user baru ke database
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        # flash('Anda berhasil mendaftar! Silakan login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')


# Home Route
@app.route('/home')
def home():
    # Debugging: check if session variables are set
    print("Session loggedin:", session.get('loggedin'))
    
    if 'loggedin' in session:
        return render_template('home.html', username=session['username'])
    else:
        return redirect(url_for('login'))

# Additional Routes for Other Pages
@app.route('/cnn_results')
def cnn_results():
    # Placeholder route for CNN Results page
    return render_template('cnn_results.html')  # Create cnn_results.html in templates


# Logout Route
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Run the application
if __name__ == '__main__':
    app.run(debug=False)