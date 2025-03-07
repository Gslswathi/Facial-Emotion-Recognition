from flask import Flask, render_template, request, redirect, url_for, session,jsonify
from werkzeug.utils import secure_filename
import mysql.connector
import os
import tensorflow as tf


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to an actual secret key
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# MySQL Configuration
db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="password",
    database="emotion",
    port=3307,
)
db_cursor = db_connection.cursor()
'''
# Initialize Database
def initialize_database():
    db_cursor.execute("CREATE TABLE IF NOT EXISTS users (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255), password VARCHAR(255))")
    db_connection.commit()
'''
# Pre-trained model for emotion detection

import cv2
import keras
from keras.preprocessing import image

import numpy as np
import numpy as np
from keras.utils import img_to_array
import time


def detect_emotion(media_path):
    # Load the pre-trained CNN model
    model = tf.keras.models.load_model(r"C:\Users\k manoj\Downloads\4-2_Project\lr_CNN_model (1).h5")
    
    # Initialize variables for frame capture control
    frame_rate = 5  # Number of frames to capture per second
    prev = 0
    
    # Check if the media_path is a video
    video = media_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))  # Add other video formats if needed
    cap = cv2.VideoCapture(media_path) if video else None
    
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    emotion_statistics = {'Angry': 0, 'Disgust': 0, 'Fear': 0, 'Happy': 0, 'Sad': 0, 'Surprise': 0, 'Neutral': 0}
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        if video:
            time_elapsed = time.time() - prev
            ret, img = cap.read()
            if not ret:
                break  # End of video
            
            # Only proceed if enough time has elapsed to meet the desired frame rate
            if time_elapsed > 1./frame_rate:
                prev = time.time()
            else:
                continue
        else:
            img = cv2.imread(media_path)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = model.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                emotion_prediction = emotion_labels[np.argmax(prediction)]
                emotion_statistics[emotion_prediction] += 1

        if video:
            cv2.imshow('Emotion Detector', img)
            if cv2.waitKey(27) & 0xFF == ord('q'):
                break  # Exit on pressing 'q'
        else:
            # If image, display the result and wait for keypress
            cv2.imshow('Emotion Detector', img)
            cv2.waitKey(0)
            break  # Process only once for images

    if video:
        cap.release()
    cv2.destroyAllWindows()
    
    return emotion_statistics

# Usage:
# emotion_stats = detect_emotion('path_to_your_image_or_video')



@app.route('/')
def home():
    return render_template('register.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        db_cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", (name, email, password))
        db_connection.commit()
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        db_cursor.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
        user = db_cursor.fetchone()
        print("details recived")
        if user:
            session['username'] = email
            print("user registred")
            return render_template('dashboard.html')
    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'username' in session:
        if request.method == 'POST':
            if 'file' not in request.files:
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                return redirect(request.url)
            if file:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                emotion_statistics = detect_emotion(file_path)
                os.remove(file_path)  # Remove the uploaded file after processing
                return render_template('result.html', emotion_statistics=emotion_statistics)
            return redirect('home.html')
        return render_template('dashboard.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
