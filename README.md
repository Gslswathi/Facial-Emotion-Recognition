echo "# Face Emotion Recognition

This project is a **Face Emotion Recognition System** that uses a Convolutional Neural Network (CNN) to detect and classify emotions from images and videos. It is built using **Flask**, **OpenCV**, **TensorFlow**, and **MySQL**.

## Features
- **User Authentication**: Register and log in using MySQL database.
- **Emotion Detection**: Upload an image or video, and the system detects emotions such as *Happy, Sad, Angry, Surprise, Neutral, Disgust,* and *Fear*.
- **Real-time Analysis**: Uses OpenCV to detect faces and TensorFlow for emotion prediction.
- **Data Visualization**: Displays the emotion statistics in a pie chart.
- **Web-based Interface**: Built with Flask and rendered using HTML templates.

## Installation

### **1. Clone the Repository**
\`\`\`sh
git clone https://github.com/YOUR_GITHUB_USERNAME/Face_Emotion_Recognition.git
cd Face_Emotion_Recognition
\`\`\`

### **2. Install Dependencies**
Ensure you have Python installed. Then, run:
\`\`\`sh
pip install flask mysql-connector-python tensorflow opencv-python keras numpy werkzeug
\`\`\`

### **3. Set Up MySQL Database**
1. Install **XAMPP** and start MySQL.
2. Open \`phpMyAdmin\` and create a database named **emotion**.
3. Run the following SQL query to create the \`users\` table:
   \`\`\`sql
   CREATE TABLE users (
       id INT AUTO_INCREMENT PRIMARY KEY,
       name VARCHAR(255),
       email VARCHAR(255),
       password VARCHAR(255)
   );
   \`\`\`
4. Update \`app.py\` with your MySQL credentials.

### **4. Start the Application**
\`\`\`sh
python app.py
\`\`\`
Then, open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Usage
1. **Register** a new account.
2. **Log in** to access the dashboard.
3. **Upload an image or video** for emotion analysis.
4. The system will **detect faces**, analyze emotions, and display results.
5. **View statistics** in a pie chart format.

## Folder Structure
\`\`\`
Face_Emotion_Recognition/
│── uploads/             # Uploaded files
│── static/              # Static assets (CSS, JS)
│── templates/           # HTML templates
│── app.py               # Main Flask application
│── lr_CNN_model.h5      # Pre-trained model
│── project_Face_Emotion_Recognition.ipynb # Jupyter Notebook
│── working_document.pdf # Documentation
│── README.md            # Project documentation
\`\`\`

## Troubleshooting
- If you get \`Access denied for user 'root'@'localhost'\`, update \`app.py\` with your MySQL **username, password, and port**.
- If Flask doesn’t start, ensure **Apache** and **MySQL** are running in XAMPP.
- If TensorFlow installation fails, install it separately:
  \`\`\`sh
  pip install tensorflow --no-cache-dir
  \`\`\`

## Contributors
- **Your Name** (replace with your details)

## License
This project is licensed under the MIT License." > README.md

