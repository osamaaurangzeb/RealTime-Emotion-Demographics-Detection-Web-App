
# Face, Age-Gender & Speech Emotion Recognition Web App

A multi-modal AI-powered web application for real-time facial emotion detection, age and gender prediction, and speech-based emotion recognition. Built with **Flask**, **OpenCV**, **Keras/TensorFlow**, and deep learning models.

> 💡 Now supports **image upload** in addition to live webcam for facial analysis.  
> 🐍 Fully compatible with **Python 3.11**

---

## 🚀 Features

- **Facial Emotion Recognition (Live + Upload):** Detects emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) from webcam or uploaded images.
- **Age & Gender Prediction:** Predicts age and gender from detected faces in real time or from uploaded images.
- **Speech Emotion Recognition:** Records audio and predicts the speaker's emotion (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised).
- **Modular Flask Architecture:** Organized using blueprints for scalability and maintainability.
- **Interactive Web UI:** Clean, user-friendly interface for all features.

---

## 🧠 Deep Learning Models & Performance

### 1. Facial Emotion Recognition Model
- **Architecture:** CNN (trained on FER2013)
- **Test Accuracy:** 85.96%
- **Notes:** Robust to real-time webcam and uploaded images. Distinguishes seven basic emotions effectively.

### 2. Age & Gender Prediction Model
- **Architecture:** CNN (regression + classification)
- **Dataset:** Custom facial dataset with labeled age and gender.
- **Performance:** High classification accuracy and low MAE. Detailed evaluation in training notebook.

### 3. Speech Emotion Recognition Model
- **Architecture:** Deep neural network using MFCC audio features.
- **Dataset:** RAVDESS
- **Test Accuracy:** 88.1%
- **Notes:** Real-time audio emotion classification supporting eight categories.

---

## 📁 Project Structure

```
Face_Emotions/
│
├── Age-Gender Model/
│   ├── Age_Gender_model.ipynb
│   ├── age.h5
│   └── age_pickle.pkl
│
├── Expression Model/
│   ├── Facial Emotion Detection & Recognition using CNN.ipynb
│   ├── expression_emotion_model.h5
│   └── expression_emotion_model.keras
│
├── Speech Model/
│   ├── speech_model.ipynb
│   ├── speech_model.h5
│   └── speech_model_pickle.pkl
│
└── Flask WebAPP/
    ├── app.py
    └── blueprints/
        ├── home/
        ├── expression/    # Supports webcam & image upload
        └── speech/
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/osamaaurangzeb/osamaaurangzeb-Real-Time-Emotion-Demographics-Detection-Web-App.git
cd Face_Emotions/Flask\ WebAPP/
```

### 2. Create & Activate a Virtual Environment (Optional but Recommended)

```bash
python3.11 -m venv venv
source venv/bin/activate       # On Linux/macOS
venv\Scripts\activate        # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ✅ Tested on **Python 3.11+**

---

## ▶️ Run the Application :)

```bash
python app.py
```

Access the app at [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 🧪 Usage

| Page             | Description                                                  |
|------------------|--------------------------------------------------------------|
| `/home`          | Landing page                                                 |
| `/expression-age`| Real-time facial emotion + age/gender (webcam & image upload)|
| `/speech`        | Record audio and predict speech emotion                      |

---

## 🎓 Model Training

- Each model has an associated Jupyter notebook for training.
- Datasets are not included — follow instructions in notebooks to download and preprocess them.

---

## 📄 File Descriptions

- `app.py`: Main entry point, registers all Flask blueprints.
- `blueprints/expression/expression.py`: Handles image uploads, video streaming, emotion, and age-gender predictions.
- `blueprints/speech/speech.py`: Records audio and predicts speech emotions.
- `blueprints/home/home.py`: Serves the homepage.

---

## 📷 Image Upload Support

- Navigate to `/expression-age` to upload an image.
- Detects face(s) in uploaded images and performs emotion + age/gender analysis.
- Supports `.jpg`, `.jpeg`, `.png` formats.

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for full terms.

---

## 🙏 Acknowledgements

- [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) — Facial Emotion Dataset  
- [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) — Speech Emotion Dataset  
- Open-source libraries: Keras, TensorFlow, Flask, OpenCV, NumPy, Librosa, etc.
