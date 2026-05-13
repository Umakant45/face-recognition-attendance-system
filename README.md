# Face Recognition Attendance System

A mobile-friendly Face Recognition Attendance System built using Python, OpenCV, Flask, HTML, CSS, and JavaScript.

This project allows users to:

* Register faces
* Train a face recognition model
* Detect and recognize faces
* Mark attendance automatically
* Access the system from desktop and mobile browsers

---

# Features

* Face Detection using OpenCV
* Face Recognition using LBPH Recognizer
* Attendance Marking System
* CSV Attendance Storage
* Mobile Camera Support
* Flask Backend API
* Responsive Frontend UI
* GitHub Deployment Ready

---

# Tech Stack

## Frontend

* HTML
* CSS
* JavaScript

## Backend

* Python
* Flask
* OpenCV
* NumPy

---

# Project Structure

```bash
face-recognition-attendance-system/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── backend/
│   ├── app.py
│   ├── train_model.py
│   ├── dataset/
│   ├── trainer/
│   └── attendance.csv
│
├── requirements.txt
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Umakant45/face-recognition-attendance-system.git
```

## Move into Project Folder

```bash
cd face-recognition-attendance-system
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Required Python Libraries

```txt
flask
opencv-python-headless
numpy
gunicorn
```

---

# Run the Project

## Start Flask Server

```bash
python app.py
```

Server will run on:

```txt
http://127.0.0.1:5000
```

---

# Face Registration

1. Open the application
2. Allow camera permissions
3. Capture face samples
4. Save dataset
5. Train the model

---

# Attendance System

* Recognized faces are automatically marked present
* Attendance is stored in:

```txt
attendance.csv
```

---

# Mobile Access

This project supports mobile browsers.

Users can:

* Open the website on mobile
* Allow camera access
* Capture face
* Mark attendance remotely

---

# Deployment

## Frontend Deployment

Deploy frontend using:

* Netlify
* Vercel

## Backend Deployment

Deploy Flask backend using:

* Render
* Railway
* PythonAnywhere

---

# GitHub Push Commands

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/Umakant45/face-recognition-attendance-system.git
git push -u origin main
```

---

# Future Improvements

* Admin Dashboard
* Database Integration
* Email Notifications
* Firebase Support
* QR + Face Recognition
* Real-Time Monitoring
* Cloud Attendance Storage
* Student Management Panel

---

# Author

Umakant Kautkar

Linkedin: https://www.linkedin.com/in/umakant-kautkar-6a0180312/

---

# License

This project is open-source and available for educational purposes.
