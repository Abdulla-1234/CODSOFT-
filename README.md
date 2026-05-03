# 🤖 CODSOFT — AI Internship Tasks

**Intern:** Doodakula Mohammad Abdulla  
**Batch:** April 2026 · B93  
**Internship:** CodSoft Artificial Intelligence  

---

## 📋 Tasks Overview

| # | Task | Tech Used | Status |
|---|------|-----------|--------|
| 1 | Rule-Based Chatbot | Python, Regex | ✅ |
| 2 | Tic-Tac-Toe AI | Minimax + Alpha-Beta Pruning | ✅ |
| 3 | Image Captioning | BLIP (ViT + BERT), HuggingFace | ✅ |
| 4 | Recommendation System | Collaborative + Content-Based Filtering | ✅ |
| 5 | Face Detection & Recognition | OpenCV Haar Cascades, LBPH | ✅ |

---

## 🗂 Repository Structure

```
CODSOFT/
├── task1_chatbot/
│   └── chatbot.py
├── task2_tictactoe/
│   └── tictactoe.py
├── task3_image_captioning/
│   ├── image_captioning.py
│   └── requirements.txt
├── task4_recommendation/
│   ├── recommendation_system.py
│   └── requirements.txt
├── task5_face_detection/
│   ├── face_detection.py
│   └── requirements.txt
└── README.md
```

---

## 🚀 Task Details

### Task 1 — Rule-Based Chatbot
A pattern-matching chatbot using regular expressions to handle greetings, FAQs, jokes, time/date queries and more.

```bash
python task1_chatbot/chatbot.py
```

**Key features:**
- 20+ intent patterns with regex
- Callable responses (live time/date)
- Random response variation
- Graceful fallback handling

---

### Task 2 — Tic-Tac-Toe AI
An unbeatable AI using the **Minimax algorithm with Alpha-Beta Pruning**.

```bash
python task2_tictactoe/tictactoe.py
```

**Key features:**
- Perfect play guaranteed (AI never loses)
- Alpha-Beta pruning for efficiency
- Choose who goes first
- Clean CLI board display

---

### Task 3 — Image Captioning
Combines computer vision and NLP using **Salesforce BLIP** (Vision Transformer + BERT).

```bash
# Install dependencies
pip install torch torchvision transformers pillow requests gradio

# Caption an image from CLI
python task3_image_captioning/image_captioning.py caption photo.jpg

# Caption from URL
python task3_image_captioning/image_captioning.py caption https://example.com/img.jpg

# Launch Gradio web UI (great for demo video!)
python task3_image_captioning/image_captioning.py ui
```

**Key features:**
- State-of-the-art BLIP-large model
- Supports local files and URLs
- Optional conditional text prompting
- Gradio web UI for demos

---

### Task 4 — Recommendation System
Movie recommendation engine implementing three strategies:

```bash
pip install numpy pandas scikit-learn
python task4_recommendation/recommendation_system.py
```

**Key features:**
- **Collaborative Filtering** — user-user cosine similarity with mean-centering
- **Content-Based Filtering** — TF-IDF on genres, director, year
- **Hybrid** — weighted combination of both
- No external dataset download required (synthetic data included)

---

### Task 5 — Face Detection & Recognition
Real-time face detection with optional identity recognition.

```bash
pip install opencv-python opencv-contrib-python numpy

# Detect faces in an image
python task5_face_detection/face_detection.py detect --input photo.jpg

# Live detection via webcam
python task5_face_detection/face_detection.py webcam

# Train recognizer on your own dataset
python task5_face_detection/face_detection.py train --dataset ./dataset

# Recognize faces
python task5_face_detection/face_detection.py recognize --input photo.jpg
```

**Dataset structure for training:**
```
dataset/
  Alice/
    img1.jpg
    img2.jpg
  Bob/
    img1.jpg
```

**Key features:**
- Haar Cascade face detection
- LBPH face recognition (no GPU required)
- Configurable confidence threshold
- Supports images and live webcam

---

## ⚙️ General Requirements

- Python 3.8+
- Task-specific packages listed in each `requirements.txt`

---

## 📎 Links

- 🌐 [CodSoft Website](https://www.codsoft.in)
- 💼 [LinkedIn](https://www.linkedin.com) — tag @CodSoft with #codsoft
- 📁 [Task Submission Form](https://forms.gle/bDEXL1khDxx41oWg8)

---

*Built with ❤️ as part of the CodSoft AI Internship — April 2026*
