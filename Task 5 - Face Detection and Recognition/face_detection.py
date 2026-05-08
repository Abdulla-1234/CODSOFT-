"""
Task 5: Face Detection and Recognition
CodSoft AI Internship
Author: Doodakula Mohammad Abdulla

Detection  : OpenCV Haar Cascades (fast, CPU-friendly)
Recognition: LBPH (Local Binary Pattern Histogram) face recognizer
             — works without a GPU; no cloud API needed.

Install:
    pip install opencv-python opencv-contrib-python numpy pillow

Usage:
    python face_detection.py detect    --input photo.jpg
    python face_detection.py webcam                          # live detection
    python face_detection.py train     --dataset ./dataset   # build recognizer
    python face_detection.py recognize --input photo.jpg     # ID faces
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ─── Config ───────────────────────────────────────────────────────────────────

HAAR_FACE    = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
HAAR_EYE     = cv2.data.haarcascades + "haarcascade_eye.xml"
MODEL_PATH   = "face_model.yml"
LABELS_PATH  = "face_labels.txt"

DETECT_PARAMS = dict(scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))

# ─── Detector ─────────────────────────────────────────────────────────────────

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(HAAR_FACE)
        self.eye_cascade  = cv2.CascadeClassifier(HAAR_EYE)
        if self.face_cascade.empty():
            raise RuntimeError("Haar cascade not found. Reinstall opencv-python.")

    def detect(self, frame: np.ndarray):
        """Return list of (x, y, w, h) face rectangles."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, **DETECT_PARAMS)
        return faces if len(faces) > 0 else []

    def draw_faces(self, frame: np.ndarray, faces) -> np.ndarray:
        """Draw rectangles around detected faces."""
        out = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(out, (x, y), (x+w, y+h), (0, 200, 0), 2)
            cv2.putText(out, "Face", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        return out


# ─── Recognizer ───────────────────────────────────────────────────────────────

class FaceRecognizer:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.detector   = FaceDetector()
        self.labels: list[str] = []

    def train(self, dataset_dir: str):
        """
        Train from a directory structured as:
            dataset/
              Person_Name_1/
                img1.jpg
                img2.jpg
              Person_Name_2/
                ...
        """
        faces, label_ids = [], []
        dataset = Path(dataset_dir)
        self.labels = sorted([p.name for p in dataset.iterdir() if p.is_dir()])

        if not self.labels:
            raise RuntimeError(f"No sub-directories found in {dataset_dir}")

        for label_id, name in enumerate(self.labels):
            person_dir = dataset / name
            for img_path in person_dir.glob("*"):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                img  = cv2.imread(str(img_path))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                detected = self.detector.face_cascade.detectMultiScale(gray, **DETECT_PARAMS)
                for (x, y, w, h) in detected:
                    faces.append(gray[y:y+h, x:x+w])
                    label_ids.append(label_id)

        if not faces:
            raise RuntimeError("No faces found in the dataset. Check image paths.")

        print(f"Training on {len(faces)} face(s) for {len(self.labels)} person(s)…")
        self.recognizer.train(faces, np.array(label_ids))
        self.recognizer.save(MODEL_PATH)

        with open(LABELS_PATH, "w") as f:
            f.write("\n".join(self.labels))

        print(f"✅ Model saved → {MODEL_PATH}")
        print(f"   Persons: {', '.join(self.labels)}")

    def load(self):
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Model file '{MODEL_PATH}' not found. Run 'train' first."
            )
        self.recognizer.read(MODEL_PATH)
        with open(LABELS_PATH) as f:
            self.labels = [l.strip() for l in f.readlines()]
        print(f"✅ Model loaded. Known persons: {', '.join(self.labels)}")

    def recognize(self, frame: np.ndarray, confidence_threshold: float = 80):
        """
        Return annotated frame with recognized names.
        Lower LBPH confidence = better match (≤80 reliable).
        """
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.face_cascade.detectMultiScale(gray, **DETECT_PARAMS)
        out   = frame.copy()

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            label_id, conf = self.recognizer.predict(face_roi)

            if conf <= confidence_threshold:
                name  = self.labels[label_id]
                color = (0, 200, 0)
                text  = f"{name} ({conf:.1f})"
            else:
                name  = "Unknown"
                color = (0, 0, 220)
                text  = "Unknown"

            cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
            cv2.putText(out, text, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        return out, faces


# ─── Modes ────────────────────────────────────────────────────────────────────

def mode_detect(args):
    detector = FaceDetector()
    img = cv2.imread(args.input)
    if img is None:
        sys.exit(f"❌ Cannot read image: {args.input}")

    faces = detector.detect(img)
    print(f"🔍 Detected {len(faces)} face(s) in '{args.input}'.")
    out = detector.draw_faces(img, faces)

    out_path = args.output or "detected_" + Path(args.input).name
    cv2.imwrite(out_path, out)
    print(f"✅ Saved → {out_path}")


def mode_webcam(args):
    rec = FaceRecognizer()
    try:
        rec.load()  # Load the face_model.yml you just trained
    except FileNotFoundError:
        sys.exit("❌ Model not found. Run 'train' mode first.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("❌ Cannot open webcam.")

    print("📷 Live recognition running… Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Use the recognize method instead of draw_faces
        annotated_frame, faces = rec.recognize(frame)
        
        cv2.putText(annotated_frame, f"Faces: {len(faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.imshow("Face Recognition — CodSoft AI", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def mode_train(args):
    rec = FaceRecognizer()
    rec.train(args.dataset)


def mode_recognize(args):
    rec = FaceRecognizer()
    rec.load()

    img = cv2.imread(args.input)
    if img is None:
        sys.exit(f"❌ Cannot read image: {args.input}")

    out, faces = rec.recognize(img, confidence_threshold=args.threshold)
    print(f"🔍 Processed {len(faces)} face(s).")

    out_path = args.output or "recognized_" + Path(args.input).name
    cv2.imwrite(out_path, out)
    print(f"✅ Saved → {out_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  👁  Face Detection & Recognition | CodSoft AI Task 5")
    print("=" * 55 + "\n")

    parser = argparse.ArgumentParser(
        description="Face Detection and Recognition | CodSoft AI Internship"
    )
    sub = parser.add_subparsers(dest="mode")

    # detect
    p_det = sub.add_parser("detect", help="Detect faces in an image")
    p_det.add_argument("--input",  required=True, help="Input image path")
    p_det.add_argument("--output", default=None,  help="Output image path")

    # webcam
    sub.add_parser("webcam", help="Live face detection via webcam")

    # train
    p_train = sub.add_parser("train", help="Train face recognizer on a dataset")
    p_train.add_argument("--dataset", required=True,
                         help="Dataset directory (sub-folder per person)")

    # recognize
    p_rec = sub.add_parser("recognize", help="Recognize faces in an image")
    p_rec.add_argument("--input",     required=True, help="Input image path")
    p_rec.add_argument("--output",    default=None,  help="Output image path")
    p_rec.add_argument("--threshold", type=float, default=80,
                       help="Confidence threshold (lower = stricter, default 80)")

    args = parser.parse_args()

    dispatch = {
        "detect":    mode_detect,
        "webcam":    mode_webcam,
        "train":     mode_train,
        "recognize": mode_recognize,
    }

    if args.mode in dispatch:
        dispatch[args.mode](args)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python face_detection.py detect    --input photo.jpg")
        print("  python face_detection.py webcam")
        print("  python face_detection.py train     --dataset ./dataset")
        print("  python face_detection.py recognize --input photo.jpg")


if __name__ == "__main__":
    main()
