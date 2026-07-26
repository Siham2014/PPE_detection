import cv2
import os
import numpy as np
from deepface import DeepFace

DATASET_PATH = "dataset"

# =========================
# 1. CHARGER DATASET PERSONNES
# =========================
known_embeddings = {}

print("Chargement dataset...")

for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)

    if os.path.isdir(person_path):
        images = os.listdir(person_path)

        if len(images) > 0:
            img_path = os.path.join(person_path, images[0])

            embedding = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]

            known_embeddings[person] = embedding

print("Dataset chargé ✔")

# =========================
# 2. DISTANCE
# =========================
def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

# =========================
# 3. CAMERA
# =========================
cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 2 != 0:
        continue

    frame = cv2.resize(frame, (640, 480))

    faces = DeepFace.extract_faces(
        img_path=frame,
        detector_backend="opencv",
        enforce_detection=False
    )

    for face in faces:
        region = face["facial_area"]
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]

        face_img = frame[y:y+h, x:x+w]

        name = "Unknown"
        best_dist = 999

        try:
            embedding = DeepFace.represent(
                img_path=face_img,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]

            for person, ref_embedding in known_embeddings.items():
                dist = cosine_distance(embedding, ref_embedding)

                if dist < 10 and dist < best_dist:
                    best_dist = dist
                    name = person

        except:
            name = "Unknown"

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        label = name

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("FaceSense - Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()