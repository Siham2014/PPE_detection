from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_PATH = Path("models/best.pt")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH.resolve()}"
        )

    # Charger ton modèle PPE entraîné
    model = YOLO(str(MODEL_PATH))

    print("Modèle chargé.")
    print("Classes détectables :")
    print(model.names)

    # 0 = webcam du PC
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Impossible de lire l'image de la caméra.")
                break

            results = model.predict(
                source=frame,
                conf=0.40,
                verbose=False
            )

            annotated_frame = results[0].plot()

            cv2.imshow(
                "Detection EPI - YOLO",
                annotated_frame
            )

            # Appuyer sur Q pour quitter
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()