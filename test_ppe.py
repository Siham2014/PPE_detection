import cv2
from ultralytics import YOLO


def main():
    # Modèle YOLO général pour vérifier que l'installation fonctionne.
    model = YOLO("yolov8n.pt")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Impossible de lire l'image de la caméra.")
            break

        results = model.predict(
            source=frame,
            conf=0.5,
            verbose=False
        )

        annotated_frame = results[0].plot()

        cv2.imshow("Test YOLO", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()