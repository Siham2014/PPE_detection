from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_PATH = Path("models/best.pt")

# Seuil minimal d'acceptation d'une détection.
CONFIDENCE_THRESHOLD = 0.40

# Nom exact des classes dans ton modèle.
PERSON_CLASS = "Person"
VEST_CLASS = "Safety Vest"
NO_VEST_CLASS = "NO-Safety Vest"


def box_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    """Retourne le centre d'une boîte englobante."""
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2


def is_vest_inside_torso(
    person_box: tuple[int, int, int, int],
    vest_box: tuple[int, int, int, int],
) -> bool:
    """
    Vérifie si le centre du gilet se situe dans la zone du torse
    de la personne.

    La zone du torse correspond approximativement à :
    - 15 % à 75 % de la largeur de la personne ;
    - 20 % à 70 % de sa hauteur.
    """
    px1, py1, px2, py2 = person_box

    person_width = px2 - px1
    person_height = py2 - py1

    torso_x1 = px1 + int(0.15 * person_width)
    torso_x2 = px1 + int(0.85 * person_width)

    torso_y1 = py1 + int(0.20 * person_height)
    torso_y2 = py1 + int(0.70 * person_height)

    vest_center_x, vest_center_y = box_center(vest_box)

    return (
        torso_x1 <= vest_center_x <= torso_x2
        and torso_y1 <= vest_center_y <= torso_y2
    )


def draw_status(
    frame,
    person_box: tuple[int, int, int, int],
    has_vest: bool,
) -> None:
    """Dessine la boîte de la personne et son statut de conformité."""
    x1, y1, x2, y2 = person_box

    if has_vest:
        color = (0, 200, 0)
        status = "SAFETY VEST OK"
    else:
        color = (0, 0, 255)
        status = "NO SAFETY VEST"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    label_y = max(y1 - 10, 30)

    cv2.putText(
        frame,
        status,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH.resolve()}"
        )

    model = YOLO(str(MODEL_PATH))

    print("Modèle chargé correctement.")
    print("Classes disponibles :", model.names)

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
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )

            result = results[0]

            person_boxes: list[tuple[int, int, int, int]] = []
            vest_boxes: list[tuple[int, int, int, int]] = []
            explicit_no_vest_boxes: list[
                tuple[int, int, int, int]
            ] = []

            # Extraire les détections utiles.
            for detected_box in result.boxes:
                class_id = int(detected_box.cls[0])
                confidence = float(detected_box.conf[0])
                class_name = model.names[class_id]

                x1, y1, x2, y2 = map(
                    int,
                    detected_box.xyxy[0].tolist(),
                )

                coordinates = (x1, y1, x2, y2)

                if class_name == PERSON_CLASS:
                    person_boxes.append(coordinates)

                elif class_name == VEST_CLASS:
                    vest_boxes.append(coordinates)

                elif class_name == NO_VEST_CLASS:
                    explicit_no_vest_boxes.append(coordinates)

                # Afficher les autres classes détectées.
                elif confidence >= CONFIDENCE_THRESHOLD:
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 170, 0),
                        2,
                    )

                    label = f"{class_name} {confidence:.2f}"

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 7, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 170, 0),
                        2,
                        cv2.LINE_AA,
                    )

            # Vérifier le gilet pour chaque personne.
            for person_box in person_boxes:
                has_vest = any(
                    is_vest_inside_torso(person_box, vest_box)
                    for vest_box in vest_boxes
                )

                # Si YOLO détecte explicitement NO-Safety Vest,
                # on garde également cette information.
                explicit_no_vest = any(
                    is_vest_inside_torso(
                        person_box,
                        no_vest_box,
                    )
                    for no_vest_box in explicit_no_vest_boxes
                )

                if explicit_no_vest:
                    has_vest = False

                draw_status(
                    frame=frame,
                    person_box=person_box,
                    has_vest=has_vest,
                )

            cv2.putText(
                frame,
                "Appuyer sur Q pour quitter",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Detection EPI - Controle du gilet",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()