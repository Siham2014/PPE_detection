from pathlib import Path

import cv2
from ultralytics import YOLO


# ==================================================
# Chemins des modèles
# ==================================================

PPE_MODEL_PATH = Path("models/best.pt")
PERSON_MODEL_PATH = Path("yolov8n.pt")


# ==================================================
# Paramètres
# ==================================================

PERSON_CONFIDENCE = 0.25
PPE_CONFIDENCE = 0.30

VEST_CLASS = "Safety Vest"

WINDOW_NAME = "Detection EPI - Controle du gilet"


# ==================================================
# Fonctions utilitaires
# ==================================================

def vest_belongs_to_person(person_box, vest_box):
    """
    Associe un gilet à une personne.

    La décision utilise :
    1. la position du centre du gilet ;
    2. le chevauchement du gilet avec la zone du torse.
    """

    px1, py1, px2, py2 = person_box
    vx1, vy1, vx2, vy2 = vest_box

    person_width = max(px2 - px1, 1)
    person_height = max(py2 - py1, 1)

    # Zone approximative du torse.
    # Elle est volontairement assez large pour éviter
    # de rejeter un gilet correctement détecté.
    torso_x1 = px1 + int(0.05 * person_width)
    torso_x2 = px1 + int(0.95 * person_width)

    torso_y1 = py1 + int(0.15 * person_height)
    torso_y2 = py1 + int(0.92 * person_height)

    # Centre du gilet.
    vest_center_x = (vx1 + vx2) // 2
    vest_center_y = (vy1 + vy2) // 2

    center_inside = (
        torso_x1 <= vest_center_x <= torso_x2
        and torso_y1 <= vest_center_y <= torso_y2
    )

    # Calcul de l'intersection entre le gilet et le torse.
    intersection_x1 = max(torso_x1, vx1)
    intersection_y1 = max(torso_y1, vy1)
    intersection_x2 = min(torso_x2, vx2)
    intersection_y2 = min(torso_y2, vy2)

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1,
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1,
    )

    intersection_area = (
        intersection_width * intersection_height
    )

    vest_area = max(
        (vx2 - vx1) * (vy2 - vy1),
        1,
    )

    overlap_ratio = intersection_area / vest_area

    # Le gilet est associé à la personne si :
    # - son centre est dans le torse ;
    # ou
    # - au moins 25 % de sa boîte chevauche le torse.
    return center_inside or overlap_ratio >= 0.25


def draw_person_status(frame, person_box, has_vest):
    """
    Dessine la boîte de la personne et affiche
    son statut concernant le gilet.
    """

    x1, y1, x2, y2 = person_box

    if has_vest:
        color = (0, 200, 0)
        text = "SAFETY VEST OK"
    else:
        color = (0, 0, 255)
        text = "NO SAFETY VEST"

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        3,
    )

    # Placer le texte à l'intérieur de la boîte,
    # pour éviter qu'il sorte de l'écran.
    text_x = max(x1 + 10, 10)
    text_y = max(y1 + 30, 35)

    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_detection(
    frame,
    box,
    label,
    color,
    confidence,
):
    """
    Dessine une détection YOLO classique.
    """

    x1, y1, x2, y2 = box

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    text = f"{label} {confidence:.2f}"

    text_x = max(x1, 5)
    text_y = max(y1 - 8, 20)

    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


# ==================================================
# Programme principal
# ==================================================

def main():
    # Vérifier la présence du modèle PPE.
    if not PPE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modele EPI introuvable : "
            f"{PPE_MODEL_PATH.resolve()}"
        )

    # Vérifier la présence du modèle général.
    if not PERSON_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modele de detection des personnes "
            f"introuvable : "
            f"{PERSON_MODEL_PATH.resolve()}"
        )

    # Modèle général COCO pour détecter les personnes.
    person_model = YOLO(str(PERSON_MODEL_PATH))

    # Modèle personnalisé pour détecter les EPI.
    ppe_model = YOLO(str(PPE_MODEL_PATH))

    print("Modeles charges correctement.")
    print("Classes du modele EPI :")
    print(ppe_model.names)

    # 0 = caméra intégrée du PC.
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Impossible d'ouvrir la camera."
        )

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print(
                    "Impossible de lire une image "
                    "depuis la camera."
                )
                break

            # ==========================================
            # 1. Détection des personnes
            # ==========================================

            person_result = person_model.predict(
                source=frame,
                classes=[0],
                conf=PERSON_CONFIDENCE,
                verbose=False,
            )[0]

            person_boxes = []

            for detected_box in person_result.boxes:
                x1, y1, x2, y2 = map(
                    int,
                    detected_box.xyxy[0].tolist(),
                )

                person_boxes.append(
                    (x1, y1, x2, y2)
                )

            # ==========================================
            # 2. Détection des EPI
            # ==========================================

            ppe_result = ppe_model.predict(
                source=frame,
                conf=PPE_CONFIDENCE,
                verbose=False,
            )[0]

            vest_boxes = []

            for detected_box in ppe_result.boxes:
                class_id = int(
                    detected_box.cls[0]
                )

                confidence = float(
                    detected_box.conf[0]
                )

                class_name = (
                    ppe_model.names[class_id]
                )

                x1, y1, x2, y2 = map(
                    int,
                    detected_box.xyxy[0].tolist(),
                )

                current_box = (
                    x1,
                    y1,
                    x2,
                    y2,
                )

                # Enregistrer les gilets pour
                # les associer aux personnes.
                if class_name == VEST_CLASS:
                    vest_boxes.append(current_box)

                    draw_detection(
                        frame=frame,
                        box=current_box,
                        label="Safety Vest",
                        color=(0, 255, 255),
                        confidence=confidence,
                    )

                # Ne pas dessiner une deuxième boîte
                # pour Person ou NO-Safety Vest.
                elif class_name not in {
                    "Person",
                    "NO-Safety Vest",
                }:
                    draw_detection(
                        frame=frame,
                        box=current_box,
                        label=class_name,
                        color=(255, 170, 0),
                        confidence=confidence,
                    )

            # ==========================================
            # 3. Vérification du gilet par personne
            # ==========================================

            for person_box in person_boxes:
                has_vest = any(
                    vest_belongs_to_person(
                        person_box,
                        vest_box,
                    )
                    for vest_box in vest_boxes
                )

                draw_person_status(
                    frame=frame,
                    person_box=person_box,
                    has_vest=has_vest,
                )

            # ==========================================
            # 4. Messages d'information
            # ==========================================

            if not person_boxes:
                cv2.putText(
                    frame,
                    "AUCUNE PERSONNE DETECTEE",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                "Q : quitter",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                WINDOW_NAME,
                frame,
            )

            if (
                cv2.waitKey(1) & 0xFF
                == ord("q")
            ):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()