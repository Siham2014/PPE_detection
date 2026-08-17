from pathlib import Path
import argparse
import time

import cv2
from ultralytics import YOLO


# ==========================================================
# CHEMINS
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent

# Nouveau modèle entraîné après fusion des datasets.
# Modifie seulement ce nom selon le vrai nom de ton nouveau modèle.
PPE_MODEL_PATH = PROJECT_DIR / "models" / "best_fusion.pt"

# Modèle YOLO général utilisé pour détecter les personnes.
# Ultralytics le téléchargera automatiquement s'il n'existe pas.
PERSON_MODEL_PATH = PROJECT_DIR / "yolov8n.pt"

RESULTS_DIR = PROJECT_DIR / "resultats_detection"

WINDOW_NAME = "Detection EPI"


# ==========================================================
# PARAMÈTRES
# ==========================================================

PERSON_CONFIDENCE = 0.25
PPE_CONFIDENCE = 0.30
IMAGE_SIZE = 640

VEST_CLASS_NAMES = {
    "Safety Vest",
    "Safety_Vest",
    "Vest",
}

HELMET_CLASS_NAMES = {
    "Helmet",
    "Hardhat",
}

SHOES_CLASS_NAMES = {
    "Shoes",
    "Safety Shoes",
    "Safety_Shoes",
    "Safety Boots",
    "Boots",
}

# Classes utiles pour l'analyse, mais que nous ne dessinons pas
# pour éviter trop de rectangles sur l'image.
IGNORED_CLASSES = {
    "Person",
    "Head",
    "Face",
    "Ear",
}


# ==========================================================
# FONCTIONS GÉOMÉTRIQUES
# ==========================================================

def normalize_class_name(class_name):
    """
    Nettoie le nom d'une classe YOLO.
    """
    return str(class_name).strip()


def box_center(box):
    """
    Retourne le centre d'une boîte.
    """

    x1, y1, x2, y2 = box

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    return center_x, center_y


def box_area(box):
    """
    Retourne l'aire d'une boîte.
    """

    x1, y1, x2, y2 = box

    width = max(x2 - x1, 0)
    height = max(y2 - y1, 0)

    return width * height


def intersection_area(box_a, box_b):
    """
    Calcule l'aire d'intersection entre deux boîtes.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1,
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1,
    )

    return intersection_width * intersection_height


def intersection_ratio(reference_box, object_box):
    """
    Calcule la proportion de object_box située
    à l'intérieur de reference_box.
    """

    object_area = max(
        box_area(object_box),
        1,
    )

    overlap_area = intersection_area(
        reference_box,
        object_box,
    )

    return overlap_area / object_area


def center_is_inside(reference_box, object_box):
    """
    Vérifie si le centre de object_box se trouve
    dans reference_box.
    """

    center_x, center_y = box_center(object_box)

    x1, y1, x2, y2 = reference_box

    return (
        x1 <= center_x <= x2
        and y1 <= center_y <= y2
    )


# ==========================================================
# ASSOCIATION DU GILET À UNE PERSONNE
# ==========================================================

def vest_belongs_to_person(person_box, vest_box):
    """
    Vérifie si le gilet se trouve dans la zone du torse.
    """

    px1, py1, px2, py2 = person_box

    person_width = max(px2 - px1, 1)
    person_height = max(py2 - py1, 1)

    torso_zone = (
        px1 + int(0.05 * person_width),
        py1 + int(0.15 * person_height),
        px1 + int(0.95 * person_width),
        py1 + int(0.82 * person_height),
    )

    center_inside = center_is_inside(
        torso_zone,
        vest_box,
    )

    overlap = intersection_ratio(
        torso_zone,
        vest_box,
    )

    return center_inside or overlap >= 0.25


# ==========================================================
# ASSOCIATION DU CASQUE À UNE PERSONNE
# ==========================================================

def helmet_belongs_to_person(person_box, helmet_box):
    """
    Vérifie si le casque se trouve dans la zone de la tête.

    Un casque tenu dans la main ne doit pas être considéré
    comme un casque porté.
    """

    px1, py1, px2, py2 = person_box

    person_width = max(px2 - px1, 1)
    person_height = max(py2 - py1, 1)

    head_zone = (
        px1 + int(0.05 * person_width),
        py1 - int(0.05 * person_height),
        px1 + int(0.95 * person_width),
        py1 + int(0.38 * person_height),
    )

    center_inside = center_is_inside(
        head_zone,
        helmet_box,
    )

    overlap = intersection_ratio(
        head_zone,
        helmet_box,
    )

    return center_inside and overlap >= 0.15


# ==========================================================
# ASSOCIATION DES CHAUSSURES À UNE PERSONNE
# ==========================================================

def shoes_belong_to_person(person_box, shoes_box):
    """
    Vérifie si les chaussures se trouvent dans la zone
    inférieure de la personne.
    """

    px1, py1, px2, py2 = person_box

    person_width = max(px2 - px1, 1)
    person_height = max(py2 - py1, 1)

    feet_zone = (
        px1 - int(0.12 * person_width),
        py1 + int(0.70 * person_height),
        px2 + int(0.12 * person_width),
        py2 + int(0.08 * person_height),
    )

    center_inside = center_is_inside(
        feet_zone,
        shoes_box,
    )

    overlap = intersection_ratio(
        feet_zone,
        shoes_box,
    )

    return center_inside or overlap >= 0.20


def feet_are_visible(person_box, frame_height):
    """
    Estime si la partie basse de la personne est visible.

    Si la boîte de la personne touche presque le bas
    de l'image, les pieds sont probablement coupés.
    """

    _, _, _, person_y2 = person_box

    bottom_margin = int(
        0.04 * frame_height
    )

    return person_y2 < (
        frame_height - bottom_margin
    )


# ==========================================================
# DESSIN DES DÉTECTIONS
# ==========================================================

def draw_detection(
    frame,
    box,
    label,
    confidence,
    color,
):
    """
    Dessine une boîte YOLO et son étiquette.
    """

    x1, y1, x2, y2 = box

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    text = f"{label} {confidence:.0%}"

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


def draw_person_status(
    frame,
    person_box,
    vest_status,
    helmet_status,
    shoes_status,
    global_status,
):
    """
    Dessine la boîte de la personne et son statut complet.
    """

    x1, y1, x2, y2 = person_box

    if global_status == "CONFORME":
        color = (0, 200, 0)

    elif global_status == "A VERIFIER":
        color = (0, 165, 255)

    else:
        color = (0, 0, 255)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        3,
    )

    lines = [
        global_status,
        f"Gilet : {vest_status}",
        f"Casque : {helmet_status}",
        f"Chaussures : {shoes_status}",
    ]

    start_y = max(
        y1 + 25,
        30,
    )

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (
                max(x1 + 5, 5),
                start_y + index * 24,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )


# ==========================================================
# CALCUL DU STATUT D'UNE PERSONNE
# ==========================================================

def calculate_person_status(
    person_box,
    vest_boxes,
    helmet_boxes,
    shoes_boxes,
    frame_height,
):
    """
    Calcule la conformité EPI d'une personne.
    """

    has_vest = any(
        vest_belongs_to_person(
            person_box,
            vest_box,
        )
        for vest_box in vest_boxes
    )

    has_helmet = any(
        helmet_belongs_to_person(
            person_box,
            helmet_box,
        )
        for helmet_box in helmet_boxes
    )

    has_shoes = any(
        shoes_belong_to_person(
            person_box,
            shoes_box,
        )
        for shoes_box in shoes_boxes
    )

    feet_visible = feet_are_visible(
        person_box,
        frame_height,
    )

    vest_status = (
        "OK"
        if has_vest
        else "MANQUANT"
    )

    helmet_status = (
        "OK"
        if has_helmet
        else "MANQUANT"
    )

    if not feet_visible:
        shoes_status = "NON VISIBLES"

    elif has_shoes:
        shoes_status = "OK"

    else:
        shoes_status = "MANQUANTES"

    # Cas totalement conforme.
    if (
        has_vest
        and has_helmet
        and has_shoes
        and feet_visible
    ):
        global_status = "CONFORME"

    # Si les chaussures ne peuvent pas être vérifiées,
    # le statut reste à vérifier uniquement lorsque
    # le casque et le gilet sont présents.
    elif (
        has_vest
        and has_helmet
        and not feet_visible
    ):
        global_status = "A VERIFIER"

    else:
        global_status = "NON CONFORME"

    return {
        "has_vest": has_vest,
        "has_helmet": has_helmet,
        "has_shoes": has_shoes,
        "feet_visible": feet_visible,
        "vest_status": vest_status,
        "helmet_status": helmet_status,
        "shoes_status": shoes_status,
        "global_status": global_status,
    }


# ==========================================================
# TRAITEMENT D'UNE IMAGE
# ==========================================================

def process_frame(
    frame,
    person_model,
    ppe_model,
):
    """
    Analyse une image et retourne :
    - l'image annotée ;
    - les détections du modèle PPE ;
    - les statuts des personnes.
    """

    annotated_frame = frame.copy()

    frame_height = frame.shape[0]

    # ------------------------------------------------------
    # 1. Détection des personnes avec YOLO COCO
    # ------------------------------------------------------

    person_result = person_model.predict(
        source=frame,
        classes=[0],
        conf=PERSON_CONFIDENCE,
        imgsz=IMAGE_SIZE,
        verbose=False,
    )[0]

    person_boxes = []

    if person_result.boxes is not None:
        for detected_box in person_result.boxes:
            coordinates = (
                detected_box
                .xyxy[0]
                .tolist()
            )

            x1, y1, x2, y2 = map(
                int,
                coordinates,
            )

            person_boxes.append(
                (x1, y1, x2, y2)
            )

    # ------------------------------------------------------
    # 2. Détection des EPI avec le modèle personnalisé
    # ------------------------------------------------------

    ppe_result = ppe_model.predict(
        source=frame,
        conf=PPE_CONFIDENCE,
        imgsz=IMAGE_SIZE,
        verbose=False,
    )[0]

    vest_boxes = []
    helmet_boxes = []
    shoes_boxes = []

    detections_summary = []

    if ppe_result.boxes is not None:
        for detected_box in ppe_result.boxes:
            class_id = int(
                detected_box.cls[0].item()
            )

            confidence = float(
                detected_box.conf[0].item()
            )

            class_name = normalize_class_name(
                ppe_model.names[class_id]
            )

            coordinates = (
                detected_box
                .xyxy[0]
                .tolist()
            )

            x1, y1, x2, y2 = map(
                int,
                coordinates,
            )

            current_box = (
                x1,
                y1,
                x2,
                y2,
            )

            detections_summary.append(
                {
                    "class_name": class_name,
                    "confidence": confidence,
                    "box": current_box,
                }
            )

            if class_name in VEST_CLASS_NAMES:
                vest_boxes.append(
                    current_box
                )

                draw_detection(
                    frame=annotated_frame,
                    box=current_box,
                    label="Safety Vest",
                    confidence=confidence,
                    color=(0, 255, 255),
                )

            elif class_name in HELMET_CLASS_NAMES:
                helmet_boxes.append(
                    current_box
                )

                draw_detection(
                    frame=annotated_frame,
                    box=current_box,
                    label="Helmet",
                    confidence=confidence,
                    color=(255, 255, 0),
                )

            elif class_name in SHOES_CLASS_NAMES:
                shoes_boxes.append(
                    current_box
                )

                draw_detection(
                    frame=annotated_frame,
                    box=current_box,
                    label="Safety Shoes",
                    confidence=confidence,
                    color=(255, 0, 255),
                )

            elif class_name not in IGNORED_CLASSES:
                draw_detection(
                    frame=annotated_frame,
                    box=current_box,
                    label=class_name,
                    confidence=confidence,
                    color=(255, 170, 0),
                )

    # ------------------------------------------------------
    # 3. Calcul du statut de chaque personne
    # ------------------------------------------------------

    persons_status = []

    for person_index, person_box in enumerate(
        person_boxes,
        start=1,
    ):
        status = calculate_person_status(
            person_box=person_box,
            vest_boxes=vest_boxes,
            helmet_boxes=helmet_boxes,
            shoes_boxes=shoes_boxes,
            frame_height=frame_height,
        )

        status["person"] = person_index
        status["box"] = person_box

        persons_status.append(status)

        draw_person_status(
            frame=annotated_frame,
            person_box=person_box,
            vest_status=status["vest_status"],
            helmet_status=status["helmet_status"],
            shoes_status=status["shoes_status"],
            global_status=status["global_status"],
        )

    if not person_boxes:
        cv2.putText(
            annotated_frame,
            "AUCUNE PERSONNE DETECTEE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

    return (
        annotated_frame,
        detections_summary,
        persons_status,
    )


# ==========================================================
# AFFICHAGE DU RÉSUMÉ DANS LE TERMINAL
# ==========================================================

def print_summary(
    detections,
    persons_status,
):
    """
    Affiche toutes les détections et la conformité.
    """

    print("\n" + "=" * 70)
    print("DETECTIONS BRUTES DU MODELE PPE")
    print("=" * 70)

    if not detections:
        print("Aucune detection PPE.")

    for index, detection in enumerate(
        detections,
        start=1,
    ):
        print(
            f"{index:02d}. "
            f"{detection['class_name']:<20} "
            f"{detection['confidence']:.2%}"
        )

    print("\n" + "=" * 70)
    print("CONTROLE DE CONFORMITE PAR PERSONNE")
    print("=" * 70)

    if not persons_status:
        print("Aucune personne detectee.")

    for status in persons_status:
        print(
            f"\nPersonne {status['person']}"
        )

        print(
            f"Statut global : "
            f"{status['global_status']}"
        )

        print(
            f"Gilet : "
            f"{status['vest_status']}"
        )

        print(
            f"Casque : "
            f"{status['helmet_status']}"
        )

        print(
            f"Chaussures : "
            f"{status['shoes_status']}"
        )


# ==========================================================
# MODE IMAGE
# ==========================================================

def run_image(
    image_path,
    person_model,
    ppe_model,
):
    """
    Analyse une image, affiche le résultat
    et l'enregistre.
    """

    image_path = Path(image_path)

    if not image_path.is_absolute():
        image_path = PROJECT_DIR / image_path

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image introuvable : "
            f"{image_path.resolve()}"
        )

    frame = cv2.imread(
        str(image_path)
    )

    if frame is None:
        raise RuntimeError(
            "OpenCV ne peut pas lire cette image."
        )

    (
        annotated_frame,
        detections,
        persons_status,
    ) = process_frame(
        frame=frame,
        person_model=person_model,
        ppe_model=ppe_model,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / f"{image_path.stem}_resultat.jpg"
    )

    success = cv2.imwrite(
        str(output_path),
        annotated_frame,
    )

    if not success:
        raise RuntimeError(
            "Impossible d'enregistrer l'image resultat."
        )

    print_summary(
        detections=detections,
        persons_status=persons_status,
    )

    print(
        "\nImage resultat enregistree dans :"
    )

    print(
        output_path.resolve()
    )

    cv2.imshow(
        WINDOW_NAME,
        annotated_frame,
    )

    print(
        "\nAppuie sur une touche pour fermer."
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# MODE CAMÉRA EN TEMPS RÉEL
# ==========================================================

def run_camera(
    camera_index,
    person_model,
    ppe_model,
):
    """
    Lance la détection avec la caméra.
    """

    camera = cv2.VideoCapture(
        camera_index
    )

    if not camera.isOpened():
        raise RuntimeError(
            f"Impossible d'ouvrir la camera "
            f"numero {camera_index}."
        )

    previous_time = time.perf_counter()

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print(
                    "Impossible de lire une image "
                    "depuis la camera."
                )
                break

            (
                annotated_frame,
                _,
                persons_status,
            ) = process_frame(
                frame=frame,
                person_model=person_model,
                ppe_model=ppe_model,
            )

            current_time = time.perf_counter()

            elapsed_time = max(
                current_time - previous_time,
                0.0001,
            )

            fps = 1.0 / elapsed_time

            previous_time = current_time

            compliant_count = sum(
                status["global_status"]
                == "CONFORME"
                for status in persons_status
            )

            non_compliant_count = sum(
                status["global_status"]
                == "NON CONFORME"
                for status in persons_status
            )

            verification_count = sum(
                status["global_status"]
                == "A VERIFIER"
                for status in persons_status
            )

            cv2.putText(
                annotated_frame,
                f"FPS : {fps:.1f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                annotated_frame,
                (
                    f"Conformes : {compliant_count} | "
                    f"Non conformes : "
                    f"{non_compliant_count} | "
                    f"A verifier : {verification_count}"
                ),
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                annotated_frame,
                "Q : quitter",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                WINDOW_NAME,
                annotated_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


# ==========================================================
# PROGRAMME PRINCIPAL
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detection du casque, du gilet et "
            "des chaussures de securite."
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Chemin d'une image ou numero de la camera. "
            "Exemples : test.webp ou 0."
        ),
    )

    args = parser.parse_args()

    if not PPE_MODEL_PATH.exists():
        raise FileNotFoundError(
            "\nNouveau modele PPE introuvable :\n"
            f"{PPE_MODEL_PATH.resolve()}\n\n"
            "Verifie le nom du nouveau modele "
            "dans le dossier models."
        )

    print("=" * 70)
    print("CHARGEMENT DES MODELES")
    print("=" * 70)

    print(
        f"Modele PPE : "
        f"{PPE_MODEL_PATH.resolve()}"
    )

    person_model = YOLO(
        str(PERSON_MODEL_PATH)
    )

    ppe_model = YOLO(
        str(PPE_MODEL_PATH)
    )

    print("\nModeles charges correctement.")

    print("\nClasses du modele PPE :")

    for class_id, class_name in ppe_model.names.items():
        print(
            f"{class_id:02d} : {class_name}"
        )

    print("=" * 70)

    if args.source.isdigit():
        run_camera(
            camera_index=int(args.source),
            person_model=person_model,
            ppe_model=ppe_model,
        )

    else:
        run_image(
            image_path=args.source,
            person_model=person_model,
            ppe_model=ppe_model,
        )


if __name__ == "__main__":
    main()