from pathlib import Path
import argparse

from ppe_detector import PPEDetector


PROJECT_DIRECTORY = Path(__file__).resolve().parent

FUSION_MODEL = PROJECT_DIRECTORY / "models" / "best_fusion.pt"
OLD_MODEL = PROJECT_DIRECTORY / "models" / "best.pt"


def parse_source(source_value: str):
    """
    Convertit 0 en entier pour utiliser la webcam.
    """
    if source_value == "0":
        return 0

    return source_value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tester le modèle EPI entraîné sur le dataset fusionné."
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Chemin image/vidéo ou 0 pour la webcam.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Seuil de confiance. Valeur par défaut : 0.40.",
    )

    parser.add_argument(
        "--model",
        choices=["fusion", "ancien"],
        default="fusion",
        help="Choisir le modèle fusionné ou l'ancien modèle.",
    )

    args = parser.parse_args()

    source = parse_source(args.source)

    if args.model == "fusion":
        selected_model = FUSION_MODEL
        experiment_name = "modele_fusion"
    else:
        selected_model = OLD_MODEL
        experiment_name = "ancien_modele"

    print("=" * 60)
    print("TEST DE DÉTECTION DES EPI")
    print("=" * 60)
    print(f"Modèle sélectionné : {selected_model}")
    print(f"Source : {source}")
    print(f"Seuil de confiance : {args.conf}")
    print("=" * 60)

    detector = PPEDetector(
        model_path=selected_model,
        confidence=args.conf,
        image_size=640,
    )

    results = detector.detect(
        source=source,
        output_directory=PROJECT_DIRECTORY / "resultats_detection",
        experiment_name=experiment_name,
        save=True,
        show=False,
    )

    detector.print_detections(results)

    print(
        "\nRésultat enregistré dans : "
        f"{PROJECT_DIRECTORY / 'resultats_detection' / experiment_name}"
    )


if __name__ == "__main__":
    main()