from pathlib import Path
from typing import Union

from ultralytics import YOLO


class PPEDetector:
    """
    Module de détection des équipements de protection individuelle.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        confidence: float = 0.40,
        image_size: int = 640,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.image_size = image_size

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Le modèle est introuvable : {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))

    def detect(
        self,
        source: Union[str, int, Path],
        output_directory: Union[str, Path] = "resultats_detection",
        experiment_name: str = "prediction",
        save: bool = True,
        show: bool = False,
    ):
        """
        Lance la détection sur une image, une vidéo ou une webcam.

        source :
            chemin image/vidéo ou 0 pour la webcam
        """

        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        if isinstance(source, Path):
            source = str(source)

        results = self.model.predict(
            source=source,
            conf=self.confidence,
            imgsz=self.image_size,
            save=save,
            show=show,
            project=str(output_directory),
            name=experiment_name,
            exist_ok=True,
            verbose=True,
        )

        return results

    def print_detections(self, results) -> None:
        """
        Affiche les classes détectées et leurs niveaux de confiance.
        """

        total_detections = 0

        for result_index, result in enumerate(results, start=1):
            print(f"\nRésultat {result_index}")

            if result.boxes is None or len(result.boxes) == 0:
                print("Aucun objet détecté")
                continue

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = self.model.names[class_id]

                total_detections += 1

                print(
                    f"- {class_name:<20} "
                    f"confiance : {confidence:.2%}"
                )

        print(f"\nNombre total de détections : {total_detections}")