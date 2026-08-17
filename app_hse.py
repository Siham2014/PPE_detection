from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import av
    from streamlit_webrtc import (
        RTCConfiguration,
        VideoProcessorBase,
        WebRtcMode,
        webrtc_streamer,
    )

    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


# ==========================================================
# CONFIGURATION GÉNÉRALE
# ==========================================================

st.set_page_config(
    page_title="Plateforme HSE Intelligente",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).resolve().parent

# Modifie seulement ce nom si ton nouveau modèle porte un autre nom.
PPE_MODEL_PATH = PROJECT_DIR / "models" / "best_fusion.pt"

RESULTS_DIR = PROJECT_DIR / "resultats_detection"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIDENCE = 0.30
IMAGE_SIZE = 640

PPE_CLASSES = [
    "Person",
    "Head",
    "Face",
    "Glasses",
    "Medical Mask",
    "Face Guard",
    "Ear",
    "Earmuffs",
    "Hands",
    "Gloves",
    "Feet",
    "Shoes",
    "Safety Vest",
    "Tools",
    "Helmet",
    "Medical Suit",
    "Safety Suit",
    "Fall Detected",
    "Ladder",
    "Safety Cone",
    "Mask",
]

VEST_NAMES = {
    "Safety Vest",
    "Safety_Vest",
    "Vest",
}

HELMET_NAMES = {
    "Helmet",
    "Hardhat",
}

SHOES_NAMES = {
    "Shoes",
    "Safety Shoes",
    "Safety_Shoes",
    "Safety Boots",
    "Boots",
}

GLOVES_NAMES = {
    "Gloves",
    "Safety Gloves",
}

GLASSES_NAMES = {
    "Glasses",
    "Safety Glasses",
}

MASK_NAMES = {
    "Mask",
    "Medical Mask",
}

EAR_NAMES = {
    "Earmuffs",
    "Ear Protection",
}


# ==========================================================
# STYLE CSS
# ==========================================================

st.markdown(
    """
    <style>
        :root {
            --ocp-dark: #0b4f3c;
            --ocp-medium: #15805f;
            --ocp-light: #35b886;
            --ocp-soft: #eaf7f2;
            --anthracite: #263238;
            --warning: #f59e0b;
            --danger: #dc2626;
            --info: #2563eb;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(53,184,134,0.08), transparent 28%),
                linear-gradient(180deg, #f8fbfa 0%, #eef4f2 100%);
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, #083f31 0%, #0b5a43 100%);
        }

        [data-testid="stSidebar"] * {
            color: white;
        }

        [data-testid="stSidebar"] .stRadio label {
            padding: 9px 12px;
            border-radius: 10px;
            margin-bottom: 4px;
            transition: 0.2s;
        }

        [data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(255, 255, 255, 0.12);
        }

        .main-header {
            background:
                linear-gradient(
                    110deg,
                    rgba(4, 54, 40, 0.97),
                    rgba(14, 112, 82, 0.90)
                );
            padding: 32px;
            border-radius: 22px;
            color: white;
            margin-bottom: 24px;
            box-shadow: 0 18px 45px rgba(5, 72, 52, 0.20);
        }

        .main-header h1 {
            margin: 0;
            font-size: 2.25rem;
            font-weight: 800;
        }

        .main-header p {
            margin-top: 10px;
            margin-bottom: 0;
            font-size: 1.05rem;
            opacity: 0.92;
            max-width: 920px;
        }

        .section-title {
            font-weight: 800;
            color: #15382f;
            margin-top: 12px;
            margin-bottom: 16px;
            font-size: 1.35rem;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid rgba(20, 100, 75, 0.10);
            border-radius: 18px;
            padding: 20px;
            min-height: 140px;
            box-shadow: 0 10px 28px rgba(28, 70, 56, 0.08);
            transition: transform 0.20s ease;
        }

        .metric-card:hover {
            transform: translateY(-3px);
        }

        .metric-icon {
            font-size: 1.65rem;
        }

        .metric-label {
            color: #60736d;
            font-weight: 600;
            margin-top: 11px;
        }

        .metric-value {
            color: #113f32;
            font-size: 1.85rem;
            font-weight: 850;
            margin-top: 4px;
        }

        .metric-delta {
            color: #16805f;
            font-size: 0.82rem;
            font-weight: 650;
            margin-top: 5px;
        }

        .info-card {
            background: white;
            border-radius: 18px;
            padding: 20px;
            border: 1px solid rgba(20, 100, 75, 0.10);
            box-shadow: 0 10px 28px rgba(28, 70, 56, 0.07);
            margin-bottom: 14px;
        }

        .zone-card {
            background: white;
            border-left: 6px solid #15805f;
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 10px 26px rgba(28, 70, 56, 0.08);
        }

        .risk-high {
            color: #dc2626;
            font-weight: 800;
        }

        .risk-medium {
            color: #f59e0b;
            font-weight: 800;
        }

        .status-ok {
            background: #dcfce7;
            color: #166534;
            padding: 5px 10px;
            border-radius: 999px;
            font-weight: 750;
        }

        .status-warning {
            background: #fef3c7;
            color: #92400e;
            padding: 5px 10px;
            border-radius: 999px;
            font-weight: 750;
        }

        .status-danger {
            background: #fee2e2;
            color: #991b1b;
            padding: 5px 10px;
            border-radius: 999px;
            font-weight: 750;
        }

        .alert-item {
            border-left: 5px solid #dc2626;
            background: white;
            padding: 14px 18px;
            border-radius: 12px;
            margin-bottom: 10px;
            box-shadow: 0 5px 14px rgba(0,0,0,0.05);
        }

        div.stButton > button {
            border-radius: 11px;
            border: none;
            font-weight: 750;
            min-height: 42px;
        }

        div.stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #0b654b, #1a9b70);
        }

        div[data-testid="stFileUploader"] {
            border: 2px dashed rgba(21, 128, 95, 0.35);
            border-radius: 18px;
            padding: 10px;
            background: rgba(255, 255, 255, 0.75);
        }

        .small-text {
            color: #60736d;
            font-size: 0.88rem;
        }

        .footer {
            margin-top: 35px;
            text-align: center;
            color: #6b7b76;
            font-size: 0.82rem;
        }

        .profile-box {
            background: rgba(255,255,255,0.10);
            padding: 14px;
            border-radius: 14px;
            margin-bottom: 15px;
        }

        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 1.65rem;
            }

            .main-header {
                padding: 22px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# DONNÉES MOCK
# ==========================================================

ZONES = [
    {
        "id": 1,
        "name": "Atelier bulls et camions",
        "service": "335",
        "description": (
            "Zone de maintenance mécanique des camions miniers, "
            "bulls et engins lourds."
        ),
        "equipment": "20 camions, 24 bulls, environ 30 engins divers",
        "equipment_count": 74,
        "ppe": ["Helmet", "Safety Vest", "Shoes", "Gloves", "Glasses"],
        "risk": "Élevé",
        "camera": "CAM-335-01",
        "camera_status": "Active",
        "analyses": 354,
        "compliance": 81.2,
        "violations": 67,
        "last_detection": "Aujourd’hui à 15:42",
    },
    {
        "id": 2,
        "name": "Atelier électrique",
        "service": "336",
        "description": (
            "Maintenance des armoires, moteurs et équipements électriques."
        ),
        "equipment": "Armoires électriques, moteurs, équipements électriques",
        "equipment_count": 36,
        "ppe": ["Helmet", "Gloves", "Shoes", "Glasses", "Face Guard"],
        "risk": "Élevé",
        "camera": "CAM-336-01",
        "camera_status": "Active",
        "analyses": 221,
        "compliance": 88.7,
        "violations": 28,
        "last_detection": "Aujourd’hui à 14:18",
    },
    {
        "id": 3,
        "name": "Atelier machines",
        "service": "337",
        "description": (
            "Zone d’entretien des draglines, pelles, sondeuses et chargeuses."
        ),
        "equipment": (
            "4 draglines, 1 pelle en butte, 1 sondeuse électrique, "
            "6 pelles hydrauliques, 4 sondeuses Diesel, 1 chargeuse"
        ),
        "equipment_count": 17,
        "ppe": [
            "Helmet",
            "Safety Vest",
            "Shoes",
            "Gloves",
            "Glasses",
            "Earmuffs",
        ],
        "risk": "Élevé",
        "camera": "CAM-337-02",
        "camera_status": "Maintenance",
        "analyses": 284,
        "compliance": 76.9,
        "violations": 83,
        "last_detection": "Hier à 17:56",
    },
    {
        "id": 4,
        "name": "La trémie",
        "service": "Sidi Chennane",
        "description": (
            "Zone de déchargement, trémie et convoyeurs de manutention."
        ),
        "equipment": "Trémie, convoyeurs, zone de déchargement",
        "equipment_count": 12,
        "ppe": [
            "Helmet",
            "Safety Vest",
            "Shoes",
            "Glasses",
            "Mask",
            "Earmuffs",
        ],
        "risk": "Élevé",
        "camera": "CAM-TRM-01",
        "camera_status": "Active",
        "analyses": 198,
        "compliance": 72.5,
        "violations": 94,
        "last_detection": "Aujourd’hui à 16:08",
    },
    {
        "id": 5,
        "name": "Parc matériel",
        "service": "Sidi Chennane",
        "description": (
            "Zone de stationnement et de circulation des engins lourds."
        ),
        "equipment": "20 camions, 24 bulls, environ 30 engins divers",
        "equipment_count": 74,
        "ppe": ["Helmet", "Safety Vest", "Shoes", "Gloves", "Glasses"],
        "risk": "Moyen",
        "camera": "CAM-PARC-03",
        "camera_status": "Active",
        "analyses": 191,
        "compliance": 89.4,
        "violations": 25,
        "last_detection": "Aujourd’hui à 13:37",
    },
]


def create_mock_history() -> pd.DataFrame:
    rows = []

    statuses = [
        "Conforme",
        "Non conforme",
        "Partiellement conforme",
        "Critique",
    ]

    sources = ["Image", "Vidéo", "Caméra"]

    violations = [
        "Aucune",
        "Casque manquant",
        "Gilet manquant",
        "Chaussures non détectées",
        "Gants manquants",
    ]

    for index in range(1, 31):
        zone = ZONES[(index - 1) % len(ZONES)]
        status = statuses[index % len(statuses)]
        violation = violations[index % len(violations)]

        rows.append(
            {
                "ID": f"AN-{2026000 + index}",
                "Date": (
                    datetime.now() - timedelta(days=index % 12)
                ).strftime("%d/%m/%Y"),
                "Heure": f"{8 + index % 10:02d}:{10 + index % 45:02d}",
                "Zone": zone["name"],
                "Source": sources[index % len(sources)],
                "Personnes": 1 + index % 8,
                "EPI présents": 2 + index % 4,
                "EPI manquants": 0 if status == "Conforme" else 1 + index % 3,
                "Conformité": (
                    96
                    if status == "Conforme"
                    else 72 - index % 20
                ),
                "Niveau": (
                    "Faible"
                    if status == "Conforme"
                    else "Élevé"
                ),
                "Violation": violation,
                "Statut": status,
            }
        )

    return pd.DataFrame(rows)


def create_mock_incidents() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Numéro": "INC-2026-017",
                "Type": "Absence de casque",
                "Zone": "Atelier machines",
                "Date": "28/07/2026",
                "Heure": "15:48",
                "Gravité": "Critique",
                "EPI manquant": "Helmet",
                "Responsable": "Superviseur 337",
                "Statut": "Nouveau",
                "Action corrective": "En attente",
            },
            {
                "Numéro": "INC-2026-016",
                "Type": "Absence de gilet",
                "Zone": "La trémie",
                "Date": "28/07/2026",
                "Heure": "14:20",
                "Gravité": "Élevé",
                "EPI manquant": "Safety Vest",
                "Responsable": "Responsable HSE",
                "Statut": "En cours",
                "Action corrective": "Sensibilisation immédiate",
            },
            {
                "Numéro": "INC-2026-015",
                "Type": "Chaussures non détectées",
                "Zone": "Parc matériel",
                "Date": "27/07/2026",
                "Heure": "17:05",
                "Gravité": "Moyen",
                "EPI manquant": "Shoes",
                "Responsable": "Superviseur parc",
                "Statut": "Traité",
                "Action corrective": "Contrôle terrain réalisé",
            },
            {
                "Numéro": "INC-2026-014",
                "Type": "Gants manquants",
                "Zone": "Atelier électrique",
                "Date": "27/07/2026",
                "Heure": "11:32",
                "Gravité": "Élevé",
                "EPI manquant": "Gloves",
                "Responsable": "Superviseur 336",
                "Statut": "Clôturé",
                "Action corrective": "EPI remis à l’opérateur",
            },
        ]
    )


HISTORY_DF = create_mock_history()
INCIDENTS_DF = create_mock_incidents()


# ==========================================================
# ÉTAT DE SESSION
# ==========================================================

if "notifications" not in st.session_state:
    st.session_state.notifications = [
        "Absence de casque détectée dans l’atelier machines.",
        "Caméra CAM-337-02 en maintenance.",
        "Rapport hebdomadaire disponible.",
    ]

if "theme" not in st.session_state:
    st.session_state.theme = "Clair"

if "confidence" not in st.session_state:
    st.session_state.confidence = DEFAULT_CONFIDENCE

if "selected_zone" not in st.session_state:
    st.session_state.selected_zone = ZONES[0]["name"]


# ==========================================================
# CHARGEMENT DU MODÈLE
# ==========================================================

@st.cache_resource(show_spinner=False)
def load_ppe_model(model_path: str):
    if YOLO is None:
        return None

    path = Path(model_path)

    if not path.exists():
        return None

    return YOLO(str(path))


# ==========================================================
# OUTILS DE DÉTECTION
# ==========================================================

def normalize_name(name: Any) -> str:
    return str(name).strip()


def get_detected_classes(result, model) -> list[dict[str, Any]]:
    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        coordinates = list(
            map(
                int,
                box.xyxy[0].tolist(),
            )
        )

        detections.append(
            {
                "class_id": class_id,
                "class_name": normalize_name(
                    model.names[class_id]
                ),
                "confidence": confidence,
                "box": coordinates,
            }
        )

    return detections


def analyse_compliance(
    detections: list[dict[str, Any]],
    required_ppe: list[str] | None = None,
) -> dict[str, Any]:
    if required_ppe is None:
        required_ppe = [
            "Helmet",
            "Safety Vest",
            "Shoes",
        ]

    detected_names = {
        item["class_name"]
        for item in detections
    }

    person_count = sum(
        item["class_name"] == "Person"
        for item in detections
    )

    def contains_any(possible_names: set[str]) -> bool:
        return bool(
            detected_names.intersection(possible_names)
        )

    status_map = {
        "Helmet": contains_any(HELMET_NAMES),
        "Safety Vest": contains_any(VEST_NAMES),
        "Shoes": contains_any(SHOES_NAMES),
        "Gloves": contains_any(GLOVES_NAMES),
        "Glasses": contains_any(GLASSES_NAMES),
        "Mask": contains_any(MASK_NAMES),
        "Earmuffs": contains_any(EAR_NAMES),
        "Face Guard": "Face Guard" in detected_names,
    }

    evaluated = {
        ppe: status_map.get(
            ppe,
            ppe in detected_names,
        )
        for ppe in required_ppe
    }

    present = [
        ppe
        for ppe, value in evaluated.items()
        if value
    ]

    missing = [
        ppe
        for ppe, value in evaluated.items()
        if not value
    ]

    if not evaluated:
        compliance_rate = 0.0
    else:
        compliance_rate = (
            len(present) / len(evaluated)
        ) * 100

    if compliance_rate == 100:
        global_status = "Conforme"
        level = "Faible"
    elif compliance_rate >= 60:
        global_status = "Partiellement conforme"
        level = "Moyen"
    elif compliance_rate > 0:
        global_status = "Non conforme"
        level = "Élevé"
    else:
        global_status = "Critique"
        level = "Critique"

    return {
        "person_count": person_count,
        "present": present,
        "missing": missing,
        "evaluated": evaluated,
        "compliance_rate": compliance_rate,
        "global_status": global_status,
        "risk_level": level,
    }


def run_yolo_inference(
    image: Image.Image,
    confidence: float,
    required_ppe: list[str],
):
    model = load_ppe_model(
        str(PPE_MODEL_PATH)
    )

    if model is None:
        return None, [], None

    image_array = np.array(
        image.convert("RGB")
    )

    result = model.predict(
        source=image_array,
        conf=confidence,
        imgsz=IMAGE_SIZE,
        verbose=False,
    )[0]

    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(
        annotated_bgr,
        cv2.COLOR_BGR2RGB,
    )

    detections = get_detected_classes(
        result,
        model,
    )

    compliance = analyse_compliance(
        detections,
        required_ppe,
    )

    return annotated_rgb, detections, compliance


# ==========================================================
# COMPOSANTS
# ==========================================================

def page_header(
    title: str,
    subtitle: str,
    icon: str = "🦺",
):
    st.markdown(
        f"""
        <div class="main-header">
            <h1>{icon} {title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )


def metric_card(
    title: str,
    value: str,
    icon: str,
    delta: str = "",
):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(status: str) -> str:
    if status == "Conforme":
        css_class = "status-ok"
    elif status in {
        "Partiellement conforme",
        "À vérifier",
        "A VERIFIER",
    }:
        css_class = "status-warning"
    else:
        css_class = "status-danger"

    return (
        f'<span class="{css_class}">'
        f"{status}"
        f"</span>"
    )


def footer():
    st.markdown(
        """
        <div class="footer">
            Plateforme HSE Intelligente · OCP Sidi Chennane ·
            Prototype IA de détection des EPI
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.markdown(
        """
        <div style="
            text-align:center;
            padding:15px 4px 20px 4px;
        ">
            <div style="
                width:64px;
                height:64px;
                border-radius:18px;
                display:flex;
                align-items:center;
                justify-content:center;
                margin:0 auto 10px auto;
                background:rgba(255,255,255,0.16);
                font-size:32px;
            ">
                🛡️
            </div>
            <div style="
                font-size:1.15rem;
                font-weight:850;
            ">
                HSE INTELLIGENTE
            </div>
            <div style="
                font-size:0.78rem;
                opacity:0.80;
            ">
                OCP Sidi Chennane
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="profile-box">
            <b>👤 Responsable HSE</b><br>
            <span style="font-size:0.82rem;">
                📍 Khouribga · Maroc
            </span><br>
            <span style="font-size:0.82rem;">
                🟢 Système opérationnel
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Accueil",
            "🔍 Détection",
            "✅ Conformité",
            "🕘 Historique",
            "🚨 Incidents",
            "📊 Statistiques",
            "🏭 Zones",
            "🎓 Sensibilisation",
            "🤖 Modèle IA",
            "⚙️ Paramètres",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    notification_count = len(
        st.session_state.notifications
    )

    st.markdown(
        f"🔔 **{notification_count} notifications**"
    )

    with st.expander("Voir les notifications"):
        for notification in st.session_state.notifications:
            st.caption(f"• {notification}")

    if st.button(
        "Se déconnecter",
        use_container_width=True,
    ):
        st.info(
            "Déconnexion simulée pour ce prototype."
        )


# ==========================================================
# PAGE ACCUEIL
# ==========================================================

def render_home():
    page_header(
        "Plateforme HSE Intelligente",
        (
            "Prévenir aujourd’hui, protéger pour demain. "
            "Surveillez, analysez et améliorez en continu "
            "la sécurité grâce à l’intelligence artificielle."
        ),
        "🛡️",
    )

    section_title("Vue d’ensemble")

    columns = st.columns(6)

    cards = [
        (
            "Analyses réalisées",
            "1 248",
            "🔎",
            "+8,4 % ce mois",
        ),
        (
            "Personnes détectées",
            "4 380",
            "👥",
            "+146 cette semaine",
        ),
        (
            "Conformité globale",
            "85,6 %",
            "✅",
            "+2,1 points",
        ),
        (
            "Violations",
            "184",
            "⚠️",
            "-12 % ce mois",
        ),
        (
            "Incidents critiques",
            "17",
            "🚨",
            "3 non traités",
        ),
        (
            "Zones surveillées",
            "5",
            "🏭",
            "4 caméras actives",
        ),
    ]

    for column, card in zip(
        columns,
        cards,
    ):
        with column:
            metric_card(*card)

    section_title("Actions rapides")

    quick_columns = st.columns(5)

    quick_actions = [
        ("📷 Importer une image", "image"),
        ("🎥 Analyser une vidéo", "video"),
        ("📡 Ouvrir la caméra", "camera"),
        ("🚨 Voir les incidents", "incident"),
        ("📊 Voir les statistiques", "stats"),
    ]

    for column, action in zip(
        quick_columns,
        quick_actions,
    ):
        with column:
            if st.button(
                action[0],
                use_container_width=True,
                key=f"quick_{action[1]}",
            ):
                st.toast(
                    "Utilise le menu de gauche pour ouvrir cette fonction."
                )

    section_title("Évolution de la sécurité")

    left, right = st.columns(
        [1.55, 1]
    )

    dates = pd.date_range(
        end=datetime.now(),
        periods=14,
        freq="D",
    )

    compliance_df = pd.DataFrame(
        {
            "Date": dates,
            "Conformité": [
                78,
                80,
                79,
                82,
                81,
                83,
                84,
                83,
                86,
                85,
                87,
                86,
                88,
                89,
            ],
            "Analyses": [
                61,
                72,
                68,
                83,
                78,
                89,
                94,
                88,
                102,
                96,
                108,
                104,
                112,
                121,
            ],
        }
    )

    with left:
        fig = px.area(
            compliance_df,
            x="Date",
            y="Conformité",
            title="Taux de conformité sur 14 jours",
            markers=True,
        )

        fig.update_layout(
            height=370,
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=10,
            ),
            yaxis_range=[60, 100],
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    violation_df = pd.DataFrame(
        {
            "EPI": [
                "Casque",
                "Gilet",
                "Chaussures",
                "Gants",
                "Lunettes",
            ],
            "Violations": [
                52,
                41,
                38,
                31,
                22,
            ],
        }
    )

    with right:
        fig = px.bar(
            violation_df,
            x="Violations",
            y="EPI",
            orientation="h",
            title="Violations par EPI",
        )

        fig.update_layout(
            height=370,
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=10,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    section_title("Alertes récentes")

    alerts = [
        (
            "Absence de casque",
            "Atelier machines · il y a 8 minutes",
            "Critique",
        ),
        (
            "Gilet de sécurité non détecté",
            "La trémie · il y a 21 minutes",
            "Élevé",
        ),
        (
            "Chaussures non visibles",
            "Parc matériel · il y a 42 minutes",
            "À vérifier",
        ),
        (
            "Gants manquants",
            "Atelier électrique · il y a 1 heure",
            "Élevé",
        ),
    ]

    for title, detail, level in alerts:
        st.markdown(
            f"""
            <div class="alert-item">
                <b>⚠️ {title}</b>
                <div class="small-text">{detail}</div>
                <div style="margin-top:7px;">
                    {status_badge(level)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==========================================================
# PAGE DÉTECTION
# ==========================================================

def render_detection():
    page_header(
        "Détection intelligente des EPI",
        (
            "Analysez des images, vidéos ou flux caméra "
            "et évaluez automatiquement la conformité HSE."
        ),
        "🔍",
    )

    model = load_ppe_model(
        str(PPE_MODEL_PATH)
    )

    if model is None:
        st.warning(
            "Le modèle YOLO n’est pas chargé. "
            f"Vérifie le fichier : {PPE_MODEL_PATH}"
        )
    else:
        st.success(
            f"Modèle chargé : {PPE_MODEL_PATH.name}"
        )

    settings_col1, settings_col2, settings_col3 = st.columns(3)

    with settings_col1:
        confidence = st.slider(
            "Seuil de confiance",
            min_value=0.10,
            max_value=0.90,
            value=float(
                st.session_state.confidence
            ),
            step=0.05,
        )

        st.session_state.confidence = confidence

    with settings_col2:
        selected_zone = st.selectbox(
            "Zone surveillée",
            [zone["name"] for zone in ZONES],
            index=0,
        )

        st.session_state.selected_zone = selected_zone

    selected_zone_data = next(
        zone
        for zone in ZONES
        if zone["name"] == selected_zone
    )

    with settings_col3:
        automatic_alert = st.toggle(
            "Alertes automatiques",
            value=True,
        )

    required_ppe = st.multiselect(
        "EPI obligatoires pour cette analyse",
        PPE_CLASSES,
        default=selected_zone_data["ppe"],
    )

    image_tab, video_tab, camera_tab = st.tabs(
        [
            "📷 Image",
            "🎥 Vidéo",
            "📡 Caméra",
        ]
    )

    with image_tab:
        st.subheader("Analyse d’une image")

        uploaded_file = st.file_uploader(
            "Importer une image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp",
            ],
            help=(
                "Formats acceptés : JPG, PNG et WEBP."
            ),
        )

        if uploaded_file is not None:
            image = Image.open(
                uploaded_file
            ).convert("RGB")

            preview_col, result_col = st.columns(2)

            with preview_col:
                st.markdown("#### Image originale")

                st.image(
                    image,
                    use_container_width=True,
                )

            if st.button(
                "🚀 Lancer l’analyse",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner(
                    "Analyse de l’image en cours..."
                ):
                    (
                        annotated_image,
                        detections,
                        compliance,
                    ) = run_yolo_inference(
                        image=image,
                        confidence=confidence,
                        required_ppe=required_ppe,
                    )

                if annotated_image is None:
                    st.error(
                        "Impossible de lancer l’analyse. "
                        "Vérifie le modèle et les dépendances."
                    )
                else:
                    with result_col:
                        st.markdown(
                            "#### Résultat annoté"
                        )

                        st.image(
                            annotated_image,
                            use_container_width=True,
                        )

                    result_path = (
                        RESULTS_DIR
                        / (
                            f"analyse_"
                            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            f".jpg"
                        )
                    )

                    cv2.imwrite(
                        str(result_path),
                        cv2.cvtColor(
                            annotated_image,
                            cv2.COLOR_RGB2BGR,
                        ),
                    )

                    st.divider()

                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

                    with kpi1:
                        metric_card(
                            "Personnes",
                            str(
                                compliance["person_count"]
                            ),
                            "👥",
                            "",
                        )

                    with kpi2:
                        metric_card(
                            "Détections",
                            str(len(detections)),
                            "🎯",
                            "",
                        )

                    with kpi3:
                        metric_card(
                            "Conformité",
                            (
                                f"{compliance['compliance_rate']:.1f} %"
                            ),
                            "✅",
                            "",
                        )

                    with kpi4:
                        metric_card(
                            "Risque",
                            compliance["risk_level"],
                            "⚠️",
                            "",
                        )

                    st.markdown(
                        f"""
                        ### Statut final :
                        {status_badge(
                            compliance["global_status"]
                        )}
                        """,
                        unsafe_allow_html=True,
                    )

                    detail_col1, detail_col2 = st.columns(2)

                    with detail_col1:
                        st.success(
                            "EPI présents : "
                            + (
                                ", ".join(
                                    compliance["present"]
                                )
                                if compliance["present"]
                                else "Aucun"
                            )
                        )

                    with detail_col2:
                        st.error(
                            "EPI manquants ou non détectés : "
                            + (
                                ", ".join(
                                    compliance["missing"]
                                )
                                if compliance["missing"]
                                else "Aucun"
                            )
                        )

                    detection_table = pd.DataFrame(
                        [
                            {
                                "Classe": item["class_name"],
                                "Confiance": (
                                    f"{item['confidence']:.2%}"
                                ),
                                "Coordonnées": str(
                                    item["box"]
                                ),
                            }
                            for item in detections
                        ]
                    )

                    st.subheader(
                        "Détail des détections"
                    )

                    st.dataframe(
                        detection_table,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.info(
                        f"Résultat enregistré dans : {result_path}"
                    )

                    if (
                        automatic_alert
                        and compliance["global_status"]
                        != "Conforme"
                    ):
                        st.toast(
                            "Alerte HSE créée automatiquement.",
                            icon="🚨",
                        )

    with video_tab:
        st.subheader("Analyse d’une vidéo")

        video_file = st.file_uploader(
            "Importer une vidéo",
            type=[
                "mp4",
                "avi",
                "mov",
                "mkv",
            ],
            key="video_uploader",
        )

        if video_file is not None:
            st.video(
                video_file
            )

            frame_frequency = st.select_slider(
                "Fréquence d’analyse",
                options=[
                    "Toutes les images",
                    "1 image sur 5",
                    "1 image sur 10",
                    "1 image par seconde",
                ],
                value="1 image sur 10",
            )

            if st.button(
                "🎬 Démarrer l’analyse vidéo",
                use_container_width=True,
            ):
                st.warning(
                    "Le prototype prépare l’intégration. "
                    "Pour une vidéo complète annotée, la prochaine étape "
                    "consistera à connecter un endpoint FastAPI."
                )

                progress = st.progress(0)

                for value in range(
                    0,
                    101,
                    10,
                ):
                    progress.progress(value)

                st.success(
                    "Simulation d’analyse terminée."
                )

                sample_video_incidents = pd.DataFrame(
                    {
                        "Horodatage": [
                            "00:00:04",
                            "00:00:12",
                            "00:00:31",
                        ],
                        "Violation": [
                            "Casque manquant",
                            "Gilet manquant",
                            "Chaussures non détectées",
                        ],
                        "Confiance": [
                            "84 %",
                            "72 %",
                            "61 %",
                        ],
                        "Niveau": [
                            "Critique",
                            "Élevé",
                            "À vérifier",
                        ],
                    }
                )

                st.dataframe(
                    sample_video_incidents,
                    use_container_width=True,
                    hide_index=True,
                )

    with camera_tab:
        st.subheader(
            "Caméra et flux temps réel"
        )

        st.info(
            "Autorise l’accès à la caméra dans ton navigateur."
        )

        camera_mode = st.radio(
            "Mode caméra",
            [
                "Capture simple",
                "Temps réel",
            ],
            horizontal=True,
        )

        if camera_mode == "Capture simple":
            camera_image = st.camera_input(
                "Prendre une photo"
            )

            if camera_image is not None:
                captured_image = Image.open(
                    camera_image
                ).convert("RGB")

                if st.button(
                    "Analyser la capture",
                    type="primary",
                ):
                    (
                        annotated_image,
                        detections,
                        compliance,
                    ) = run_yolo_inference(
                        image=captured_image,
                        confidence=confidence,
                        required_ppe=required_ppe,
                    )

                    if annotated_image is not None:
                        st.image(
                            annotated_image,
                            caption=(
                                f"Résultat : "
                                f"{compliance['global_status']}"
                            ),
                            use_container_width=True,
                        )

                        st.metric(
                            "Conformité",
                            (
                                f"{compliance['compliance_rate']:.1f} %"
                            ),
                        )
                    else:
                        st.error(
                            "Le modèle n’est pas disponible."
                        )

        else:
            if not WEBRTC_AVAILABLE:
                st.error(
                    "Le mode temps réel nécessite "
                    "streamlit-webrtc."
                )

                st.code(
                    "pip install streamlit-webrtc av"
                )
            elif model is None:
                st.error(
                    "Le modèle YOLO est introuvable."
                )
            else:
                class PPEVideoProcessor(
                    VideoProcessorBase
                ):
                    def recv(
                        self,
                        frame,
                    ):
                        image_bgr = frame.to_ndarray(
                            format="bgr24"
                        )

                        result = model.predict(
                            source=image_bgr,
                            conf=confidence,
                            imgsz=IMAGE_SIZE,
                            verbose=False,
                        )[0]

                        annotated = result.plot()

                        return av.VideoFrame.from_ndarray(
                            annotated,
                            format="bgr24",
                        )

                webrtc_streamer(
                    key="ppe-realtime",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTCConfiguration(
                        {
                            "iceServers": [
                                {
                                    "urls": [
                                        "stun:stun.l.google.com:19302"
                                    ]
                                }
                            ]
                        }
                    ),
                    video_processor_factory=PPEVideoProcessor,
                    media_stream_constraints={
                        "video": True,
                        "audio": False,
                    },
                    async_processing=True,
                )


# ==========================================================
# PAGE CONFORMITÉ
# ==========================================================

def render_compliance():
    page_header(
        "Suivi de la conformité",
        (
            "Analysez les résultats HSE par date, zone, source, "
            "type d’EPI et niveau de risque."
        ),
        "✅",
    )

    filter1, filter2, filter3, filter4 = st.columns(4)

    with filter1:
        selected_zone = st.selectbox(
            "Zone",
            [
                "Toutes"
            ]
            + [
                zone["name"]
                for zone in ZONES
            ],
            key="compliance_zone",
        )

    with filter2:
        selected_status = st.selectbox(
            "Statut",
            [
                "Tous",
                "Conforme",
                "Partiellement conforme",
                "Non conforme",
                "Critique",
            ],
        )

    with filter3:
        selected_source = st.selectbox(
            "Source",
            [
                "Toutes",
                "Image",
                "Vidéo",
                "Caméra",
            ],
        )

    with filter4:
        selected_risk = st.selectbox(
            "Niveau de risque",
            [
                "Tous",
                "Faible",
                "Moyen",
                "Élevé",
                "Critique",
            ],
        )

    filtered_df = HISTORY_DF.copy()

    if selected_zone != "Toutes":
        filtered_df = filtered_df[
            filtered_df["Zone"]
            == selected_zone
        ]

    if selected_status != "Tous":
        filtered_df = filtered_df[
            filtered_df["Statut"]
            == selected_status
        ]

    if selected_source != "Toutes":
        filtered_df = filtered_df[
            filtered_df["Source"]
            == selected_source
        ]

    if selected_risk != "Tous":
        filtered_df = filtered_df[
            filtered_df["Niveau"]
            == selected_risk
        ]

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
    )

    action1, action2, action3 = st.columns(3)

    with action1:
        st.download_button(
            "📥 Exporter en CSV",
            data=filtered_df.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="conformite_hse.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with action2:
        if st.button(
            "🚨 Créer un incident",
            use_container_width=True,
        ):
            st.success(
                "Formulaire d’incident ouvert."
            )

    with action3:
        if st.button(
            "🔄 Actualiser",
            use_container_width=True,
        ):
            st.rerun()


# ==========================================================
# PAGE HISTORIQUE
# ==========================================================

def render_history():
    page_header(
        "Historique des analyses",
        (
            "Retrouvez toutes les analyses réalisées "
            "sur les images, vidéos et caméras."
        ),
        "🕘",
    )

    search = st.text_input(
        "Rechercher par ID, zone ou violation",
        placeholder="Exemple : AN-2026005 ou Atelier machines",
    )

    history = HISTORY_DF.copy()

    if search:
        search_lower = search.lower()

        history = history[
            history.astype(str)
            .apply(
                lambda row: row.str.lower()
                .str.contains(
                    search_lower,
                    regex=False,
                )
                .any(),
                axis=1,
            )
        ]

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_id = st.selectbox(
            "Sélectionner une analyse",
            history["ID"].tolist()
            if not history.empty
            else ["Aucune"],
        )

    with col2:
        if st.button(
            "👁️ Voir le détail",
            use_container_width=True,
        ):
            st.info(
                f"Détail demandé pour {selected_id}"
            )

    with col3:
        if st.button(
            "🔁 Rejouer l’analyse",
            use_container_width=True,
        ):
            st.success(
                f"Nouvelle analyse simulée pour {selected_id}"
            )


# ==========================================================
# PAGE INCIDENTS
# ==========================================================

def render_incidents():
    page_header(
        "Gestion des incidents",
        (
            "Suivez les violations, affectez les actions "
            "correctives et contrôlez leur résolution."
        ),
        "🚨",
    )

    kpi_columns = st.columns(4)

    incident_metrics = [
        (
            "Incidents ouverts",
            "12",
            "🚨",
            "3 critiques",
        ),
        (
            "En cours",
            "7",
            "🛠️",
            "4 affectés aujourd’hui",
        ),
        (
            "Traités",
            "31",
            "✅",
            "Cette semaine",
        ),
        (
            "Taux de résolution",
            "88 %",
            "📈",
            "+5 points",
        ),
    ]

    for column, card in zip(
        kpi_columns,
        incident_metrics,
    ):
        with column:
            metric_card(*card)

    section_title("Liste des incidents")

    gravity_filter = st.multiselect(
        "Filtrer par gravité",
        [
            "Faible",
            "Moyen",
            "Élevé",
            "Critique",
        ],
        default=[
            "Faible",
            "Moyen",
            "Élevé",
            "Critique",
        ],
    )

    filtered = INCIDENTS_DF[
        INCIDENTS_DF["Gravité"].isin(
            gravity_filter
        )
    ]

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander(
        "➕ Créer un nouvel incident"
    ):
        incident_type = st.selectbox(
            "Type",
            [
                "Absence de casque",
                "Absence de gilet",
                "Chaussures non détectées",
                "Gants manquants",
                "Accès à une zone à risque",
                "Autre",
            ],
        )

        incident_zone = st.selectbox(
            "Zone",
            [
                zone["name"]
                for zone in ZONES
            ],
            key="incident_zone",
        )

        gravity = st.selectbox(
            "Gravité",
            [
                "Faible",
                "Moyen",
                "Élevé",
                "Critique",
            ],
        )

        responsible = st.text_input(
            "Responsable"
        )

        corrective_action = st.text_area(
            "Action corrective"
        )

        if st.button(
            "Créer l’incident",
            type="primary",
        ):
            st.success(
                "Incident créé avec succès."
            )


# ==========================================================
# PAGE STATISTIQUES
# ==========================================================

def render_statistics():
    page_header(
        "Statistiques et performance HSE",
        (
            "Visualisez les tendances de conformité, "
            "les violations et les zones les plus exposées."
        ),
        "📊",
    )

    period = st.segmented_control(
        "Période",
        [
            "Aujourd’hui",
            "Cette semaine",
            "Ce mois",
            "Personnalisée",
        ],
        default="Cette semaine",
    )

    kpi_columns = st.columns(4)

    stats_cards = [
        (
            "Conformité globale",
            "85,6 %",
            "✅",
            "+2,1 points",
        ),
        (
            "Violations totales",
            "184",
            "⚠️",
            "-12 %",
        ),
        (
            "Incidents critiques",
            "17",
            "🚨",
            "3 ouverts",
        ),
        (
            "Taux de résolution",
            "88 %",
            "🛠️",
            "+5 points",
        ),
    ]

    for column, card in zip(
        kpi_columns,
        stats_cards,
    ):
        with column:
            metric_card(*card)

    chart1, chart2 = st.columns(2)

    zone_stats = pd.DataFrame(
        {
            "Zone": [
                zone["name"]
                for zone in ZONES
            ],
            "Conformité": [
                zone["compliance"]
                for zone in ZONES
            ],
            "Violations": [
                zone["violations"]
                for zone in ZONES
            ],
        }
    )

    with chart1:
        fig = px.bar(
            zone_stats,
            x="Zone",
            y="Conformité",
            title="Conformité par zone",
            text_auto=".1f",
        )

        fig.update_layout(
            height=420,
            xaxis_tickangle=-20,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with chart2:
        fig = px.pie(
            zone_stats,
            names="Zone",
            values="Violations",
            title="Répartition des violations",
            hole=0.48,
        )

        fig.update_layout(
            height=420
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    chart3, chart4 = st.columns(2)

    daily_df = pd.DataFrame(
        {
            "Jour": [
                "Lun",
                "Mar",
                "Mer",
                "Jeu",
                "Ven",
                "Sam",
                "Dim",
            ],
            "Analyses": [
                152,
                173,
                166,
                188,
                204,
                91,
                72,
            ],
            "Conformité": [
                82,
                84,
                83,
                86,
                88,
                87,
                89,
            ],
        }
    )

    with chart3:
        fig = px.line(
            daily_df,
            x="Jour",
            y="Conformité",
            markers=True,
            title="Tendance hebdomadaire",
        )

        fig.update_layout(
            height=390
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with chart4:
        fig = px.bar(
            daily_df,
            x="Jour",
            y="Analyses",
            title="Nombre d’analyses par jour",
        )

        fig.update_layout(
            height=390
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    export1, export2, export3 = st.columns(3)

    with export1:
        st.download_button(
            "Exporter CSV",
            data=zone_stats.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="statistiques_zones.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with export2:
        st.button(
            "Exporter Excel",
            use_container_width=True,
        )

    with export3:
        st.button(
            "Exporter PDF",
            use_container_width=True,
        )


# ==========================================================
# PAGE ZONES
# ==========================================================

def render_zones():
    page_header(
        "Zones surveillées",
        (
            "Supervisez les cinq zones principales "
            "du site Sidi Chennane à Khouribga."
        ),
        "🏭",
    )

    selected_zone_name = st.selectbox(
        "Accéder directement à une zone",
        [
            zone["name"]
            for zone in ZONES
        ],
    )

    for zone in ZONES:
        risk_class = (
            "risk-high"
            if zone["risk"] == "Élevé"
            else "risk-medium"
        )

        camera_icon = (
            "🟢"
            if zone["camera_status"] == "Active"
            else "🟠"
        )

        st.markdown(
            f"""
            <div class="zone-card">
                <div style="
                    display:flex;
                    justify-content:space-between;
                    gap:15px;
                    align-items:flex-start;
                    flex-wrap:wrap;
                ">
                    <div>
                        <h3 style="
                            margin:0;
                            color:#123f32;
                        ">
                            🏭 {zone["name"]}
                        </h3>
                        <div class="small-text">
                            Service : {zone["service"]}
                        </div>
                    </div>

                    <div class="{risk_class}">
                        Risque {zone["risk"]}
                    </div>
                </div>

                <p>{zone["description"]}</p>

                <div>
                    <b>Équipements :</b>
                    {zone["equipment"]}
                </div>

                <div style="margin-top:8px;">
                    <b>EPI obligatoires :</b>
                    {", ".join(zone["ppe"])}
                </div>

                <div style="margin-top:8px;">
                    <b>Caméra :</b>
                    {camera_icon}
                    {zone["camera"]}
                    · {zone["camera_status"]}
                </div>

                <div style="margin-top:8px;">
                    <b>Analyses :</b>
                    {zone["analyses"]}
                    ·
                    <b>Conformité :</b>
                    {zone["compliance"]} %
                    ·
                    <b>Violations :</b>
                    {zone["violations"]}
                </div>

                <div style="margin-top:8px;">
                    <b>Dernière détection :</b>
                    {zone["last_detection"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        action1, action2, action3, action4 = st.columns(4)

        with action1:
            st.button(
                "Voir le détail",
                key=f"detail_{zone['id']}",
                use_container_width=True,
            )

        with action2:
            st.button(
                "Lancer analyse",
                key=f"analysis_{zone['id']}",
                use_container_width=True,
            )

        with action3:
            st.button(
                "Voir incidents",
                key=f"incidents_{zone['id']}",
                use_container_width=True,
            )

        with action4:
            st.button(
                "Modifier règles",
                key=f"rules_{zone['id']}",
                use_container_width=True,
            )

        st.divider()

    zone = next(
        zone
        for zone in ZONES
        if zone["name"] == selected_zone_name
    )

    section_title(
        f"Détail analytique : {zone['name']}"
    )

    detail1, detail2, detail3 = st.columns(3)

    with detail1:
        st.metric(
            "Conformité",
            f"{zone['compliance']} %",
        )

    with detail2:
        st.metric(
            "Analyses",
            zone["analyses"],
        )

    with detail3:
        st.metric(
            "Violations",
            zone["violations"],
        )


# ==========================================================
# PAGE SENSIBILISATION
# ==========================================================

def render_awareness():
    page_header(
        "Sensibilisation HSE",
        (
            "Guides, affiches, procédures, formations "
            "et bonnes pratiques pour renforcer la culture sécurité."
        ),
        "🎓",
    )

    search = st.text_input(
        "Rechercher une ressource",
        placeholder="Casque, gants, risques mécaniques...",
    )

    category = st.selectbox(
        "Catégorie",
        [
            "Toutes",
            "Guide",
            "Affiche",
            "Vidéo",
            "Procédure",
            "Formation",
        ],
    )

    resources = [
        {
            "title": "Pourquoi porter un casque ?",
            "category": "Guide",
            "description": (
                "Comprendre les risques liés aux chocs "
                "et aux chutes d’objets."
            ),
            "icon": "⛑️",
        },
        {
            "title": "Comment choisir ses gants ?",
            "category": "Guide",
            "description": (
                "Choisir des gants adaptés au risque mécanique, "
                "chimique ou électrique."
            ),
            "icon": "🧤",
        },
        {
            "title": "Contrôle du gilet de sécurité",
            "category": "Procédure",
            "description": (
                "Vérification de la visibilité, des bandes "
                "réfléchissantes et de l’état général."
            ),
            "icon": "🦺",
        },
        {
            "title": "Chaussures de sécurité",
            "category": "Formation",
            "description": (
                "Rôle des semelles, embouts et protections "
                "contre l’écrasement."
            ),
            "icon": "🥾",
        },
        {
            "title": "Risques électriques",
            "category": "Affiche",
            "description": (
                "Règles essentielles avant toute intervention "
                "sur une installation électrique."
            ),
            "icon": "⚡",
        },
        {
            "title": "Circulation des engins",
            "category": "Vidéo",
            "description": (
                "Bonnes pratiques dans les zones "
                "de circulation des engins lourds."
            ),
            "icon": "🚛",
        },
    ]

    filtered_resources = []

    for resource in resources:
        text_match = (
            not search
            or search.lower()
            in (
                resource["title"]
                + " "
                + resource["description"]
            ).lower()
        )

        category_match = (
            category == "Toutes"
            or resource["category"] == category
        )

        if text_match and category_match:
            filtered_resources.append(resource)

    columns = st.columns(3)

    for index, resource in enumerate(
        filtered_resources
    ):
        with columns[index % 3]:
            st.markdown(
                f"""
                <div class="info-card">
                    <div style="font-size:2rem;">
                        {resource["icon"]}
                    </div>
                    <h4>{resource["title"]}</h4>
                    <div class="small-text">
                        {resource["category"]}
                    </div>
                    <p>{resource["description"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            button_col1, button_col2 = st.columns(2)

            with button_col1:
                st.button(
                    "Consulter",
                    key=f"view_resource_{index}",
                    use_container_width=True,
                )

            with button_col2:
                st.button(
                    "☆ Favori",
                    key=f"favorite_resource_{index}",
                    use_container_width=True,
                )


# ==========================================================
# PAGE MODÈLE IA
# ==========================================================

def render_model():
    page_header(
        "Modèle d’intelligence artificielle",
        (
            "Consultez les performances du modèle YOLO "
            "et son état de déploiement."
        ),
        "🤖",
    )

    model_exists = PPE_MODEL_PATH.exists()

    if model_exists:
        api_status = "Opérationnel"
        model_status = "Disponible"
    else:
        api_status = "Non connecté"
        model_status = "Introuvable"

    columns = st.columns(4)

    model_cards = [
        (
            "Modèle",
            "YOLOv8s Fusion",
            "🤖",
            model_status,
        ),
        (
            "Nombre de classes",
            "21",
            "🏷️",
            "Dataset PPE + SH17",
        ),
        (
            "mAP50",
            "0,563",
            "🎯",
            "Après 10 epochs",
        ),
        (
            "Statut API",
            api_status,
            "🔌",
            "FastAPI à connecter",
        ),
    ]

    for column, card in zip(
        columns,
        model_cards,
    ):
        with column:
            metric_card(*card)

    metrics_df = pd.DataFrame(
        {
            "Métrique": [
                "Precision",
                "Recall",
                "mAP50",
                "mAP50-95",
            ],
            "Valeur": [
                0.778,
                0.538,
                0.563,
                0.334,
            ],
        }
    )

    class_metrics_df = pd.DataFrame(
        {
            "Classe": [
                "Person",
                "Head",
                "Face",
                "Helmet",
                "Safety Vest",
                "Shoes",
                "Ladder",
            ],
            "mAP50": [
                0.877,
                0.878,
                0.880,
                0.874,
                0.623,
                0.461,
                0.933,
            ],
            "mAP50-95": [
                0.614,
                0.617,
                0.594,
                0.483,
                0.402,
                0.238,
                0.610,
            ],
        }
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig = px.bar(
            metrics_df,
            x="Métrique",
            y="Valeur",
            title="Métriques globales",
            text_auto=".3f",
        )

        fig.update_layout(
            height=410,
            yaxis_range=[0, 1],
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with chart_col2:
        fig = px.bar(
            class_metrics_df,
            x="Classe",
            y=[
                "mAP50",
                "mAP50-95",
            ],
            barmode="group",
            title="Performances par classe",
        )

        fig.update_layout(
            height=410,
            yaxis_range=[0, 1],
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.subheader("Classes du modèle")

    class_df = pd.DataFrame(
        {
            "ID": list(
                range(len(PPE_CLASSES))
            ),
            "Classe": PPE_CLASSES,
            "Activée": [
                True
                for _ in PPE_CLASSES
            ],
        }
    )

    st.dataframe(
        class_df,
        use_container_width=True,
        hide_index=True,
    )

    action1, action2, action3 = st.columns(3)

    with action1:
        if st.button(
            "Tester le modèle",
            use_container_width=True,
        ):
            st.info(
                "Ouvre la page Détection pour tester une image."
            )

    with action2:
        if model_exists:
            with open(
                PPE_MODEL_PATH,
                "rb",
            ) as model_file:
                st.download_button(
                    "Télécharger le modèle",
                    data=model_file,
                    file_name=PPE_MODEL_PATH.name,
                    mime="application/octet-stream",
                    use_container_width=True,
                )
        else:
            st.button(
                "Modèle indisponible",
                disabled=True,
                use_container_width=True,
            )

    with action3:
        st.button(
            "Changer de modèle",
            use_container_width=True,
        )


# ==========================================================
# PAGE PARAMÈTRES
# ==========================================================

def render_settings():
    page_header(
        "Paramètres de la plateforme",
        (
            "Configurez les préférences, les alertes, "
            "les classes et les règles de conformité."
        ),
        "⚙️",
    )

    general_tab, detection_tab, users_tab = st.tabs(
        [
            "Général",
            "Détection",
            "Utilisateurs",
        ]
    )

    with general_tab:
        language = st.selectbox(
            "Langue",
            [
                "Français",
                "English",
                "العربية",
            ],
        )

        theme = st.selectbox(
            "Thème",
            [
                "Clair",
                "Sombre",
                "Système",
            ],
        )

        api_address = st.text_input(
            "Adresse API FastAPI",
            value="http://127.0.0.1:8000",
        )

        default_site = st.text_input(
            "Site par défaut",
            value="OCP Sidi Chennane",
        )

        default_zone = st.selectbox(
            "Zone par défaut",
            [
                zone["name"]
                for zone in ZONES
            ],
        )

        notifications_enabled = st.toggle(
            "Activer les notifications",
            value=True,
        )

        if st.button(
            "Enregistrer les paramètres",
            type="primary",
        ):
            st.session_state.theme = theme
            st.session_state.selected_zone = default_zone

            st.success(
                "Paramètres enregistrés."
            )

    with detection_tab:
        new_confidence = st.slider(
            "Seuil de confiance",
            min_value=0.10,
            max_value=0.90,
            value=float(
                st.session_state.confidence
            ),
            step=0.05,
            key="settings_confidence",
        )

        enabled_classes = st.multiselect(
            "Classes activées",
            PPE_CLASSES,
            default=PPE_CLASSES,
        )

        frequency = st.selectbox(
            "Fréquence d’analyse",
            [
                "Temps réel",
                "1 image sur 5",
                "1 image sur 10",
                "1 image par seconde",
            ],
        )

        analysis_mode = st.radio(
            "Mode d’analyse",
            [
                "Automatique",
                "Manuel",
            ],
            horizontal=True,
        )

        st.text_area(
            "Règles de conformité",
            value=(
                "Une personne est conforme lorsque "
                "tous les EPI obligatoires de la zone "
                "sont détectés et correctement portés."
            ),
        )

        if st.button(
            "Enregistrer la configuration IA",
            type="primary",
        ):
            st.session_state.confidence = new_confidence

            st.success(
                "Configuration de détection enregistrée."
            )

    with users_tab:
        roles_df = pd.DataFrame(
            {
                "Rôle": [
                    "Administrateur",
                    "Responsable HSE",
                    "Superviseur",
                    "Utilisateur simple",
                ],
                "Utilisateurs": [
                    2,
                    4,
                    8,
                    19,
                ],
                "Accès principal": [
                    "Configuration complète",
                    "Analyses et incidents",
                    "Gestion de zone",
                    "Analyse et sensibilisation",
                ],
            }
        )

        st.dataframe(
            roles_df,
            use_container_width=True,
            hide_index=True,
        )

        with st.expander(
            "Ajouter un utilisateur"
        ):
            st.text_input(
                "Nom complet"
            )

            st.text_input(
                "Adresse e-mail"
            )

            st.selectbox(
                "Rôle",
                roles_df["Rôle"].tolist(),
            )

            st.selectbox(
                "Zone",
                [
                    "Toutes"
                ]
                + [
                    zone["name"]
                    for zone in ZONES
                ],
            )

            if st.button(
                "Créer l’utilisateur"
            ):
                st.success(
                    "Utilisateur créé."
                )


# ==========================================================
# ROUTAGE
# ==========================================================

if page == "🏠 Accueil":
    render_home()

elif page == "🔍 Détection":
    render_detection()

elif page == "✅ Conformité":
    render_compliance()

elif page == "🕘 Historique":
    render_history()

elif page == "🚨 Incidents":
    render_incidents()

elif page == "📊 Statistiques":
    render_statistics()

elif page == "🏭 Zones":
    render_zones()

elif page == "🎓 Sensibilisation":
    render_awareness()

elif page == "🤖 Modèle IA":
    render_model()

elif page == "⚙️ Paramètres":
    render_settings()

footer()