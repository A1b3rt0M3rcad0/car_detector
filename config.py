"""
Configurações do sistema - Detecção de Veículos
"""
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# ============================================
# RTSP Camera
# ============================================
RTSP_URL = os.getenv("RTSP_URL", "rtsp://arcnet:12345678@d4440c81205e.sn.mynetname.net:556/stream1")

# ============================================
# YOLO Model Settings
# ============================================
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")

# ============================================
# Car Detection Settings
# ============================================
# IDs de classes do YOLO para veículos (2=car, 3=motorcycle, 5=bus, 7=truck)
vehicle_ids_str = os.getenv("VEHICLE_CLASS_IDS", "2,3,5,7")
VEHICLE_CLASS_IDS = [int(x.strip()) for x in vehicle_ids_str.split(",")]

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

# ============================================
# Display Settings
# ============================================
SHOW_DETECTIONS = os.getenv("SHOW_DETECTIONS", "true").lower() == "true"
WINDOW_NAME = os.getenv("WINDOW_NAME", "Detecção de Veículos")

