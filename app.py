"""
Sistema de detecção de veículos usando YOLO e stream RTSP
"""
import cv2
import numpy as np
from ultralytics import YOLO
from config import RTSP_URL, VEHICLE_CLASS_IDS, CONFIDENCE_THRESHOLD, SHOW_DETECTIONS, WINDOW_NAME, YOLO_MODEL


class VehicleDetector:
    """Detector de veículos usando YOLO"""
    
    def __init__(self):
        """Inicializa o modelo YOLO"""
        print(f"Carregando modelo YOLO: {YOLO_MODEL}...")
        # Usa o modelo YOLOv8 pré-treinado configurado no .env
        self.model = YOLO(YOLO_MODEL)
        print("Modelo YOLO carregado com sucesso!")
        
    def detect_vehicles(self, frame):
        """
        Detecta veículos no frame
        
        Args:
            frame: Frame do vídeo (numpy array)
            
        Returns:
            Lista de detecções: [(x1, y1, x2, y2, confidence, class_id), ...]
        """
        # Executa detecção
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        vehicles = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Filtra apenas veículos (carros, motos, ônibus, caminhões)
                class_id = int(box.cls[0])
                if class_id in VEHICLE_CLASS_IDS:
                    # Pega coordenadas da bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    vehicles.append((int(x1), int(y1), int(x2), int(y2), confidence, class_id))
        
        return vehicles
    
    def draw_detections(self, frame, vehicles):
        """
        Desenha as detecções no frame
        
        Args:
            frame: Frame do vídeo
            vehicles: Lista de detecções de veículos
            
        Returns:
            Frame com detecções desenhadas
        """
        frame_copy = frame.copy()
        
        for x1, y1, x2, y2, confidence, class_id in vehicles:
            # Desenha retângulo
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Desenha label com confiança
            vehicle_type = {2: "Carro", 3: "Moto", 5: "Ônibus", 7: "Caminhão"}.get(class_id, "Veículo")
            label = f"{vehicle_type}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame_copy, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Mostra contador de veículos
        count_text = f"Veículos detectados: {len(vehicles)}"
        cv2.putText(frame_copy, count_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame_copy


class RTSPStream:
    """Gerenciador de stream RTSP"""
    
    def __init__(self, rtsp_url):
        """
        Inicializa conexão com stream RTSP
        
        Args:
            rtsp_url: URL do stream RTSP
        """
        self.rtsp_url = rtsp_url
        self.cap = None
        
    def connect(self):
        """Conecta ao stream RTSP"""
        print(f"Conectando ao stream RTSP: {self.rtsp_url}")
        self.cap = cv2.VideoCapture(self.rtsp_url)
        
        if not self.cap.isOpened():
            raise ConnectionError(f"Erro ao conectar ao stream RTSP: {self.rtsp_url}")
        
        print("Conectado com sucesso!")
        
    def read_frame(self):
        """
        Lê um frame do stream
        
        Returns:
            tuple: (success, frame) ou (False, None) se não conseguir ler
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def release(self):
        """Libera recursos do stream"""
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    """Função principal"""
    detector = VehicleDetector()
    stream = RTSPStream(RTSP_URL)
    
    try:
        # Conecta ao stream
        stream.connect()
        
        print("Iniciando detecção de veículos. Pressione 'q' para sair.")
        
        frame_count = 0
        
        while True:
            # Lê frame
            ret, frame = stream.read_frame()
            
            if not ret:
                print("Erro ao ler frame. Tentando reconectar...")
                stream.release()
                stream.connect()
                continue
            
            frame_count += 1
            
            # Detecta veículos
            vehicles = detector.detect_vehicles(frame)
            
            # Mostra informações no console a cada 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: {len(vehicles)} veículo(s) detectado(s)")
            
            # Desenha detecções no frame
            if SHOW_DETECTIONS:
                frame_with_detections = detector.draw_detections(frame, vehicles)
                cv2.imshow(WINDOW_NAME, frame_with_detections)
                
                # Sai se pressionar 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Se não mostrar, apenas processa
                pass
                
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário")
    except Exception as e:
        print(f"Erro: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stream.release()
        print("Stream liberado. Encerrando...")


if __name__ == "__main__":
    main()