"""
Sistema de detecção de veículos usando YOLO e stream RTSP
"""
import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk
import os
from datetime import datetime
from ultralytics import YOLO
from config import (
    RTSP_URL,
    VEHICLE_CLASS_IDS,
    CONFIDENCE_THRESHOLD,
    SHOW_DETECTIONS,
    WINDOW_NAME,
    YOLO_MODEL,
    MIN_BBOX_AREA_RATIO,
    MIN_BOTTOM_Y_RATIO,
    CENTER_X_MIN,
    CENTER_X_MAX,
    PLATE_MIN_AREA_RATIO,
    PLATE_MAX_AREA_RATIO,
)


class VehicleDetector:
    """Detector de veículos usando YOLO"""
    
    def __init__(self, stationary_threshold=5, stationary_frames=10):
        """
        Inicializa o modelo YOLO
        
        Args:
            stationary_threshold: Threshold de movimento para considerar parado (pixels)
            stationary_frames: Número de frames para confirmar que está parado
        """
        print(f"Carregando modelo YOLO: {YOLO_MODEL}...")
        # Usa o modelo YOLOv8 pré-treinado configurado no .env
        self.model = YOLO(YOLO_MODEL)
        print("Modelo YOLO carregado com sucesso!")
        
        # Para rastreamento de movimento
        self.stationary_threshold = stationary_threshold
        self.stationary_frames = stationary_frames
        self.vehicle_history = {}  # {id: {'centers': [...], 'frames_stopped': 0}}
        self.next_id = 0
        
        # Para salvar screenshots
        self.saved_vehicles = set()  # IDs dos veículos já salvos
        self.save_dir = "veiculos"
        os.makedirs(self.save_dir, exist_ok=True)
        self.last_saved_position = None  # Última posição (center_x, center_y) salva
        self.min_distance = 100  # Distância mínima em pixels para salvar novo veículo
        
        # Para carros em movimento
        self.moving_cars = {}  # {vehicle_id: {'frame': frame_copy, 'position': (x, y), 'last_seen': frame_count}}
        self.disappeared_cars = {}  # Carros que saíram da tela
        self.max_disappeared = 30  # Máximo de frames sem ver antes de salvar
        
        # Modelo YOLO para placas (se disponível)
        try:
            self.plate_model = YOLO("yolov8n.pt")  # Tentativa de usar YOLO para placas
        except:
            self.plate_model = None
        
    def is_front_view(self, x1, y1, x2, y2):
        """
        Verifica se o veículo está de frente (baseado na proporção e posição)
        
        Args:
            x1, y1, x2, y2: Coordenadas da bounding box
            
        Returns:
            bool: True se parece ser visão frontal
        """
        width = x2 - x1
        height = y2 - y1
        
        if height == 0:
            return False
            
        aspect_ratio = width / height
        
        # Veículos de frente geralmente têm aspect ratio entre 0.8 e 1.5
        # e são grandes o suficiente para a placa ser legível
        is_good_ratio = 0.8 <= aspect_ratio <= 1.5
        
        # Aumenta o tamanho mínimo para garantir placa legível
        # width > 150px e height > 120px para garantir boa resolução da placa
        is_large_enough = width > 150 and height > 120
        
        return is_good_ratio and is_large_enough
    
    def update_vehicle_tracking(self, vehicles):
        """
        Atualiza o rastreamento de veículos para detectar movimento
        
        Args:
            vehicles: Lista de veículos detectados
            
        Returns:
            dict: Informações de rastreamento atualizadas
        """
        if not vehicles:
            # Limpa histórico antigo
            to_remove = [vid for vid, data in self.vehicle_history.items() 
                        if len(data['centers']) > 0]
            for vid in to_remove:
                del self.vehicle_history[vid]
            return {}
        
        current_centers = {}
        tracking_info = {}
        
        # Calcula centros dos veículos atuais
        for x1, y1, x2, y2, conf, class_id, is_front in vehicles:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center = (center_x, center_y)
            current_centers[id((x1, y1, x2, y2))] = center
        
        # Encontra correspondências com histórico
        for vid, data in list(self.vehicle_history.items()):
            if not data['centers']:
                continue
                
            last_center = data['centers'][-1]
            min_dist = float('inf')
            best_match = None
            
            for curr_id, curr_center in current_centers.items():
                dist = np.sqrt((last_center[0] - curr_center[0])**2 + 
                              (last_center[1] - curr_center[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_match = curr_id
            
            if best_match and min_dist < 50:  # Threshold de 50 pixels
                # Veículo encontrado - verifica se se moveu
                if min_dist < self.stationary_threshold:
                    data['frames_stopped'] += 1
                else:
                    data['frames_stopped'] = 0
                
                data['centers'].append(current_centers[best_match])
                # Mantém apenas últimos 5 centros
                if len(data['centers']) > 5:
                    data['centers'] = data['centers'][-5:]
                
                del current_centers[best_match]
                
                tracking_info[best_match] = {
                    'frames_stopped': data['frames_stopped'],
                    'is_stationary': data['frames_stopped'] >= self.stationary_frames
                }
        
        # Cria novos IDs para veículos não rastreados
        for curr_id in current_centers:
            vid = self.next_id
            self.next_id += 1
            self.vehicle_history[vid] = {
                'centers': [current_centers[curr_id]],
                'frames_stopped': 0
            }
            tracking_info[curr_id] = {
                'frames_stopped': 0,
                'is_stationary': False
            }
        
        return tracking_info
        
    def detect_vehicles(self, frame):
        """
        Detecta veículos no frame
        
        Args:
            frame: Frame do vídeo (numpy array)
            
        Returns:
            Lista de detecções: [(x1, y1, x2, y2, confidence, class_id, is_front, is_stationary), ...]
        """
        # Executa detecção
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        h, w = frame.shape[:2]
        frame_area = float(h * w)
        
        vehicles = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Filtra apenas classes permitidas
                class_id = int(box.cls[0])
                if class_id not in VEHICLE_CLASS_IDS:
                    continue
                
                # Coordenadas
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                width = max(1.0, x2 - x1)
                height = max(1.0, y2 - y1)
                area = width * height
                bottom = y2
                center_x = (x1 + x2) / 2.0
                
                # Filtros anti falsos positivos:
                # - tamanho mínimo relativo ao frame
                if area / frame_area < MIN_BBOX_AREA_RATIO:
                    continue
                # - deve estar mais para baixo no frame (carro próximo)
                if bottom < h * MIN_BOTTOM_Y_RATIO:
                    continue
                # - manter na região central horizontal
                if not (w * CENTER_X_MIN <= center_x <= w * CENTER_X_MAX):
                    continue
                
                # Verifica se é visão frontal
                is_front = self.is_front_view(x1, y1, x2, y2)
                
                vehicles.append((int(x1), int(y1), int(x2), int(y2), confidence, class_id, is_front))
        
        # Atualiza rastreamento
        tracking_info = self.update_vehicle_tracking(vehicles)
        
        # Adiciona informação de movimento
        vehicles_with_movement = []
        for x1, y1, x2, y2, conf, class_id, is_front in vehicles:
            vehicle_id = id((x1, y1, x2, y2))
            is_stationary = tracking_info.get(vehicle_id, {}).get('is_stationary', False)
            frames_stopped = tracking_info.get(vehicle_id, {}).get('frames_stopped', 0)
            
            vehicles_with_movement.append((x1, y1, x2, y2, conf, class_id, is_front, is_stationary, frames_stopped))
        
        return vehicles_with_movement
    
    def detect_plate_in_vehicle(self, vehicle_roi):
        """
        Detecta se há uma placa visível na região do veículo
        
        Args:
            vehicle_roi: Região de interesse (ROI) do veículo
            
        Returns:
            bool: True se detectou placa, False caso contrário
        """
        if vehicle_roi is None or vehicle_roi.size == 0:
            return False
        
        # Converte para escala de cinza
        gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
        
        # Aplica threshold para destacar regiões escuras (placas)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Aplica operações morfológicas para melhorar detecção
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        
        # Detecta contornos
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtra contornos que podem ser placas
        h, w = vehicle_roi.shape[:2]
        min_area = (w * h) * PLATE_MIN_AREA_RATIO
        max_area = (w * h) * PLATE_MAX_AREA_RATIO
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Pega o retângulo que engloba o contorno
                x, y, w_c, h_c = cv2.boundingRect(contour)
                
                # Verifica se tem proporção de placa (tipo retangular/horizontal)
                aspect_ratio = w_c / h_c if h_c > 0 else 0
                
                # Placas geralmente têm aspect ratio entre 1.5 e 4.0
                if 1.5 <= aspect_ratio <= 4.0:
                    return True
        
        return False
    
    def capture_moving_car(self, frame, x1, y1, x2, y2, confidence, class_id, vehicle_id, frame_count):
        """
        Captura um carro em movimento de frente e salva quando ele parar ou sair da tela
        
        Args:
            frame: Frame completo do vídeo
            x1, y1, x2, y2: Coordenadas da bounding box
            confidence: Confiança da detecção
            class_id: ID da classe do veículo
            vehicle_id: ID único do veículo
            frame_count: Número do frame atual
        """
        # Filtra apenas CARROS (class_id = 2), ignora caminhões, ônibus e motos
        if class_id != 2:
            return
        
        # Calcula o centro do veículo
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        current_position = (center_x, center_y)
        
        # Se já está em moving_cars, atualiza a última vez que foi visto
        if vehicle_id in self.moving_cars:
            self.moving_cars[vehicle_id]['last_seen'] = frame_count
            return
        
        # Se já salvou esse veículo (parou recentemente), não captura de novo
        if vehicle_id in self.saved_vehicles:
            return
        
        # Novo carro em movimento - captura o frame
        self.moving_cars[vehicle_id] = {
            'frame': frame.copy(),
            'position': current_position,
            'last_seen': frame_count,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'confidence': confidence,
            'class_id': class_id
        }
    
    def check_disappeared_cars(self, frame_count, vehicles):
        """
        Verifica carros que desapareceram da tela e salva se necessário
        
        Args:
            frame_count: Número do frame atual
            vehicles: Lista de veículos detectados no frame atual
        """
        # IDs dos veículos atualmente visíveis
        visible_ids = {id((x1, y1, x2, y2, frames_stopped)) 
                      for x1, y1, x2, y2, _, _, _, _, frames_stopped in vehicles}
        
        # Encontra carros que desapareceram
        to_remove = []
        for vehicle_id, data in self.moving_cars.items():
            if vehicle_id not in visible_ids:
                # Carro desapareceu
                frames_disappeared = frame_count - data['last_seen']
                if frames_disappeared > self.max_disappeared:
                    # Salva porque passou muito tempo sem ver
                    self._save_captured_car(vehicle_id)
                    to_remove.append(vehicle_id)
        
        # Remove carros já salvos
        for vehicle_id in to_remove:
            del self.moving_cars[vehicle_id]
    
    def _save_captured_car(self, vehicle_id):
        """Salva o carro que foi capturado em movimento"""
        if vehicle_id not in self.moving_cars:
            return
        
        data = self.moving_cars[vehicle_id]
        frame_copy = data['frame']
        x1, y1, x2, y2 = data['x1'], data['y1'], data['x2'], data['y2']
        confidence = data['confidence']
        class_id = data['class_id']
        
        # Adiciona buffer
        buffer = 20
        h, w = frame_copy.shape[:2]
        x1_buf = max(0, x1 - buffer)
        y1_buf = max(0, y1 - buffer)
        x2_buf = min(w, x2 + buffer)
        y2_buf = min(h, y2 + buffer)
        
        vehicle_roi = frame_copy[y1_buf:y2_buf, x1_buf:x2_buf]
        
        # Verifica placa
        has_plate = self.detect_plate_in_vehicle(vehicle_roi)
        if not has_plate:
            print("⚠️ Carro em movimento: placa não detectada, screenshot não salvo")
            return
        
        # Salva
        vehicle_type = {2: "Carro", 3: "Moto", 5: "Onibus", 7: "Caminhao"}.get(class_id, "Veiculo")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{vehicle_type}_{confidence:.2f}_{timestamp}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        
        cv2.imwrite(filepath, vehicle_roi)
        self.saved_vehicles.add(vehicle_id)
        self.last_saved_position = data['position']
        
        print(f"✅ Screenshot salvo (carro em movimento desapareceu): {filepath}")
    
    def save_vehicle_image(self, frame, x1, y1, x2, y2, confidence, class_id, vehicle_id):
        """
        Salva a imagem de um veículo parado de frente
        
        Args:
            frame: Frame original do vídeo
            x1, y1, x2, y2: Coordenadas da bounding box
            confidence: Confiança da detecção
            class_id: ID da classe do veículo
            vehicle_id: ID único do veículo (para evitar salvar múltiplas vezes)
        """
        # Filtra apenas CARROS (class_id = 2), ignora caminhões, ônibus e motos
        if class_id != 2:
            return None
        
        # Calcula o centro do veículo
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        current_position = (center_x, center_y)
        
        # Verifica se já salvou esse veículo recentemente (posição muito próxima)
        if self.last_saved_position is not None:
            distance = np.sqrt((center_x - self.last_saved_position[0])**2 + 
                             (center_y - self.last_saved_position[1])**2)
            if distance < self.min_distance:
                return  # Muito próximo da última captura, ignora
        
        # Adiciona um buffer ao redor do veículo para incluir mais contexto
        buffer = 20
        h, w = frame.shape[:2]
        
        # Ajusta coordenadas com buffer (limitando aos limites da imagem)
        x1_buf = max(0, x1 - buffer)
        y1_buf = max(0, y1 - buffer)
        x2_buf = min(w, x2 + buffer)
        y2_buf = min(h, y2 + buffer)
        
        # Extrai a região do veículo
        vehicle_roi = frame[y1_buf:y2_buf, x1_buf:x2_buf]
        
        # Verifica se há placa visível no veículo
        has_plate = self.detect_plate_in_vehicle(vehicle_roi)
        
        if not has_plate:
            print("⚠️ Placa não detectada no veículo, screenshot não salvo")
            return None
        
        # Tipo de veículo
        vehicle_type = {2: "Carro", 3: "Moto", 5: "Onibus", 7: "Caminhao"}.get(class_id, "Veiculo")
        
        # Nome do arquivo com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milissegundos
        filename = f"{vehicle_type}_{confidence:.2f}_{timestamp}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        
        # Salva a imagem
        cv2.imwrite(filepath, vehicle_roi)
        
        # Marca como salvo
        self.saved_vehicles.add(vehicle_id)
        
        # Atualiza a última posição salva
        self.last_saved_position = current_position
        
        print(f"✅ Screenshot salvo (placa detectada): {filepath}")
        
        return filepath
    
    def draw_detections(self, frame, vehicles):
        """
        Desenha as detecções no frame
        
        Args:
            frame: Frame do vídeo
            vehicles: Lista de detecções de veículos com (x1, y1, x2, y2, conf, class_id, is_front, is_stationary, frames_stopped)
            
        Returns:
            Frame com detecções desenhadas
        """
        frame_copy = frame.copy()
        
        valid_vehicles = 0
        for x1, y1, x2, y2, confidence, class_id, is_front, is_stationary, frames_stopped in vehicles:
            # Define cor baseada em status
            if is_stationary and is_front:
                color = (0, 255, 255)  # Amarelo - veículo parado de frente (para capturar placa)
                status = "PARADO DE FRENTE"
            elif is_front:
                color = (255, 0, 0)  # Azul - veículo de frente mas em movimento
                status = "DE FRENTE"
            elif is_stationary:
                color = (0, 0, 255)  # Vermelho - veículo parado mas não de frente
                status = "PARADO"
            else:
                color = (0, 255, 0)  # Verde - veículo normal
                status = ""
            
            # Desenha retângulo
            thickness = 3 if is_stationary and is_front else 2
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Desenha label com confiança e status
            vehicle_type = {2: "Carro", 3: "Moto", 5: "Ônibus", 7: "Caminhão"}.get(class_id, "Veículo")
            if status:
                label = f"{vehicle_type} - {status}"
            else:
                label = f"{vehicle_type}: {confidence:.2f}"
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame_copy, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Se está parado de frente, conta como válido
            if is_stationary and is_front:
                valid_vehicles += 1
        
        # Mostra contadores de veículos
        total_text = f"Total detectados: {len(vehicles)}"
        cv2.putText(frame_copy, total_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        valid_text = f"Parados de frente (placa): {valid_vehicles}"
        cv2.putText(frame_copy, valid_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
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
        self.show_window = False
        self.frame = None
        self.lock = threading.Lock()
        
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
        
        # Atualiza o frame atual com thread safety
        with self.lock:
            if ret:
                self.frame = frame
        
        return ret, frame
    
    def toggle_window(self):
        """Alterna a exibição da janela"""
        self.show_window = not self.show_window
        
        if not self.show_window:
            # Fecha a janela do OpenCV se ela existe
            try:
                cv2.destroyWindow(WINDOW_NAME)
                print("Janela fechada")
            except:
                pass  # Janela já não existe
        else:
            print("Janela aberta")
        
        return self.show_window
    
    def release(self):
        """Libera recursos do stream"""
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()


class ControlGUI:
    """Interface gráfica para controlar a exibição"""
    
    def __init__(self, stream, detector):
        """
        Inicializa a GUI
        
        Args:
            stream: Objeto RTSPStream
            detector: Objeto VehicleDetector
        """
        self.stream = stream
        self.detector = detector
        self.running = True
        
        # Cria a janela principal
        self.root = tk.Tk()
        self.root.title("Controle de Detecção de Veículos")
        self.root.geometry("450x280")
        
        # Label de status
        self.status_label = ttk.Label(
            self.root, 
            text="Status: Aguardando...", 
            font=("Arial", 12, "bold")
        )
        self.status_label.pack(pady=15)
        
        # Frame para contadores
        info_frame = ttk.Frame(self.root)
        info_frame.pack(pady=10)
        
        # Label para contador total
        self.count_label = ttk.Label(
            info_frame,
            text="Total de veículos: 0",
            font=("Arial", 10)
        )
        self.count_label.pack()
        
        # Label para veículos parados de frente
        self.valid_count_label = ttk.Label(
            info_frame,
            text="Veículos parados de frente: 0",
            font=("Arial", 10, "bold"),
            foreground="blue"
        )
        self.valid_count_label.pack(pady=5)
        
        # Label para screenshots salvos
        self.saved_count_label = ttk.Label(
            info_frame,
            text="Screenshots salvos: 0",
            font=("Arial", 9),
            foreground="green"
        )
        self.saved_count_label.pack(pady=2)
        
        # Botão para abrir/fechar janela
        self.toggle_button = ttk.Button(
            self.root,
            text="Abrir Visualização",
            command=self.toggle_visualization
        )
        self.toggle_button.pack(pady=10)
        
        # Botão para sair
        self.quit_button = ttk.Button(
            self.root,
            text="Sair",
            command=self.quit_application
        )
        self.quit_button.pack(pady=5)
        
    def toggle_visualization(self):
        """Alterna a visualização"""
        is_showing = self.stream.toggle_window()
        
        if is_showing:
            self.toggle_button.config(text="Fechar Visualização")
            self.status_label.config(text="Status: Visualização ATIVA")
        else:
            self.toggle_button.config(text="Abrir Visualização")
            self.status_label.config(text="Status: Visualização INATIVA")
    
    def quit_application(self):
        """Encerra a aplicação"""
        self.running = False
        self.stream.release()
        self.root.quit()
        self.root.destroy()
    
    def update_count(self, total_count, valid_count, saved_count):
        """Atualiza os contadores de veículos"""
        self.count_label.config(text=f"Total de veículos: {total_count}")
        self.valid_count_label.config(text=f"Veículos parados de frente: {valid_count}")
        self.saved_count_label.config(text=f"Screenshots salvos: {saved_count}")
    
    def run(self):
        """Inicia a thread principal da GUI"""
        thread = threading.Thread(target=self._run_detection)
        thread.daemon = True
        thread.start()
        
        # Inicia o loop principal do Tkinter
        self.root.mainloop()
    
    def _run_detection(self):
        """Thread de detecção"""
        frame_count = 0
        
        while self.running:
            # Lê frame
            ret, frame = self.stream.read_frame()
            
            if not ret:
                if self.running:
                    print("Erro ao ler frame. Tentando reconectar...")
                    try:
                        self.stream.release()
                        self.stream.connect()
                    except:
                        pass
                continue
            
            frame_count += 1
            
            # Detecta veículos
            vehicles = self.detector.detect_vehicles(frame)
            
            # Conta veículos parados de frente e salva imagens
            valid_count = 0
            for x1, y1, x2, y2, confidence, class_id, is_front, is_stationary, frames_stopped in vehicles:
                vehicle_id = id((x1, y1, x2, y2, frames_stopped))
                
                if is_front and is_stationary:
                    valid_count += 1
                    # Salva a imagem do veículo parado de frente
                    self.detector.save_vehicle_image(frame, x1, y1, x2, y2, confidence, class_id, vehicle_id)
                elif is_front and not is_stationary:
                    # Carro em movimento de frente - captura para salvar se parar ou sair da tela
                    self.detector.capture_moving_car(frame, x1, y1, x2, y2, confidence, class_id, vehicle_id, frame_count)
            
            # Verifica carros que desapareceram da tela
            self.detector.check_disappeared_cars(frame_count, vehicles)
            
            # Atualiza a GUI com os contadores
            try:
                saved_count = len(self.detector.saved_vehicles)
                self.root.after(0, self.update_count, len(vehicles), valid_count, saved_count)
            except:
                pass
            
            # Mostra informações no console a cada 30 frames
            if frame_count % 30 == 0:
                    print(f"Frame {frame_count}: {len(vehicles)} veículo(s) detectado(s) - {valid_count} parado(s) de frente")
            
            # Desenha detecções no frame
            if self.stream.show_window and frame is not None:
                frame_with_detections = self.detector.draw_detections(frame, vehicles)
                cv2.imshow(WINDOW_NAME, frame_with_detections)
                
                # Verifica se pressionou 'q' para fechar
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stream.show_window = False
                    try:
                        cv2.destroyWindow(WINDOW_NAME)
                    except:
                        pass
                    self.root.after(0, self.toggle_visualization)


def main():
    """Função principal"""
    detector = VehicleDetector()
    stream = RTSPStream(RTSP_URL)
    
    try:
        # Conecta ao stream
        stream.connect()
        
        print("Iniciando detecção de veículos...")
        
        # Cria e inicia a GUI
        gui = ControlGUI(stream, detector)
        gui.run()
                
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário")
    except Exception as e:
        print(f"Erro: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stream.release()
        cv2.destroyAllWindows()
        print("Stream liberado. Encerrando...")


if __name__ == "__main__":
    main()

