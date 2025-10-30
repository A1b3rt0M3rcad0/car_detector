# Sistema de Detecção de Veículos com Captura de Placa

Sistema de detecção de veículos usando YOLO e stream RTSP, com foco em veículos parados de frente para captura de placas.

## Características

- 🎯 **Detecção de Veículos**: Usando modelo YOLOv8 para detectar carros, motos, ônibus e caminhões
- 🚗 **Detecção de Frente**: Identifica veículos de frente baseado em proporção e tamanho
- 🛑 **Detecção de Parados**: Rastreia movimento para identificar veículos parados
- 🎨 **Interface Gráfica**: Controle através de GUI com Tkinter
- 📊 **Visualização**: Cores diferentes para diferentes estados dos veículos
- 📡 **Stream RTSP**: Conecta a câmeras IP via RTSP

## Cores das Detecções

- **🟡 Amarelo**: Veículo parado de frente (ideal para capturar placa) - ESPESSURA 3
- **🔵 Azul**: Veículo de frente mas em movimento
- **🔴 Vermelho**: Veículo parado mas não de frente
- **🟢 Verde**: Veículo normal

## Requisitos

```bash
pip install opencv-python numpy ultralytics python-dotenv
```

## Configuração

Crie um arquivo `.env` na raiz do projeto:

```env
# URL do stream RTSP
RTSP_URL=rtsp://usuario:senha@ip:porta/stream

# Modelo YOLO (padrão: yolov8n.pt)
YOLO_MODEL=yolov8n.pt

# IDs das classes de veículos (2=car, 3=motorcycle, 5=bus, 7=truck)
VEHICLE_CLASS_IDS=2,3,5,7

# Threshold de confiança (0.0 a 1.0)
CONFIDENCE_THRESHOLD=0.5
```

## Como Usar

1. Execute o sistema:
```bash
python app.py
```

2. Uma janela de controle será aberta com:
   - Status da visualização
   - Contador de veículos totais detectados
   - Contador de veículos parados de frente

3. Clique em "Abrir Visualização" para ver o vídeo em tempo real

4. Veículos parados de frente aparecerão em **amarelo com espessura 3**

5. Pressione 'q' na janela de vídeo ou clique em "Sair" para encerrar

## Funcionalidades Técnicas

### Detecção de Veículos de Frente
- Analisa a proporção (aspect ratio) da bounding box
- Considera veículos de frente aqueles com ratio entre 0.8 e 1.5
- Verifica se o veículo é grande o suficiente (width > 100px, height > 80px)

### Detecção de Veículos Parados
- Rastreia a posição dos veículos através de múltiplos frames
- Considera parado se o movimento for menor que 5 pixels por frame
- Confirma parado após 10 frames consecutivos sem movimento

### Interface
- **Threading**: Interface GUI e detecção rodam em threads separadas
- **Thread-safe**: Uso de locks para segurança dos dados
- **Tempo real**: Atualização contínua dos contadores

## Estrutura do Projeto

```
car_detector/
├── app.py              # Código principal
├── config.py           # Configurações e variáveis
├── pyproject.toml      # Dependências
├── README.md           # Este arquivo
└── .env                # Variáveis de ambiente (criar)
```

## Exemplo de Uso

O sistema é ideal para:
- 🚪 **Portarias**: Identificar veículos parados em portaria
- 🏪 **Estacionamentos**: Capturar placas na entrada
- 🛣️ **Pedágios**: Identificar veículos parados no guichê
- 🏢 **Controle de Acesso**: Veículos parados para identificação

## Notas

- Certifique-se de que a câmera está bem posicionada para capturar veículos de frente
- A iluminação adequada melhora a detecção
- O sistema funciona melhor com veículos a uma distância de 3-8 metros

