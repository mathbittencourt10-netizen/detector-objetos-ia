"""
Encapsula o modelo YOLO (Ultralytics) para detecção de objetos.

Usa PyTorch em modo CPU — não depende de GPU dedicada, o que garante
compatibilidade com qualquer computador, independente da placa de vídeo.
"""
from ultralytics import YOLO


class DetectorObjetos:
    """Carrega um modelo YOLO e detecta objetos em imagens ou frames de vídeo."""

    def __init__(self, modelo: str = "yolov8n.pt", confianca_minima: float = 0.5):
        """
        modelo: nome do modelo YOLO (baixado automaticamente na primeira execução).
                - "yolov8n.pt" (nano): o mais leve, ideal para webcam em CPU.
                - "yolov8s.pt" (small): mais preciso, porém mais pesado.
        confianca_minima: só considera detecções com confiança >= esse valor (0 a 1).
        """
        self.model = YOLO(modelo)
        self.confianca_minima = confianca_minima

    def detectar(self, frame):
        """
        Executa a detecção em um frame (array NumPy BGR, formato do OpenCV).

        Retorna uma tupla (frame_anotado, deteccoes):
        - frame_anotado: o mesmo frame, com caixas e rótulos desenhados
        - deteccoes: lista de dicts {"classe": str, "confianca": float},
                     uma entrada por objeto detectado
        """
        resultados = self.model(frame, conf=self.confianca_minima, verbose=False)
        resultado = resultados[0]

        frame_anotado = resultado.plot()  # já desenha as caixas + rótulos no frame

        deteccoes = []
        for caixa in resultado.boxes:
            classe_id = int(caixa.cls[0])
            nome_classe = self.model.names[classe_id]
            confianca = float(caixa.conf[0])
            deteccoes.append({"classe": nome_classe, "confianca": confianca})

        # Ordena as mais confiantes primeiro, para exibição
        deteccoes.sort(key=lambda d: d["confianca"], reverse=True)

        return frame_anotado, deteccoes
