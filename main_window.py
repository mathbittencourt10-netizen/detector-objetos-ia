import sys
import cv2
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

from detector import DetectorObjetos

# ── Configurações do detector (fáceis de ajustar) ─────────────────────────
# "yolov8n.pt" = nano (mais rápido, ideal para webcam em CPU)
# "yolov8s.pt" = small (mais preciso, melhor para fotos, porém mais pesado)
MODELO = "yolov8s.pt"
# Confiança mínima (0 a 1): detecções abaixo desse valor são descartadas
CONFIANCA_MINIMA = 0.5


class MainWindow(QMainWindow):
    """Janela principal: carregar imagem ou usar webcam, com detecção de objetos via YOLO."""

    def __init__(self):
        super().__init__()
        self.detector = None          # carregado sob demanda (evita atraso na abertura da janela)
        self.captura = None           # cv2.VideoCapture, quando a webcam está ativa
        self.timer = QTimer()
        self.timer.timeout.connect(self._atualizar_frame_webcam)
        self.webcam_ativa = False

        self._configurar_janela()
        self._criar_interface()

    # ------------------------------------------------------------------ #
    #  Configuração da janela                                              #
    # ------------------------------------------------------------------ #

    def _configurar_janela(self):
        self.setWindowTitle("Detector de Objetos com IA")
        self.setMinimumSize(980, 620)
        self.setStyleSheet("""
            QMainWindow        { background-color: #F0F4F8; }
            QGroupBox          { font-weight: bold; border: 1px solid #CBD5E0;
                                 border-radius: 6px; margin-top: 10px; padding-top: 10px;
                                 background-color: white; }
            QGroupBox::title   { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #0B2545; }
            QPushButton        { padding: 10px; border-radius: 4px; font-size: 12px;
                                 font-weight: bold; border: none; color: white; }
            QPushButton:disabled { background-color: #B0BEC5; }
            QListWidget        { border: 1px solid #E2E8F0; border-radius: 4px;
                                 font-size: 12px; background: white; }
        """)

    def _criar_interface(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout_principal = QVBoxLayout(central)
        layout_principal.setContentsMargins(0, 0, 0, 0)
        layout_principal.setSpacing(0)

        layout_principal.addWidget(self._criar_topbar())

        corpo = QHBoxLayout()
        corpo.setContentsMargins(14, 14, 14, 14)
        corpo.setSpacing(12)
        layout_principal.addLayout(corpo)

        corpo.addWidget(self._criar_painel_controles(), stretch=0)
        corpo.addWidget(self._criar_painel_exibicao(), stretch=1)

        self.statusBar().showMessage("Pronto. Carregue uma imagem ou ligue a webcam para começar.")

    def _criar_topbar(self) -> QWidget:
        barra = QWidget()
        barra.setStyleSheet("background-color: #0B2545;")
        bl = QHBoxLayout(barra)
        bl.setContentsMargins(16, 12, 16, 12)
        titulo = QLabel("🎯  Detector de Objetos com IA")
        titulo.setFont(QFont("Arial", 14, QFont.Bold))
        titulo.setStyleSheet("color: #DFF0E8;")
        subtitulo = QLabel("YOLOv8 · PyTorch (CPU) · OpenCV")
        subtitulo.setStyleSheet("color: #7AAABB; font-size: 10px;")
        bl.addWidget(titulo)
        bl.addStretch()
        bl.addWidget(subtitulo)
        return barra

    # ------------------------------------------------------------------ #
    #  Painel esquerdo: controles                                          #
    # ------------------------------------------------------------------ #

    def _criar_painel_controles(self) -> QGroupBox:
        grupo = QGroupBox(" Controles ")
        grupo.setFixedWidth(280)
        layout = QVBoxLayout(grupo)
        layout.setSpacing(8)

        self.btn_carregar = QPushButton("🖼  Carregar Imagem")
        self.btn_carregar.setStyleSheet("background-color: #1D9E75;")
        self.btn_carregar.clicked.connect(self._carregar_imagem)
        layout.addWidget(self.btn_carregar)

        self.btn_webcam = QPushButton("📷  Ligar Webcam")
        self.btn_webcam.setStyleSheet("background-color: #1C7293;")
        self.btn_webcam.clicked.connect(self._alternar_webcam)
        layout.addWidget(self.btn_webcam)

        layout.addSpacing(8)
        layout.addWidget(QLabel("Objetos detectados:"))

        self.lista_deteccoes = QListWidget()
        layout.addWidget(self.lista_deteccoes, stretch=1)

        self.lbl_info = QLabel("Nenhuma detecção ainda.")
        self.lbl_info.setStyleSheet("color: #888; font-size: 10px;")
        self.lbl_info.setWordWrap(True)
        layout.addWidget(self.lbl_info)

        return grupo

    def _criar_painel_exibicao(self) -> QGroupBox:
        grupo = QGroupBox(" Visualização ")
        layout = QVBoxLayout(grupo)

        self.lbl_imagem = QLabel("Carregue uma imagem ou ligue a webcam\npara ver a detecção de objetos aqui.")
        self.lbl_imagem.setAlignment(Qt.AlignCenter)
        self.lbl_imagem.setStyleSheet("color: #999; font-size: 13px; background-color: #E9EDF2; border-radius: 4px;")
        self.lbl_imagem.setMinimumSize(600, 450)
        layout.addWidget(self.lbl_imagem)

        return grupo

    # ------------------------------------------------------------------ #
    #  Carregamento do modelo (sob demanda)                                #
    # ------------------------------------------------------------------ #

    def _garantir_detector_carregado(self) -> bool:
        """Carrega o modelo YOLO na primeira vez que for necessário."""
        if self.detector is not None:
            return True
        try:
            self.statusBar().showMessage(f"Carregando modelo de IA ({MODELO})... isso pode levar alguns segundos.")
            QApplication_processEvents()
            self.detector = DetectorObjetos(modelo=MODELO, confianca_minima=CONFIANCA_MINIMA)
            self.statusBar().showMessage("Modelo carregado com sucesso.")
            return True
        except Exception as e:
            QMessageBox.critical(
                self, "Erro ao carregar o modelo",
                f"Não foi possível carregar o modelo YOLO:\n{e}\n\n"
                "Verifique sua conexão com a internet (o modelo é baixado "
                "automaticamente na primeira execução)."
            )
            self.statusBar().showMessage("Falha ao carregar o modelo.")
            return False

    # ------------------------------------------------------------------ #
    #  Modo imagem                                                         #
    # ------------------------------------------------------------------ #

    def _carregar_imagem(self):
        if self.webcam_ativa:
            self._alternar_webcam()  # desliga a webcam antes de trocar de modo

        caminho, _ = QFileDialog.getOpenFileName(
            self, "Selecionar imagem", "",
            "Imagens (*.png *.jpg *.jpeg *.bmp)"
        )
        if not caminho:
            return

        if not self._garantir_detector_carregado():
            return

        frame = cv2.imread(caminho)
        if frame is None:
            QMessageBox.warning(self, "Erro", "Não foi possível abrir essa imagem.")
            return

        self.statusBar().showMessage("Detectando objetos...")
        frame_anotado, deteccoes = self.detector.detectar(frame)
        self._exibir_frame(frame_anotado)
        self._atualizar_lista_deteccoes(deteccoes)
        self.statusBar().showMessage(f"Detecção concluída — {len(deteccoes)} objeto(s) encontrado(s).")

    # ------------------------------------------------------------------ #
    #  Modo webcam                                                         #
    # ------------------------------------------------------------------ #

    def _alternar_webcam(self):
        if self.webcam_ativa:
            self._desligar_webcam()
            return

        if not self._garantir_detector_carregado():
            return

        self.captura = cv2.VideoCapture(0)
        if not self.captura.isOpened():
            QMessageBox.warning(
                self, "Webcam não encontrada",
                "Não foi possível acessar nenhuma webcam neste computador.\n"
                "Use 'Carregar Imagem' para testar com uma foto."
            )
            self.captura = None
            return

        self.webcam_ativa = True
        self.btn_webcam.setText("⏹  Desligar Webcam")
        self.btn_webcam.setStyleSheet("background-color: #C0392B;")
        self.btn_carregar.setEnabled(False)
        self.timer.start(30)  # tenta atualizar a cada 30ms (a velocidade real depende do processamento)
        self.statusBar().showMessage("Webcam ligada — detectando em tempo real.")

    def _desligar_webcam(self):
        self.timer.stop()
        if self.captura is not None:
            self.captura.release()
            self.captura = None
        self.webcam_ativa = False
        self.btn_webcam.setText("📷  Ligar Webcam")
        self.btn_webcam.setStyleSheet("background-color: #1C7293;")
        self.btn_carregar.setEnabled(True)
        self.statusBar().showMessage("Webcam desligada.")

    def _atualizar_frame_webcam(self):
        if self.captura is None:
            return
        ok, frame = self.captura.read()
        if not ok:
            self.statusBar().showMessage("Não foi possível ler o frame da webcam.")
            return

        frame_anotado, deteccoes = self.detector.detectar(frame)
        self._exibir_frame(frame_anotado)
        self._atualizar_lista_deteccoes(deteccoes)

    # ------------------------------------------------------------------ #
    #  Exibição                                                            #
    # ------------------------------------------------------------------ #

    def _exibir_frame(self, frame_bgr):
        """Converte um frame OpenCV (BGR) para QPixmap e mostra no QLabel."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        altura, largura, canais = frame_rgb.shape
        bytes_por_linha = canais * largura
        imagem_qt = QImage(frame_rgb.data, largura, altura, bytes_por_linha, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(imagem_qt)

        pixmap = pixmap.scaled(
            self.lbl_imagem.width(), self.lbl_imagem.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.lbl_imagem.setPixmap(pixmap)

    def _atualizar_lista_deteccoes(self, deteccoes):
        self.lista_deteccoes.clear()
        if not deteccoes:
            self.lbl_info.setText("Nenhum objeto detectado nesta imagem.")
            return

        for d in deteccoes:
            texto = f"{d['classe']}  —  {d['confianca'] * 100:.0f}%"
            item = QListWidgetItem(texto)
            self.lista_deteccoes.addItem(item)

        self.lbl_info.setText(f"{len(deteccoes)} objeto(s) detectado(s).")

    # ------------------------------------------------------------------ #
    #  Encerramento                                                        #
    # ------------------------------------------------------------------ #

    def closeEvent(self, event):
        """Garante que a webcam seja liberada ao fechar a janela."""
        if self.captura is not None:
            self.captura.release()
        event.accept()


def QApplication_processEvents():
    """Pequeno auxiliar para manter a interface responsiva durante o carregamento do modelo."""
    from PyQt5.QtWidgets import QApplication
    QApplication.processEvents()
