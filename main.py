# IMPORTANTE (Windows): o torch precisa ser importado ANTES do PyQt5 e do OpenCV.
# Se o PyQt for carregado primeiro, o PyTorch pode falhar ao iniciar com
# "WinError 1114" (falha na inicialização de DLL — c10.dll).
import torch  # noqa: F401

import sys
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
