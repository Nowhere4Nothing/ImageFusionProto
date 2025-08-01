import sys
from PySide6.QtWidgets import QApplication
from GUI.main_window import DicomViewer

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = DicomViewer()
    viewer.resize(1000, 800)
    viewer.show()
    sys.exit(app.exec())