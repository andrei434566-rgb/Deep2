"""Entry point for DeepCore 2."""

import sys

from PySide6.QtWidgets import QApplication

from app.ui.windows.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("DeepCore 2")
    window = MainWindow()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
