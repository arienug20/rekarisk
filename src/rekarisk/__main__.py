"""
Launch Rekarisk GUI via ``python -m rekarisk``.

Example::

    python -m rekarisk
"""

from __future__ import annotations

import sys


def main() -> None:
    """Entry point: create QApplication and show the main window."""
    from PyQt6.QtWidgets import QApplication
    from rekarisk.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Rekarisk")
    app.setOrganizationName("Rekarisk")
    app.setOrganizationDomain("rekarisk.org")

    # High-DPI scaling
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
