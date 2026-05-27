#!/usr/bin/env python3
"""Capture mixture editor."""
import os, sys
os.environ.setdefault("DISPLAY", ":99")
os.environ["QT_QPA_PLATFORM"] = "xcb"

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

app = QApplication(sys.argv)
OUTPUT = "/home/arienugraha-rei/rekarisk/screenshots"

def run():
    from rekarisk.core.substance_db import get_database
    db = get_database()
    from rekarisk.ui.mixture_editor import MixtureEditorDialog
    d = MixtureEditorDialog(substance_db=db)
    d.resize(700, 500); d.show()
    from PyQt6.QtWidgets import QApplication
    QApplication.processEvents()
    p = d.grab()
    p.save(os.path.join(OUTPUT, "16_mixture_editor.png"))
    print("✓ 16_mixture_editor.png")
    d.close()
    print("✅ Done!")
    app.quit()

QTimer.singleShot(500, run)
app.exec()
