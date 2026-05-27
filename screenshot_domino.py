import sys, os
sys.path.insert(0, '/home/arienugraha-rei/rekarisk/src')
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from PyQt6.QtWidgets import QApplication
app = QApplication(sys.argv)

from rekarisk.ui.domino_panel import DominoPanel

panel = DominoPanel()
panel.resize(1200, 900)
panel.show()

pixmap = panel.grab()
out = '/home/arienugraha-rei/.openclaw/workspace/outputs/risk_scenario/12_domino_panel.png'
pixmap.save(out)
print(f'Screenshot saved: {out}')
app.quit()
