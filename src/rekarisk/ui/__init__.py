"""
Rekarisk UI Module — Desktop application interface.

Provides the PyQt6-based desktop GUI with dock-based layout,
project management, substance selection, and scenario editing.
"""

try:
    from .main_window import MainWindow
except ImportError:
    MainWindow = None

try:
    from .menu_bar import RekariskMenuBar
except ImportError:
    RekariskMenuBar = None
    
try:
    from .project_panel import ProjectPanel
except ImportError:
    ProjectPanel = None
    
try:
    from .substance_selector import SubstanceSelector
except ImportError:
    SubstanceSelector = None
    
try:
    from .source_term_panel import SourceTermPanel
except ImportError:
    SourceTermPanel = None
    
try:
    from .source_term_results import SourceTermResultsPanel, SourceTermResultsDock
except ImportError:
    SourceTermResultsPanel = None
    SourceTermResultsDock = None

try:
    from .dispersion_panel import DispersionPanel
except ImportError:
    DispersionPanel = None

try:
    from .dispersion_results import DispersionResultsPanel
except ImportError:
    DispersionResultsPanel = None

try:
    from .fire_panel import FirePanel
except ImportError:
    FirePanel = None

try:
    from .fire_results import FireResultsPanel
except ImportError:
    FireResultsPanel = None

try:
    from .qra_panel import QRAPanel
except ImportError:
    QRAPanel = None

try:
    from .qra_results import QRAResultsPanel, QRAResultsDock
except ImportError:
    QRAResultsPanel = None
    QRAResultsDock = None
