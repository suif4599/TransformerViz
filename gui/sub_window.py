from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QSizePolicy, QApplication, QStyle
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QScreen
from types import TracebackType
from html import escape
import traceback
import os
from .theme_manager import ThemeManager

ROOT_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".."
    )
)

class HelpWindow(QDialog):
    def __init__(self, language="en", parent=None):
        if language not in ["en", "zh"]:
            raise ValueError("Language must be either 'en' or 'zh'")
        super().__init__(parent)
        
        self.theme_manager = ThemeManager()
        
        self.setWindowTitle("Help")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setModal(True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        
        self.setStyleSheet(self.theme_manager.get_stylesheet())
        
        style = self.style()
        if style:
            self.setWindowIcon(style.standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        self.setWindowTitle("Help")

        self.vertical_layout = QVBoxLayout(self)
        self.vertical_layout.setContentsMargins(0, 0, 0, 0)
        self.vertical_layout.setSpacing(0)
        self.vertical_layout.setObjectName("verticalLayout")
        self.vertical_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(self.vertical_layout)

        self.text_browser = QTextBrowser(self)
        self.text_browser.setOpenExternalLinks(True)
        with open(os.path.join(ROOT_PATH, f"help-{language}.md"), "r", encoding="utf-8") as f:
            help_text = f.read()
        self.text_browser.setMarkdown(help_text)
        self.text_browser.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        text_color = self.theme_manager.get_color('text_primary')
        self.text_browser.setStyleSheet(f"font-size: 16pt; font-weight: bold; color: {text_color};")

        self.vertical_layout.addWidget(self.text_browser)
        
        primary_screen = QApplication.primaryScreen()
        if primary_screen:
            screen_size = primary_screen.availableGeometry().size()
            self.resize(
                int(screen_size.width() * 0.8), 
                int(screen_size.height() * 0.8)
            )

def get_about_html(theme_colors: dict) -> str:
    """Generate themed about page HTML"""
    bg_color = theme_colors.get('background', '#FFFFFF')
    text_color = theme_colors.get('text_primary', '#212121')
    primary_color = theme_colors.get('primary', '#2196F3')
    link_color = theme_colors.get('primary', '#2196F3')
    
    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>About TransformerViz</title>
    <style>
        body {{
            margin: 20px;
            font-family: Arial, sans-serif;
            background-color: {bg_color};
            color: {text_color};
        }}
        h1 {{
            font-weight: bold;
            text-align: center;
            color: {primary_color};
        }}
        h2 {{
            font-weight: normal;
            text-align: center;
            color: {text_color};
        }}
        a {{
            color: {link_color};
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <h1>A Transformer Attention Visualization Tool</h1>
    <h2><a href="https://github.com/suif4599">@suif4599</a></h2>
    <h2><a href="https://github.com/suif4599/TransformerViz">repo</a></h2>
</body>
</html>
"""

class AboutWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.theme_manager = ThemeManager()
        
        self.setWindowTitle("About TransformerViz")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        
        self.setStyleSheet(self.theme_manager.get_stylesheet())
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.text_browser = QTextBrowser(self)
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.text_browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.text_browser.setHtml(get_about_html(self.theme_manager.current_theme))
        
        layout.addWidget(self.text_browser)
        self.setLayout(layout)
        
        primary_screen = QApplication.primaryScreen()
        if primary_screen:
            screen_size = primary_screen.availableGeometry().size()
            self.resize(
                int(screen_size.width() * 0.3), 
                int(screen_size.height() * 0.2)
            )

class ExceptionWindow(QDialog):
    def __init__(self, parent, exc_type: type, exc_value: object, tb: TracebackType):
        super().__init__(parent)

        self.theme_manager = ThemeManager()
        
        self.setStyleSheet(self.theme_manager.get_stylesheet())
        
        # Get theme colors for HTML
        bg_color = self.theme_manager.get_color('background')
        text_color = self.theme_manager.get_color('text_primary')
        error_color = self.theme_manager.get_color('error')
        border_color = self.theme_manager.get_color('border')
        primary_color = self.theme_manager.get_color('primary')

        exc_type_str = escape(f"{exc_type.__module__}.{exc_type.__name__}")
        exc_value_str = escape(str(exc_value)) if exc_value else ""
        tb_lines_raw = traceback.format_tb(tb)
        tb_lines = []
        for line in tb_lines_raw:
            tb_lines.extend(line.splitlines())
        formatted_tb = "".join([f"<p>{escape(line)}</p>" for line in tb_lines])
        
        message = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        body {{
            background-color: {bg_color};
            color: {text_color};
            font-family: 'Courier New', monospace;
            margin: 0;
            padding: 0;
        }}
        </style>
        </head>
        <body>
        <div style="
            font-family: 'Courier New', monospace;
            border: 1px solid {border_color};
            border-radius: 5px;
            padding: 15px;
            margin: 10px;
            color: {text_color};
            background-color: {bg_color};
        ">
            <div style="
                color: {error_color};
                padding: 10px;
                border-radius: 3px;
                margin-bottom: 10px;
                font-weight: bold;
            ">
                {exc_type_str}
            </div>

            <div style="
                margin-bottom: 15px;
                padding: 10px;
                border-left: 4px solid {primary_color};
            ">
                <span style="color: {text_color};">Message: </span>
                <span style="color: {primary_color};">{exc_value_str}</span>
            </div>

            <div style="
                color: {text_color};
                padding: 8px;
                border-radius: 3px;
                margin-bottom: 10px;
            ">
                Stack Trace:
            </div>

            <div style="
                padding: 10px;
                border-radius: 3px;
                line-height: 1.6;
                white-space: pre-wrap;
                color: {text_color};
            ">
                {formatted_tb}
            </div>
        </div>
        </body>
        </html>
        """
        
        self.setWindowTitle(f"{exc_type.__module__}.{exc_type.__name__}")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.text_browser = QTextBrowser(self)
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.text_browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.text_browser.setHtml(message)
        
        layout.addWidget(self.text_browser)
        self.setLayout(layout)
        
        primary_screen = QApplication.primaryScreen()
        if primary_screen:
            screen_size = primary_screen.availableGeometry().size()
            self.resize(
                int(screen_size.width() * 0.8), 
                int(screen_size.height() * 0.8)
            )
