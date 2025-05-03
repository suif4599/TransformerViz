from PyQt5.QtWidgets import QDialog, QStyle, QVBoxLayout, QTextBrowser, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QScreen
from PyQt5.QtWidgets import QApplication
from types import TracebackType
from html import escape
import traceback
import os

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
        self.setWindowTitle("Help")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setModal(True)
        self.setWindowModality(Qt.ApplicationModal)
        self.setStyleSheet("background-color: rgb(216, 216, 216);")
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self.setWindowTitle("Help")

        self.vertical_layout = QVBoxLayout(self)
        self.vertical_layout.setContentsMargins(0, 0, 0, 0)
        self.vertical_layout.setSpacing(0)
        self.vertical_layout.setObjectName("verticalLayout")
        self.vertical_layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.vertical_layout)

        self.text_browser = QTextBrowser(self)
        self.text_browser.setOpenExternalLinks(True)
        with open(os.path.join(ROOT_PATH, f"help-{language}.md"), "r", encoding="utf-8") as f:
            help_text = f.read()
        self.text_browser.setMarkdown(help_text)
        self.text_browser.setAlignment(Qt.AlignTop)
        self.text_browser.setStyleSheet("font-size: 16pt; font-weight: bold; color: black;")

        self.vertical_layout.addWidget(self.text_browser)
        
        screen_size = QScreen.availableGeometry(QApplication.primaryScreen()).size()
        self.resize(
            int(screen_size.width() * 0.8), 
            int(screen_size.height() * 0.8)
        )

ABOUT = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>About TransformerViz</title>
    <style>
        body {
            margin: 20px;
            font-family: Arial, sans-serif;
            background-color: rgb(216, 216, 216);
            color: #333;
        }
        h1 {
            font-weight: bold;
            text-align: center;
            color: #2c3e50;
        }
        h2 {
            font-weight: normal;
            text-align: center;
            color: #34495e;
        }
        a {
            color: #2980b9;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
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
        
        self.setWindowTitle("About TransformerViz")
        self.setAttribute(Qt.WA_DeleteOnClose)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.text_browser = QTextBrowser(self)
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setAlignment(Qt.AlignTop)
        self.text_browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.text_browser.setHtml(ABOUT)
        
        layout.addWidget(self.text_browser)
        self.setLayout(layout)
        
        screen_size = QScreen.availableGeometry(QApplication.primaryScreen()).size()
        self.resize(
            int(screen_size.width() * 0.3), 
            int(screen_size.height() * 0.2)
        )

class ExceptionWindow(QDialog):
    def __init__(self, parent, exc_type: type, exc_value: object, tb: TracebackType):
        super().__init__(parent)

        exc_type_str = escape(f"{exc_type.__module__}.{exc_type.__name__}")
        exc_value_str = escape(str(exc_value)) if exc_value else ""
        tb_lines_raw = traceback.format_tb(tb)
        tb_lines = []
        for line in tb_lines_raw:
            tb_lines.extend(line.splitlines())
        formatted_tb = "".join([f"<p>{escape(line)}</p>" for line in tb_lines])
        message = f"""
        <div style="
            font-family: 'Courier New', monospace;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            padding: 15px;
            margin: 10px;
            color: #333;
        ">
            <div style="
                color: #c7254e;
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
                border-left: 4px solid #3498db;
            ">
                <span style="color: #2c3e50;">Message: </span>
                <span style="color: blue;">{exc_value_str}</span>
            </div>

            <div style="
                color: black;
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
            ">
                {formatted_tb}
            </div>
        </div>
        """
        
        self.setWindowTitle(f"{exc_type.__module__}.{exc_type.__name__}")
        self.setAttribute(Qt.WA_DeleteOnClose)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.text_browser = QTextBrowser(self)
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setAlignment(Qt.AlignTop)
        self.text_browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.text_browser.setHtml(message)
        
        layout.addWidget(self.text_browser)
        self.setLayout(layout)
        
        screen_size = QScreen.availableGeometry(QApplication.primaryScreen()).size()
        self.resize(
            int(screen_size.width() * 0.8), 
            int(screen_size.height() * 0.8)
        )
