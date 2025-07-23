from PyQt6.QtWidgets import QMainWindow, QApplication, QListView, QTextBrowser, \
    QComboBox, QLineEdit, QPushButton, QVBoxLayout, QFrame, \
    QHBoxLayout, QLabel, QStyle
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QFont, QAction, QKeyEvent
from PyQt6.QtCore import Qt, QItemSelectionModel
from .Ui_root import Ui_MainWindow
from core import AbstractModule
from .viz_frame import VizFrameScroll
from .validators import FloatValidator, PositiveIntValidator
from .sub_window import HelpWindow, AboutWindow, ExceptionWindow
from .theme_manager import ThemeManager
from threading import Thread
from typing import Optional, Callable

def get_loading_html(theme_colors: dict) -> str:
    """Generate themed loading HTML"""
    bg_color = theme_colors.get('background', '#FFFFFF')
    text_color = theme_colors.get('text_primary', '#212121')
    
    return f"""
<!DOCTYPE html>
<html>
<head>
<style>
body {{
  background-color: {bg_color};
  color: {text_color};
  margin: 0;
  padding: 0;
  font-family: Arial, sans-serif;
}}
.container {{
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
  width: 100%;
}}
.loading-text {{
  margin-top: 10px;
  text-align: center;
  font-size: 50px;
  color: {text_color};
}}
</style>
</head>
<body>
<div class="container">
  <div>
    <div class="loading-text">Loading Model...</div>
  </div>
</div>
</body>
</html>
"""

def wrap_html_with_theme(html_content: str, theme_colors: dict) -> str:
    """Wrap HTML content with theme styles"""
    bg_color = theme_colors.get('background', '#FFFFFF')
    text_color = theme_colors.get('text_primary', '#212121')
    primary_color = theme_colors.get('primary', '#2196F3')
    surface_color = theme_colors.get('surface', '#F5F5F5')
    border_color = theme_colors.get('border', '#E0E0E0')
    
    # Remove hardcoded color styles that conflict with theme
    import re
    # Remove color: #333 and similar hardcoded colors from inline styles
    html_content = re.sub(r'color:\s*#[0-9a-fA-F]{3,6};?', '', html_content)
    # Remove background: #xxx and background-color: #xxx
    html_content = re.sub(r'background(-color)?:\s*#[0-9a-fA-F]{3,6};?', '', html_content)
    
    # Create theme styles
    theme_style = f"""
<style>
body {{
    background-color: {bg_color} !important;
    color: {text_color} !important;
    font-family: 'Microsoft YaHei', Arial, sans-serif !important;
}}
div {{
    color: {text_color} !important;
    background-color: transparent !important;
}}
h1, h2, h3, h4, h5, h6 {{
    color: {primary_color} !important;
    border-color: {primary_color} !important;
    background-color: transparent !important;
}}
p, li, span {{
    color: {text_color} !important;
    background-color: transparent !important;
}}
div[style*="border"] {{
    background-color: transparent !important;
    border-color: {border_color} !important;
}}
strong {{
    color: {text_color} !important;
    background-color: transparent !important;
}}
ul, ol {{
    color: {text_color} !important;
}}
</style>
"""
    
    # If HTML contains <html> tag, insert styles
    if '<html>' in html_content or '<HTML>' in html_content:
        if '<head>' in html_content or '<HEAD>' in html_content:
            return html_content.replace('<head>', f'<head>{theme_style}', 1)
        else:
            return html_content.replace('<html>', f'<html><head>{theme_style}</head>', 1)
    else:
        # Wrap entire content
        return f"""
<!DOCTYPE html>
<html>
<head>
{theme_style}
</head>
<body>
{html_content}
</body>
</html>
"""

class _Root(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None, theme_manager=None):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.setupUi(self)
        self.showMaximized()
        self.setWindowTitle("Attention Visualization")
        self.module_list.setSelectionMode(QListView.SelectionMode.SingleSelection)

        self.module_list: QListView
        self.text_description: QTextBrowser
        self.position_mode_option: QComboBox
        self.layer_mix_option: QComboBox
        self.head_mix_option: QComboBox
        self.text_input: QLineEdit
        self.confirm_button: QPushButton
        self.verticalLayout_6: QVBoxLayout
        self.temperature_input: QLineEdit
        self.temperature_set_button: QPushButton
        self.fontsize_input: QLineEdit
        self.fontsize_set_button: QPushButton
        self.help_action_en: QAction
        self.help_action_zh: QAction
        self.about_action: QAction

        self.viz_scroll = VizFrameScroll(self.frame_attention, 1)
        self.verticalLayout_6.addWidget(self.viz_scroll)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)

        self.temperature_input.setValidator(FloatValidator())
        self.temperature_input.setPlaceholderText("1.0")

        self.fontsize_input.setValidator(PositiveIntValidator())

        self.more_option_frame = []
        self.more_option_layout = []
        self.more_option_label = []
        self.more_option_combo = []
        self.more_option_input = []
        self.more_option_button = []
        
        # Connect theme menu actions
        if self.theme_manager:
            self.theme_auto_action.triggered.connect(lambda: self._set_theme_mode('auto'))
            self.theme_light_action.triggered.connect(lambda: self._set_theme_mode('light'))
            self.theme_dark_action.triggered.connect(lambda: self._set_theme_mode('dark'))
            
            # Update menu state based on current theme mode
            self._update_theme_menu_state()
    
    def _set_theme_mode(self, mode: str):
        """Set theme mode and update UI"""
        if self.theme_manager:
            from .theme_manager import ThemeMode
            theme_mode = ThemeMode(mode)
            self.theme_manager.set_theme_mode(theme_mode)
            self._update_theme_menu_state()
    
    def _update_theme_menu_state(self):
        """Update theme menu checked state"""
        if self.theme_manager:
            mode = self.theme_manager.get_theme_mode()
            self.theme_auto_action.setChecked(mode.value == 'auto')
            self.theme_light_action.setChecked(mode.value == 'light') 
            self.theme_dark_action.setChecked(mode.value == 'dark')


    def keyPressEvent(self, a0):
        event = a0
        if event and event.key() == Qt.Key.Key_Escape:
            self.close()
        return super().keyPressEvent(event)
    
    def add_control_option(self, name: str, option: list[str] | None, callback: Optional[Callable[[str], None]]):
        self.more_option_frame.append(QFrame(self.frame_control))
        self.more_option_frame[-1].setFrameShape(QFrame.Shape.StyledPanel)
        self.more_option_frame[-1].setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_5.addWidget(self.more_option_frame[-1])
        self.more_option_layout.append(QHBoxLayout(self.more_option_frame[-1]))
        self.more_option_label.append(QLabel(self.more_option_frame[-1]))
        self.more_option_label[-1].setText(name + ": ")
        self.more_option_layout[-1].addWidget(self.more_option_label[-1])
        if option is None:
            self.more_option_input.append(QLineEdit(self.more_option_frame[-1]))
            self.more_option_input[-1].setPlaceholderText(name)
            self.more_option_layout[-1].addWidget(self.more_option_input[-1])
            self.more_option_button.append(QPushButton(self.more_option_frame[-1]))
            self.more_option_layout[-1].addWidget(self.more_option_button[-1])
            self.more_option_button[-1].setText("Set")
            self.more_option_combo.append(None)
            def wrapper(input_widget=self.more_option_input[-1]):
                text = input_widget.text()
                if text:
                    input_widget.setPlaceholderText(text)
                    input_widget.clear()
                if callback:
                    callback(text)
            self.more_option_button[-1].clicked.connect(wrapper)
            self.more_option_input[-1].returnPressed.connect(wrapper)
        else:
            self.more_option_combo.append(QComboBox(self.more_option_frame[-1]))
            for item in option:
                self.more_option_combo[-1].addItem(item)
            self.more_option_layout[-1].addWidget(self.more_option_combo[-1])
            self.more_option_input.append(None)
            self.more_option_button.append(None)
            self.more_option_combo[-1].currentTextChanged.connect(callback)
            self.more_option_combo[-1].setCurrentIndex(0)
        self.more_option_frame[-1].setLayout(self.more_option_layout[-1])
        
        if self.theme_manager:
            surface_color = self.theme_manager.get_color('surface')
            border_color = self.theme_manager.get_color('border')
            self.more_option_frame[-1].setStyleSheet(f"""
                QFrame {{
                    background-color: {surface_color};
                    border: 1px solid {border_color};
                }}
            """)
    
    def clear_control_option(self):
        for frame in self.more_option_frame:
            self.verticalLayout_5.removeWidget(frame)
            frame.setParent(None)
            frame.deleteLater()
        self.more_option_frame.clear()
        self.more_option_layout.clear()
        self.more_option_label.clear()
        self.more_option_combo.clear()
        self.more_option_input.clear()
        self.more_option_button.clear()

    

class Root:
    def __init__(self, argv: list = []):
        self.app = QApplication(argv)
        self.theme_manager = ThemeManager()
        self.win = _Root(theme_manager=self.theme_manager)
        
        # Connect theme change signal
        self.theme_manager.theme_changed.connect(self.apply_theme)
        
        self.apply_theme()
        
        self.modules: dict[str, AbstractModule] = {}
        
        self.selected_module_name: Optional[str] = None
        self.active_module: Optional[AbstractModule] = None
        self.__item_model = QStandardItemModel()
        self.win.module_list.setModel(self.__item_model)
        self.win.module_list.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        selection_model = self.win.module_list.selectionModel()
        if selection_model:
            selection_model.currentChanged.connect(self.on_module_selected)

        self.position_mode_option: Optional[str] = None
        self.win.position_mode_option.currentTextChanged.connect(self.on_position_mode_changed)

        self.layer_mix_option: Optional[str] = None
        self.win.layer_mix_option.currentTextChanged.connect(self.on_layer_mix_option_changed)

        self.head_mix_option: Optional[str] = None
        self.win.head_mix_option.currentTextChanged.connect(self.on_head_mix_option_changed)

        self.win.confirm_button.clicked.connect(self.on_confirm_button_clicked)
        self.win.text_input.returnPressed.connect(self.on_confirm_button_clicked)
        
        self.temperature = 1.0
        self.win.temperature_set_button.clicked.connect(self.on_temperature_set_button_clicked)
        self.win.temperature_input.returnPressed.connect(self.on_temperature_set_button_clicked)

        self.fontsize = 20
        self.win.fontsize_set_button.clicked.connect(self.on_fontsize_set_button_clicked)
        self.win.fontsize_input.setPlaceholderText(f"{self.fontsize}pt")
        self.win.fontsize_input.returnPressed.connect(self.on_fontsize_set_button_clicked)
        self.win.viz_scroll.set_fontsize(self.fontsize)

        self.win.help_action_en.triggered.connect(lambda: self.show_help("en"))
        self.win.help_action_zh.triggered.connect(lambda: self.show_help("zh"))
        self.win.about_action.triggered.connect(self.show_about)
        style = self.app.style()
        if style:
            self.win.setWindowIcon(style.standardIcon(QStyle.StandardPixmap.SP_FileIcon))

    def mainloop(self):
        self.win.show()
        self.app.exec()
    
    def add_module(self, module: AbstractModule):
        self.modules[name := module.get_name()] = module
        item = QStandardItem(name)
        item.setEditable(False)
        item.setFont(QFont("Arial", 18))
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.__item_model.appendRow(item)
        if len(self.modules) == 1:
            self.win.module_list.setCurrentIndex(self.__item_model.index(0, 0))
        
    def visualize(self):
        if not self.active_module or not self.position_mode_option or not self.layer_mix_option or not self.head_mix_option:
            return
        target, source = self.active_module.get_sentence(self.position_mode_option)
        self.win.viz_scroll.reset(
            self.active_module.get_n_head(
                self.position_mode_option,
                self.layer_mix_option,
                self.head_mix_option
            ), 
            self.key_changed
        )
        self.win.viz_scroll.show_sentence(target, source)
        self.win.viz_scroll.set_fontsize(self.fontsize)

    def on_module_selected(self, current, previous):
        if current.data() == self.selected_module_name:
            return
        self.checkout_to(current)

    def checkout_to(self, new):
        exception = None
        raw_module = self.active_module if hasattr(self, "active_module") else None
        raw_row = None
        raw_index = None
        if raw_module:
            raw_row = self.__item_model.findItems(raw_module.get_name())[0].row()
            raw_index = self.__item_model.index(raw_row, 0)
        def inner():
            nonlocal exception
            try:
                if self.active_module:
                    self.active_module.unload()
                self.selected_module_name = new.data()
                if self.selected_module_name:
                    self.active_module = self.modules[self.selected_module_name]
                    self.active_module.load()
            except Exception as e:
                exception = e
        
        self.win.text_description.clear()
        self.win.text_description.setHtml(get_loading_html(self.theme_manager.current_theme))
        loading_thread = Thread(target=inner)
        loading_thread.start()
        self.win.setEnabled(False)
        while loading_thread.is_alive():
            QApplication.processEvents()
        
        if exception and raw_module and raw_row is not None:
            raw_module.load()
            self.active_module = raw_module
            self.selected_module_name = raw_module.get_name()
            raw_index = self.__item_model.index(raw_row, 0)
            selection_model = self.win.module_list.selectionModel()
            if selection_model:
                selection_model.setCurrentIndex(
                    raw_index, 
                    QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Current
                )
            exception_window = ExceptionWindow(
                self.win,
                type(exception),
                exception,
                exception.__traceback__
            )
            exception_window.show()
            exception_window.raise_()
            exception_window.activateWindow()
        elif exception:
            raise exception
        self.win.setEnabled(True)
        
        if not self.active_module:
            return
            
        self.win.text_description.clear()
        description_html = self.active_module.get_description()
        themed_html = wrap_html_with_theme(description_html, self.theme_manager.current_theme)
        self.win.text_description.setHtml(themed_html)
        self.win.position_mode_option.blockSignals(True)
        self.win.position_mode_option.clear()
        for mode in self.active_module.get_position_mode_list():
            self.win.position_mode_option.addItem(mode)
        self.win.position_mode_option.setCurrentIndex(0)
        self.on_position_mode_changed(self.win.position_mode_option.currentText())
        self.win.position_mode_option.blockSignals(False)

        self.win.layer_mix_option.blockSignals(True)
        self.win.layer_mix_option.clear()
        for mode in self.active_module.get_layer_mix_mode_list():
            self.win.layer_mix_option.addItem(mode)
        self.win.layer_mix_option.setCurrentIndex(0)
        self.on_layer_mix_option_changed(self.win.layer_mix_option.currentText())
        self.win.layer_mix_option.blockSignals(False)

        self.win.head_mix_option.blockSignals(True)
        self.win.head_mix_option.clear()
        for mode in self.active_module.get_head_mix_mode_list():
            self.win.head_mix_option.addItem(mode)
        self.win.head_mix_option.setCurrentIndex(0)
        self.on_head_mix_option_changed(self.win.head_mix_option.currentText())
        self.win.head_mix_option.blockSignals(False)

        self.win.clear_control_option()
        info = self.active_module.get_other_info()
        if info is None:
            return
        for name, option in info.items():
            def callback(text, name=name):
                if text and self.active_module:
                    self.active_module.set_other_info({name: text})
                    try:
                        self.on_confirm_button_clicked()
                    except RuntimeError:
                        pass
            self.win.add_control_option(name, option, callback)


    
    def on_position_mode_changed(self, text):
        self.position_mode_option = text
        try:
            self.visualize()
        except RuntimeError:
            pass

    def on_layer_mix_option_changed(self, text):
        self.layer_mix_option = text
        try:
            self.visualize()
        except RuntimeError:
            pass

    def on_head_mix_option_changed(self, text):
        self.head_mix_option = text
        try:
            self.visualize()
        except RuntimeError:
            pass

    def on_confirm_button_clicked(self):
        if not self.active_module:
            return
        text = self.win.text_input.text()
        self.active_module.forward(text)
        self.visualize()
    
    def key_changed(self, key: int):
        if not self.active_module or not self.position_mode_option or not self.layer_mix_option or not self.head_mix_option:
            return
        self.win.viz_scroll.show_color(
            self.active_module.get_attention_weights(
                key,
                self.position_mode_option,
                self.layer_mix_option,
                self.head_mix_option,
                self.temperature + 1e-6,
            )
        )
    
    def on_temperature_set_button_clicked(self):
        if not self.active_module:
            return
        text = self.win.temperature_input.text()
        try:
            self.temperature = float(text)
        except ValueError:
            pass
        self.win.temperature_input.setPlaceholderText(str(self.temperature))
        self.win.temperature_input.clear()
        try:
            self.visualize()
        except RuntimeError:
            pass
    
    def on_fontsize_set_button_clicked(self):
        if not self.active_module:
            return
        text = self.win.fontsize_input.text()
        try:
            self.fontsize = int(text)
        except ValueError:
            pass
        self.win.fontsize_input.setPlaceholderText(f"{self.fontsize}pt")
        self.win.fontsize_input.clear()
        self.win.viz_scroll.set_fontsize(self.fontsize)
        try:
            self.visualize()
        except RuntimeError:
            pass
    
    def show_help(self, language):
        help_window = HelpWindow(language, self.win)
        help_window.show()
        help_window.raise_()
        help_window.activateWindow()

    def show_about(self):
        about_window = AboutWindow(self.win)
        about_window.show()
        about_window.raise_()
        about_window.activateWindow()
    
    def apply_theme(self):
        """Apply theme styles"""
        self.app.setStyleSheet(self.theme_manager.get_stylesheet())
        
        if hasattr(self, 'win'):
            self._override_hardcoded_styles()
            
            # Update description HTML with new theme
            if hasattr(self, 'active_module') and self.active_module:
                description_html = self.active_module.get_description()
                themed_html = wrap_html_with_theme(description_html, self.theme_manager.current_theme)
                self.win.text_description.setHtml(themed_html)
        
        if hasattr(self, 'win') and hasattr(self.win, 'viz_scroll'):
            self.win.viz_scroll.set_theme_colors(
                self.theme_manager.get_attention_colors(),
                self.theme_manager.get_key_query_colors()
            )
            
        if hasattr(self, 'active_module') and self.active_module:
            try:
                self.visualize()
            except RuntimeError:
                pass
    
    def _override_hardcoded_styles(self):
        """Override UI file hardcoded styles"""
        surface_color = self.theme_manager.get_color('surface')
        background_color = self.theme_manager.get_color('background')
        border_color = self.theme_manager.get_color('border')
        
        self.win.frame_model_menu.setStyleSheet(f"""
            QFrame {{
                background-color: {surface_color};
                border: 1px solid {border_color};
            }}
        """)
        
        self.win.frame_content.setStyleSheet(f"""
            QFrame {{
                background-color: {background_color};
                border: 1px solid {border_color};
            }}
        """)
        
        if hasattr(self.win, 'frame_detail'):
            self.win.frame_detail.setStyleSheet(f"""
                QFrame {{
                    background-color: {background_color};
                    border: 1px solid {border_color};
                }}
            """)
        
        if hasattr(self.win, 'frame_description'):
            self.win.frame_description.setStyleSheet(f"""
                QFrame {{
                    background-color: {surface_color};
                    border: 1px solid {border_color};
                }}
            """)
        
        if hasattr(self.win, 'frame_control'):
            self.win.frame_control.setStyleSheet(f"""
                QFrame {{
                    background-color: {surface_color};
                    border: 1px solid {border_color};
                }}
            """)
        
        if hasattr(self.win, 'frame_input'):
            self.win.frame_input.setStyleSheet(f"""
                QFrame {{
                    background-color: {surface_color};
                    border: 1px solid {border_color};
                }}
            """)
        
        if hasattr(self.win, 'frame_attention'):
            self.win.frame_attention.setStyleSheet(f"""
                QFrame {{
                    background-color: {surface_color};
                    border: 1px solid {border_color};
                }}
            """)
        
        for frame in self.win.more_option_frame:
            frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {surface_color};
                    border: 1px solid {border_color};
                }}
            """)
    
    def refresh_theme(self):
        """Refresh theme when system theme changes"""
        self.theme_manager = ThemeManager()
        self.apply_theme()