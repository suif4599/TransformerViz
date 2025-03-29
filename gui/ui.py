from PyQt5.QtWidgets import QMainWindow, QApplication, QListView, QTextBrowser, \
    QComboBox, QLineEdit, QPushButton, QVBoxLayout, QAction, QStyle
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QFont, QColor, QValidator
from PyQt5.QtCore import Qt
from .Ui_root import Ui_MainWindow
from core import AbstractModule
from .viz_frame import VizFrameScroll
from .validators import FloatValidator, PositiveIntValidator
from .sub_window import HelpWindow, AboutWindow

class _Root(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.showMaximized()
        self.setWindowTitle("Attention Visualization")
        self.module_list.setSelectionMode(QListView.SingleSelection)

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
        self.help_action: QAction
        self.about_action: QAction

        self.viz_scroll = VizFrameScroll(self.frame_attention, 1)
        self.verticalLayout_6.addWidget(self.viz_scroll)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)

        self.temperature_input.setValidator(FloatValidator())
        self.temperature_input.setPlaceholderText("1.0")

        self.fontsize_input.setValidator(PositiveIntValidator())
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        return super().keyPressEvent(event)
    

class Root:
    def __init__(self, argv: list = []):
        self.app = QApplication(argv)
        self.win = _Root()
        self.modules: dict[str, AbstractModule] = {}
        
        self.selected_module_name: str | None = None
        self.active_module: AbstractModule | None = None
        self.__item_model = QStandardItemModel()
        self.win.module_list.setModel(self.__item_model)
        self.win.module_list.selectionModel().currentChanged.connect(self.on_module_selected)

        self.position_mode_option = None
        self.win.position_mode_option.currentTextChanged.connect(self.on_position_mode_changed)

        self.layer_mix_option = None
        self.win.layer_mix_option.currentTextChanged.connect(self.on_layer_mix_option_changed)

        self.head_mix_option = None
        self.win.head_mix_option.currentTextChanged.connect(self.on_head_mix_option_changed)

        self.win.confirm_button.clicked.connect(self.on_confirm_button_clicked)
        
        self.temperature = 1.0
        self.win.temperature_set_button.clicked.connect(self.on_temperature_set_button_clicked)

        self.fontsize = 20
        self.win.fontsize_set_button.clicked.connect(self.on_fontsize_set_button_clicked)
        self.win.fontsize_input.setPlaceholderText(f"{self.fontsize}pt")
        self.win.viz_scroll.set_fontsize(self.fontsize)

        self.win.help_action.triggered.connect(self.show_help)
        self.win.about_action.triggered.connect(self.show_about)
        self.win.setWindowIcon(self.app.style().standardIcon(QStyle.SP_FileIcon))

    def mainloop(self):
        self.win.show()
        self.app.exec_()
    
    def add_module(self, module: AbstractModule):
        self.modules[name := module.get_name()] = module
        item = QStandardItem(name)
        item.setEditable(False)
        item.setFont(QFont("Arial", 18))
        item.setTextAlignment(Qt.AlignCenter)
        self.__item_model.appendRow(item)
        if len(self.modules) == 1:
            self.win.module_list.setCurrentIndex(self.__item_model.index(0, 0))
        
    def visualize(self):
        sentence = self.active_module.get_input()
        self.win.viz_scroll.reset(
            self.active_module.get_n_head(
                self.position_mode_option,
                self.layer_mix_option,
                self.head_mix_option
            ), 
            self.key_changed
        )
        self.win.viz_scroll.show_sentence(sentence)
        self.win.viz_scroll.set_fontsize(self.fontsize)

    def on_module_selected(self, current, previous):
        # Handle the module selection change
        if current.data() == self.selected_module_name:
            return
        print(f"Selected module: {current.data()}")
        if self.active_module:
            self.active_module.unload()
        self.selected_module_name = current.data()
        self.active_module = self.modules[self.selected_module_name]

        self.win.text_description.clear()
        self.win.text_description.setHtml(self.active_module.get_description())

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

        self.active_module.load()
    
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
        self.win.viz_scroll.show_color(
            self.active_module.get_attention_weights(
                key,
                self.position_mode_option,
                self.layer_mix_option,
                self.head_mix_option,
                self.temperature
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
        self.visualize()
    
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
        self.visualize()
    
    def show_help(self):
        help_window = HelpWindow(self.win)
        help_window.show()
        help_window.raise_()
        help_window.activateWindow()

    def show_about(self):
        about_window = AboutWindow(self.win)
        about_window.show()
        about_window.raise_()
        about_window.activateWindow()