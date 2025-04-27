from PyQt5.QtWidgets import QMainWindow, QApplication, QListView, QTextBrowser, \
    QComboBox, QLineEdit, QPushButton, QVBoxLayout, QAction, QStyle, QFrame, \
    QHBoxLayout, QLabel
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QFont
from PyQt5.QtCore import Qt
from .Ui_root import Ui_MainWindow
from core import AbstractModule
from .viz_frame import VizFrameScroll
from .validators import FloatValidator, PositiveIntValidator
from .sub_window import HelpWindow, AboutWindow
from threading import Thread

LOADING = """
<!DOCTYPE html>
<html>
<head>
<style>
.container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
  width: 100%;
}
.loading-text {
  margin-top: 10px;
  text-align: center;
  font-size: 50px;
  color: #666;
}
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

        self.more_option_frame = []
        self.more_option_layout = []
        self.more_option_label = []
        self.more_option_combo = []
        self.more_option_input = []
        self.more_option_button = []


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        return super().keyPressEvent(event)
    
    def add_control_option(self, name: str, option: list[str] | None, callback: callable):
        self.more_option_frame.append(QFrame(self.frame_control))
        self.more_option_frame[-1].setFrameShape(QFrame.StyledPanel)
        self.more_option_frame[-1].setFrameShadow(QFrame.Raised)
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
        self.win.text_input.returnPressed.connect(self.on_confirm_button_clicked)
        
        self.temperature = 1.0
        self.win.temperature_set_button.clicked.connect(self.on_temperature_set_button_clicked)
        self.win.temperature_input.returnPressed.connect(self.on_temperature_set_button_clicked)

        self.fontsize = 20
        self.win.fontsize_set_button.clicked.connect(self.on_fontsize_set_button_clicked)
        self.win.fontsize_input.setPlaceholderText(f"{self.fontsize}pt")
        self.win.fontsize_input.returnPressed.connect(self.on_fontsize_set_button_clicked)
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
        # Handle the module selection change
        def inner():
            if self.active_module:
                self.active_module.unload()
            self.selected_module_name = new.data()
            self.active_module = self.modules[self.selected_module_name]
            self.active_module.load()
        
        self.win.text_description.clear()
        self.win.text_description.setHtml(LOADING)
        loading_thread = Thread(target=inner)
        loading_thread.start()
        self.win.setEnabled(False)
        while loading_thread.is_alive():
            QApplication.processEvents()
        self.win.setEnabled(True)
        
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

        self.win.clear_control_option()
        info = self.active_module.get_other_info()
        if info is None:
            return
        for name, option in info.items():
            def callback(text, name=name):
                if text:
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