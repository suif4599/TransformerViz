from PyQt5.QtWidgets import QMainWindow, QApplication, QListView, QTextBrowser, \
    QComboBox, QLineEdit, QPushButton, QVBoxLayout
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QFont, QColor
from PyQt5.QtCore import Qt
from .Ui_root import Ui_MainWindow
from core import AbstractModule
from .viz_frame import VizFrameScroll

class _Root(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.showFullScreen()
        self.module_list.setSelectionMode(QListView.SingleSelection)

        self.module_list: QListView
        self.text_description: QTextBrowser
        self.layer_mix_option: QComboBox
        self.head_mix_option: QComboBox
        self.text_input: QLineEdit
        self.confirm_button: QPushButton
        self.verticalLayout_6: QVBoxLayout

        self.viz_scroll = VizFrameScroll(self.frame_attention, 12)
        self.verticalLayout_6.addWidget(self.viz_scroll)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)

    
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

        self.layer_mix_option = None
        self.win.layer_mix_option.currentTextChanged.connect(self.on_layer_mix_option_changed)

        self.head_mix_option = None
        self.win.head_mix_option.currentTextChanged.connect(self.on_head_mix_option_changed)

        self.win.confirm_button.clicked.connect(self.on_confirm_button_clicked)
    
    def mainloop(self):
        self.win.show()
        self.app.exec_()
    
    def add_module(self, module: AbstractModule):
        self.modules[name := module.get_name()] = module
        item = QStandardItem(name)
        item.setEditable(False)
        item.setFont(QFont("Arial", 18))
        item.setTextAlignment(Qt.AlignCenter)
        item.setBackground(QColor(188, 188, 188))
        self.__item_model.appendRow(item)
        
    def visualize(self):
        sentence = self.active_module.get_input()
        self.win.viz_scroll.show_sentence(sentence)

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
        self.win.head_mix_option.blockSignals(False)

        self.active_module.load()
    
    def on_layer_mix_option_changed(self, text):
        self.layer_mix_option = text
        print(f"Layer mix option changed to {text}")

    def on_head_mix_option_changed(self, text):
        self.head_mix_option = text
        print(f"Head mix option changed to {text}")

    def on_confirm_button_clicked(self):
        if not self.active_module:
            return
        text = self.win.text_input.text()
        self.active_module.forward(text)
        self.visualize()