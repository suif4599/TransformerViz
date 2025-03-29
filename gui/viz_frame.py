from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, \
    QScrollArea, QWidget, QSizePolicy
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtCore import Qt, pyqtSignal
from collections.abc import Callable
import re

class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)
    
    def setStyleSheet(self, styleSheet):
        self.__style_sheet = styleSheet
        return super().setStyleSheet(styleSheet)
    
    def getStyleSheet(self):
        return self.__style_sheet

class VizFrame(QFrame):
    __NUM = 0
    def get_name(self):
        VizFrame.__NUM += 1
        self.name = f"VizFrame_{VizFrame.__NUM}"
        return self.name

    def __init__(self, parent, title: str, callback: Callable[[str], None] = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setObjectName(self.get_name())
        self.setStyleSheet("border: 0px;")
        self.horizen_layout_outer = QHBoxLayout(self)
        self.horizen_layout_outer.setObjectName(f"{self.name}_horizen_layout_outer")
        self.horizen_layout_outer.setSpacing(0)
        self.horizen_layout_outer.setContentsMargins(0, 0, 0, 0)
        self.frame_inner = QFrame(self)
        self.frame_inner.setObjectName(f"{self.name}_frame_inner")
        self.frame_inner.setFrameShape(QFrame.StyledPanel)
        self.frame_inner.setFrameShadow(QFrame.Raised)
        self.frame_inner.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.horizen_layout_outer.addWidget(self.frame_inner, 10)
        self.horizen_layout = QHBoxLayout(self.frame_inner)
        self.horizen_layout.setObjectName(f"{self.name}_horizen_layout")
        self.horizen_layout.setSpacing(12)
        self.horizen_layout.setContentsMargins(0, 0, 0, 0)
        self.label_list: list[ClickableLabel] = []
        self.title = title
        self.title_label: ClickableLabel | None = None
        self.callback = callback
        self.fontsize = 20
    
    def clear(self):
        if self.title_label:
            self.horizen_layout_outer.removeWidget(self.title_label)
            self.title_label.setParent(None)
            self.title_label.deleteLater()
            self.title_label = None
        for label in self.label_list:
            self.horizen_layout.removeWidget(label)
            label.setParent(None)
            label.deleteLater()
        self.label_list.clear()
    
    def show_sencence(self, sentence: list[str]):
        self.clear()
        self.title_label = ClickableLabel(self.title, self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setObjectName(f"{self.name}_title_label")
        self.title_label.setStyleSheet("font-size: %dpt; font-weight: bold; color: yellow;" % self.fontsize)
        self.horizen_layout_outer.insertWidget(0, self.title_label, 1, Qt.AlignLeft)
        for word in sentence:
            label = ClickableLabel(word, self)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-size: %dpt; font-weight: bold; color: black;" % self.fontsize)
            self.label_list.append(label)
            self.horizen_layout.addWidget(label)
            if self.callback:
                def callback(*args, word=word):
                    for label in self.label_list:
                        if label.text() == word:
                            label.setStyleSheet("font-size: %dpt; "
                                                "font-weight: bold; "
                                                "color: white; "
                                                "background-color: rgb(200, 20, 20);" % self.fontsize)
                        else:
                            label.setStyleSheet("font-size: %dpt; "
                                                "font-weight: bold; "
                                                "color: black;" % self.fontsize)
                    self.callback(word)
                label.clicked.connect(callback)
    
    def set_color(self, color: list[float]):
        if len(color) != len(self.label_list):
            raise ValueError("Color list length does not match the number of labels.")
        for label, color_label in zip(self.label_list, color):
            q_color = (int(color_label * 255), int(color_label * 20), int(color_label * 20))
            label.setStyleSheet("font-size: %dpt; "
                                "font-weight: bold; "
                                "color: white; "
                                "background-color: rgb(%d, %d, %d);" % (self.fontsize, *q_color))
    
    def set_fontsize(self, fontsize: int):
        self.fontsize = fontsize
        if self.title_label:
            self.title_label.setStyleSheet(
                re.sub(r"font-size: \d+pt", f"font-size: {fontsize}pt", self.title_label.getStyleSheet())
            )
        for label in self.label_list:
            label.setStyleSheet(
                re.sub(r"font-size: \d+pt", f"font-size: {fontsize}pt", label.getStyleSheet())
            )
        self.setFixedHeight(int(QFontMetrics(self.title_label.font()).height() * 1.8))

class VizFrameScroll(QWidget):
    __NUM = 0
    def get_name(self):
        VizFrameScroll.__NUM += 1
        self.name = f"VizFrameScroll_{VizFrameScroll.__NUM}"
        return self.name

    def __init__(self, parent, n_head: int, frame_height: int = 160):
        self.frame_height = frame_height
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setObjectName(self.get_name())
        self.outer_layout = QVBoxLayout(self)
        self.outer_layout.setObjectName(f"{self.name}_outer_layout")
        self.outer_layout.setSpacing(0)
        self.outer_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area = QScrollArea(self)
        self.outer_layout.addWidget(self.scroll_area)
        self.scroll_area.setObjectName(f"{self.name}_scroll_area")
        self.scroll_area.setWidgetResizable(True)
        self.container = QWidget(self.scroll_area)
        self.container.setObjectName(f"{self.name}_container")
        self.container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.container.setStyleSheet("background-color: rgb(188, 188, 188);")
        self.scroll_area.setWidget(self.container)
        self.v_layout = QVBoxLayout(self.container)
        self.v_layout.setObjectName(f"{self.name}_v_layout")
        self.v_layout.setSpacing(3)
        self.v_layout.setContentsMargins(0, 0, 0, 0)

        self.viz_frame_list: list[VizFrame] = []
    
    def clear(self):
        for frame in self.viz_frame_list:
            frame.clear()
    
    def add_frame(self, name: str, callback: Callable[[str], None] = None) -> VizFrame:
        frame = VizFrame(self.parent(), name, callback)
        frame.setFixedHeight(self.frame_height)
        self.viz_frame_list.append(frame)
        self.v_layout.addWidget(frame)
        return frame
    
    def show_sentence(self, sentence: list[str]):
        self.clear()
        for frame in self.viz_frame_list:
            frame.show_sencence(sentence)
    
    def reset(self, n_head: int, callback: Callable[[str], None]):
        self.clear()
        if n_head < 1:
            raise ValueError("Number of heads must be greater than 0.")
        if n_head == len(self.viz_frame_list):
            return
        for frame in self.viz_frame_list:
            self.v_layout.removeWidget(frame)
            frame.setParent(None)
            frame.deleteLater()
        self.viz_frame_list.clear()
        self.add_frame("Key", callback)
        for i in range(1, n_head + 1):
            self.add_frame(f"Query_{i}")
    
    def show_color(self, color: list[float | list[float]]):
        if len(color) == 0:
            raise ValueError("Color list is empty.")
        if isinstance(color[0], list):
            # multiple colors
            if len(color) != len(self.viz_frame_list) - 1:
                raise ValueError("Color list length does not match the number of frames.")
            for color_each, frame in zip(color, self.viz_frame_list[1:]):
                frame.set_color(color_each)
        else:
            if len(self.viz_frame_list) - 1 != 1:
                raise ValueError("Color list length does not match the number of frames.")
            self.viz_frame_list[1].set_color(color)
    
    def set_fontsize(self, fontsize: int):
        for frame in self.viz_frame_list:
            frame.set_fontsize(fontsize)
        
