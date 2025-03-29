from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, \
    QScrollArea, QWidget, QSizePolicy
from PyQt5.QtCore import Qt

class VizFrame(QFrame):
    __NUM = 0
    def get_name(self):
        VizFrame.__NUM += 1
        self.name = f"VizFrame_{VizFrame.__NUM}"
        return self.name

    def __init__(self, parent):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setObjectName(self.get_name())
        self.horizen_layout = QHBoxLayout(self)
        self.horizen_layout.setObjectName(f"{self.name}_horizen_layout")
        self.label_list: list[QLabel] = []
    
    def clear(self):
        for label in self.label_list:
            self.horizen_layout.removeWidget(label)
            label.setParent(None)
            label.deleteLater()
        self.label_list.clear()
    
    def show_sencence(self, sentence: list[str]):
        self.clear()
        for word in sentence:
            label = QLabel(word, self)
            label.setAlignment(Qt.AlignCenter)
            self.label_list.append(label)
            self.horizen_layout.addWidget(label)

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
        self.scroll_area.setWidget(self.container)
        self.v_layout = QVBoxLayout(self.container)
        self.v_layout.setObjectName(f"{self.name}_v_layout")
        self.v_layout.setSpacing(3)
        self.v_layout.setContentsMargins(0, 0, 0, 0)

        self.viz_frame_list: list[VizFrame] = []
        for _ in range(n_head):
            self.add_frame()
    
    def clear(self):
        for frame in self.viz_frame_list:
            frame.clear()
    
    def add_frame(self):
        frame = VizFrame(self.parent())
        frame.setFixedHeight(self.frame_height)
        self.viz_frame_list.append(frame)
        self.v_layout.addWidget(frame)
        return frame
    
    def show_sentence(self, sentence: list[str]):
        self.clear()
        for frame in self.viz_frame_list:
            frame.show_sencence(sentence)
    
    def reset(self, n_head: int):
        self.clear()
        if n_head == len(self.viz_frame_list):
            return
        for frame in self.viz_frame_list:
            self.v_layout.removeWidget(frame)
            frame.setParent(None)
            frame.deleteLater()
        self.viz_frame_list.clear()
        for _ in range(n_head):
            self.add_frame()