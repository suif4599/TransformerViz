from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, \
    QScrollArea, QWidget, QSizePolicy
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtCore import Qt, pyqtSignal
from collections.abc import Callable
from typing import Optional
import re

class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, ev):
        event = ev
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

    def __init__(self, parent, title: str, callback: Optional[Callable[[int], None]] = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setObjectName(self.get_name())
        self.setStyleSheet("border: 0px;")
        self.horizen_layout_outer = QHBoxLayout(self)
        self.horizen_layout_outer.setObjectName(f"{self.name}_horizen_layout_outer")
        self.horizen_layout_outer.setSpacing(0)
        self.horizen_layout_outer.setContentsMargins(0, 0, 0, 0)
        self.frame_inner = QFrame(self)
        self.frame_inner.setObjectName(f"{self.name}_frame_inner")
        self.frame_inner.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_inner.setFrameShadow(QFrame.Shadow.Raised)
        self.frame_inner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
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
        self._current_colors: list[float] = []  # Store current color values for theme updates
        
        # Theme colors will be set by set_theme_colors method
        self.theme_colors = {
            'text_primary': '#212121',
            'text_normal': '#212121',
            'border': '#E0E0E0',
            'background': '#FFFFFF'
        }
    
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
    
    def show_sencence(self, sentence: list[str], max_index: int = 0):
        self.clear()
        self.title_label = ClickableLabel(self.title, self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setObjectName(f"{self.name}_title_label")
        # Initial styling will be updated by set_theme_colors
        self.title_label.setStyleSheet(f"""
            font-size: {self.fontsize}pt; 
            font-weight: bold; 
            color: #FFFFFF;
            background-color: #2196F3;
            border-radius: 3px;
            padding: 1px;
            margin: 1px;
        """)
        # 设置title label的最小高度
        font_metrics = QFontMetrics(self.title_label.font())
        min_height = int(font_metrics.height() * 1.3)
        self.title_label.setMinimumHeight(min_height)
        self.horizen_layout_outer.insertWidget(0, self.title_label, 1, Qt.AlignmentFlag.AlignLeft)
        for index, word in enumerate(sentence):
            label = ClickableLabel(word, self)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(f"""
                font-size: {self.fontsize}pt; 
                font-weight: bold; 
                color: {self.theme_colors.get('text_normal', '#212121')};
                background-color: {self.theme_colors.get('background', '#FFFFFF')};
                border-radius: 3px;
                padding: 1px;
                margin: 1px;
            """)
            # 设置label的最小高度以确保文字有足够空间显示
            font_metrics = QFontMetrics(label.font())
            min_height = int(font_metrics.height() * 1.3)
            self.label_list.append(label)
            self.horizen_layout.addWidget(label)
            if self.callback and index < max_index:
                def callback(*args, index=index):
                    for ind, label in enumerate(self.label_list):
                        if ind == index:
                            label.setStyleSheet(f"""
                                font-size: {self.fontsize}pt;
                                font-weight: bold;
                                color: {self.theme_colors.get('text_highlighted', '#FFFFFF')};
                                background-color: {self.theme_colors.get('max_attention', '#FF5722')};
                                border-radius: 3px;
                                padding: 1px;
                                margin: 1px;
                            """)
                        else:
                            label.setStyleSheet(f"""
                                font-size: {self.fontsize}pt;
                                font-weight: bold;
                                color: {self.theme_colors.get('text_normal', '#212121')};
                                background-color: {self.theme_colors.get('background', '#FFFFFF')};
                                border-radius: 3px;
                                padding: 1px;
                                margin: 1px;
                            """)
                    if self.callback:
                        self.callback(index)
                label.clicked.connect(callback)
    
    def set_theme_colors(self, theme_colors: dict):
        """Set theme colors and apply to frame styles"""
        old_theme_colors = self.theme_colors
        self.theme_colors = theme_colors
        
        frame_bg = theme_colors.get('background', '#F5F5F5')
        border_color = theme_colors.get('border', '#E0E0E0')
        
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {frame_bg};
                border: 0px solid {border_color};
            }}
        """)
        
        self.frame_inner.setStyleSheet(f"""
            QFrame {{
                background-color: {frame_bg};
                border: 0px solid {border_color};
                border-radius: 3px;
            }}
        """)
        
        # Update title label colors if it exists
        if self.title_label:
            # Determine if this is Key/Query frame and calculate appropriate title colors
            is_key_query_frame = self.title and ('Key' in self.title or 'Query' in self.title)
            text_primary = theme_colors.get('text_primary', '#212121')
            is_dark_mode = text_primary.lower() in ['#ffffff', '#fff', 'white'] or text_primary.startswith('#f') or text_primary.startswith('#e')
            
            if is_key_query_frame:
                # Blue title for Key/Query frames
                title_bg = '#2196F3' if not is_dark_mode else '#1976D2'
            else:
                # Red title for regular frames
                title_bg = '#F44336' if not is_dark_mode else '#D32F2F'
            
            title_text = '#FFFFFF'  # Always white text on colored backgrounds
            
            self.title_label.setStyleSheet(f"""
                font-size: {self.fontsize}pt; 
                font-weight: bold; 
                color: {title_text};
                background-color: {title_bg};
                padding: 1px;
                margin: 1px;
            """)
        
        # Force re-render all labels if they exist and theme actually changed
        if old_theme_colors != theme_colors and hasattr(self, '_current_colors') and self._current_colors:
            self.set_color(self._current_colors)
    
    def _get_attention_color(self, attention_value: float) -> tuple[str, str]:
        """Get background and text color based on attention value using smooth gradient"""
        attention_value = max(0.0, min(1.0, attention_value))
        text_primary = self.theme_colors.get('text_primary', '#212121')
        is_dark_mode = text_primary.lower() in ['#ffffff', '#fff', 'white'] or text_primary.startswith('#f') or text_primary.startswith('#e')
        if is_dark_mode:
            red_component = int(255 * attention_value)
            background_color = f"#{red_component:02x}0000"
            text_color = "#FFFFFF"
        else:
            green_blue_component = int(255 * (1 - attention_value))
            background_color = f"#ff{green_blue_component:02x}{green_blue_component:02x}"
            text_color = "#000000"
            
        return background_color, text_color
    
    def set_color(self, color: list[float]):
        self._current_colors = color.copy() if color else []
        
        if len(color) > len(self.label_list):
            raise ValueError("Color list length does not match the number of labels.")
        extended_color: list[float | None] = list(color)
        if len(color) < len(self.label_list):
            extended_color = extended_color + [None] * (len(self.label_list) - len(color))
        for label, color_label in zip(self.label_list, extended_color):
            if color_label is None:
                text_color = self.theme_colors.get('text_normal', self.theme_colors.get('text_primary', '#212121'))
                label.setStyleSheet(f"""
                    font-size: {self.fontsize}pt;
                    font-weight: bold;
                    color: {text_color};
                    background-color: {self.theme_colors['background']};
                    padding: 1px;
                    margin: 1px;
                """)
            else:
                bg_color, text_color = self._get_attention_color(color_label)
                label.setStyleSheet(f"""
                    font-size: {self.fontsize}pt;
                    font-weight: bold;
                    color: {text_color};
                    background-color: {bg_color};
                    padding: 1px;
                    margin: 1px;
                """)
            font_metrics = QFontMetrics(label.font())
            min_height = int(font_metrics.height() * 1.3)
            label.setMinimumHeight(min_height)
    
    def set_fontsize(self, fontsize: int):
        self.fontsize = fontsize
        if self.title_label:
            current_style = self.title_label.styleSheet() or ""
            self.title_label.setStyleSheet(
                re.sub(r"font-size: \d+pt", f"font-size: {fontsize}pt", current_style)
            )
            # 更新title label的最小高度
            font_metrics = QFontMetrics(self.title_label.font())
            min_height = int(font_metrics.height() * 1.3)
            self.title_label.setMinimumHeight(min_height)
        for label in self.label_list:
            current_style = label.styleSheet() or ""
            label.setStyleSheet(
                re.sub(r"font-size: \d+pt", f"font-size: {fontsize}pt", current_style)
            )
            font_metrics = QFontMetrics(label.font())
            min_height = int(font_metrics.height() * 1.3)
            label.setMinimumHeight(min_height)
        if self.title_label:
            font = self.title_label.font()
            if font:
                self.setFixedHeight(int(QFontMetrics(font).height() * 2.0))
    
    def get_width(self, index: int) -> int:
        if len(self.label_list) == 0:
            return 0
        label = self.label_list[index]
        font_metrics = QFontMetrics(label.font())
        text_width = font_metrics.boundingRect(label.text()).width()
        return int(text_width * 1.2 + 10)
    
    def set_width(self, index: int, width: int):
        if len(self.label_list) == 0:
            return
        self.label_list[index].setFixedWidth(width)

class VizFrameScroll(QWidget):
    __NUM = 0
    def get_name(self):
        VizFrameScroll.__NUM += 1
        self.name = f"VizFrameScroll_{VizFrameScroll.__NUM}"
        return self.name

    def __init__(self, parent, n_head: int, frame_height: int = 160):
        self.frame_height = frame_height
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setObjectName(self.get_name())
        self.current_theme_colors = None
        self.key_query_colors = None
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
        self.container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.scroll_area.setWidget(self.container)
        self.v_layout = QVBoxLayout(self.container)
        self.v_layout.setObjectName(f"{self.name}_v_layout")
        self.v_layout.setSpacing(3)
        self.v_layout.setContentsMargins(0, 0, 0, 0)

        self.viz_frame_list: list[VizFrame] = []
    
    def clear(self):
        for frame in self.viz_frame_list:
            frame.clear()
    
    def add_frame(self, name: str, callback: Optional[Callable[[int], None]] = None) -> VizFrame:
        frame = VizFrame(self.parent(), name, callback)
        frame.setFixedHeight(self.frame_height)
        if self.current_theme_colors:
            if name and ('Key' in name or 'Query' in name):
                if hasattr(self, 'key_query_colors') and self.key_query_colors:
                    frame.set_theme_colors(self.key_query_colors)
                else:
                    frame.set_theme_colors(self.current_theme_colors)
            else:
                frame.set_theme_colors(self.current_theme_colors)
        
        self.viz_frame_list.append(frame)
        self.v_layout.addWidget(frame)
        return frame
    
    def show_sentence(self, target: list[str], source: list[str]):
        self.clear()
        target_length = len(target)
        if len(target) < len(source):
            target = target + [""] * (len(source) - len(target))
        elif len(target) > len(source):
            source = source + [""] * (len(target) - len(source))
        if len(target) == 0 or len(source) == 0:
            raise ValueError("Target or source sentence is empty.")
        self.viz_frame_list[0].show_sencence(target, target_length)
        for frame in self.viz_frame_list[1:]:
            frame.show_sencence(source)
        # align the width of all frames
        for i in range(len(target)):
            max_width = max(frame.get_width(i) for frame in self.viz_frame_list)
            for frame in self.viz_frame_list:
                frame.set_width(i, max_width)
    
    def reset(self, n_head: int, callback: Callable[[int], None]):
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
                if isinstance(color_each, list):
                    frame.set_color(color_each)
        else:
            if len(self.viz_frame_list) - 1 != 1:
                raise ValueError("Color list length does not match the number of frames.")
            # color is list[float] in this case
            self.viz_frame_list[1].set_color(color)  # type: ignore
    
    def set_fontsize(self, fontsize: int):
        for frame in self.viz_frame_list:
            frame.set_fontsize(fontsize)
    
    def set_theme_colors(self, attention_colors: dict, key_query_colors: dict | None = None):
        """Apply theme colors to scroll container and all frames"""
        old_attention_colors = getattr(self, 'current_theme_colors', None)
        old_key_query_colors = getattr(self, 'key_query_colors', None)
        
        self.current_theme_colors = attention_colors
        self.key_query_colors = key_query_colors
        
        container_bg = attention_colors.get('background', '#F5F5F5')
        self.container.setStyleSheet(f"background-color: {container_bg};")
        
        for frame in self.viz_frame_list:
            frame_title = getattr(frame, 'title', 'Unknown')
            # Check if this frame is for Key or Query
            if frame.title and ('Key' in frame.title or 'Query' in frame.title):
                # Use blue colors for Key and Query frames
                if key_query_colors:
                    frame.set_theme_colors(key_query_colors)
                else:
                    frame.set_theme_colors(attention_colors)
            else:
                # Use default red colors for other frames
                frame.set_theme_colors(attention_colors)
        
        # Force refresh all frames if theme actually changed
        if (old_attention_colors != attention_colors or old_key_query_colors != key_query_colors):
            for frame in self.viz_frame_list:
                if hasattr(frame, '_current_colors') and frame._current_colors:
                    frame.set_color(frame._current_colors)
        
