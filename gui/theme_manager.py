from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QTimer
from PyQt6.QtGui import QPalette, QColor
from enum import Enum
from typing import Optional

class ThemeMode(Enum):
    AUTO = "auto"
    LIGHT = "light"
    DARK = "dark"

class ThemeManager(QObject):
    # Signal emitted when theme changes
    theme_changed = pyqtSignal()
    # Light theme colors (Material Design)
    LIGHT_THEME = {
        'background': '#FFFFFF',
        'surface': '#F5F5F5',
        'primary': '#2196F3',
        'primary_variant': '#1976D2',
        'secondary': '#03DAC6',
        'error': '#B00020',
        'on_background': '#000000',
        'on_surface': '#212121',
        'on_primary': '#FFFFFF',
        'on_secondary': '#000000',
        'on_error': '#FFFFFF',
        'text_primary': '#212121',
        'text_secondary': '#757575',
        'divider': '#E0E0E0',
        'border': '#E0E0E0',
        'hover': '#F0F0F0',
        'pressed': '#E0E0E0',
        'disabled': '#BDBDBD',
        'disabled_text': '#9E9E9E'
    }
    
    # Dark theme colors (Material Design)
    DARK_THEME = {
        'background': '#121212',
        'surface': '#1E1E1E',
        'primary': '#4A90E2',         # Changed from purple to blue
        'primary_variant': '#357ABD', # Changed from purple to darker blue
        'secondary': '#03DAC6',
        'error': '#CF6679',
        'on_background': '#FFFFFF',
        'on_surface': '#FFFFFF',
        'on_primary': '#FFFFFF',      # Changed to white for better contrast
        'on_secondary': '#000000',
        'on_error': '#000000',
        'text_primary': '#FFFFFF',
        'text_secondary': '#B3B3B3',
        'divider': '#2C2C2C',
        'border': '#404040',
        'hover': '#2C2C2C',
        'pressed': '#404040',
        'disabled': '#404040',
        'disabled_text': '#6C6C6C'
    }
    
    def __init__(self):
        super().__init__()
        self.theme_mode = ThemeMode.AUTO
        self.is_dark_mode = self._detect_system_theme()
        self.current_theme = self.DARK_THEME if self.is_dark_mode else self.LIGHT_THEME
        
        # Timer to periodically check system theme changes
        self.theme_check_timer = QTimer()
        self.theme_check_timer.timeout.connect(self._check_system_theme_change)
        self.theme_check_timer.start(1000)  # Check every 1 second
    
    def set_theme_mode(self, mode: ThemeMode):
        """Set theme mode (auto, light, or dark)"""
        self.theme_mode = mode
        self._update_theme()
    
    def get_theme_mode(self) -> ThemeMode:
        """Get current theme mode"""
        return self.theme_mode
    
    def _update_theme(self):
        """Update current theme based on mode"""
        old_is_dark = self.is_dark_mode
        
        if self.theme_mode == ThemeMode.AUTO:
            self.is_dark_mode = self._detect_system_theme()
        elif self.theme_mode == ThemeMode.DARK:
            self.is_dark_mode = True
        else:  # LIGHT
            self.is_dark_mode = False
        
        self.current_theme = self.DARK_THEME if self.is_dark_mode else self.LIGHT_THEME
        
        # Emit signal if theme actually changed
        if old_is_dark != self.is_dark_mode:
            self.theme_changed.emit()
    
    def _check_system_theme_change(self):
        """Check if system theme has changed (only when in auto mode)"""
        if self.theme_mode == ThemeMode.AUTO:
            old_is_dark = self.is_dark_mode
            self.is_dark_mode = self._detect_system_theme()
            self.current_theme = self.DARK_THEME if self.is_dark_mode else self.LIGHT_THEME
            
            # Emit signal if theme changed
            if old_is_dark != self.is_dark_mode:
                self.theme_changed.emit()
    
    def _detect_system_theme(self) -> bool:
        """Detect if system uses dark theme"""
        app = QApplication.instance()
        if app and isinstance(app, QApplication):
            palette = app.palette()
            bg_color = palette.color(QPalette.ColorRole.Window)
            brightness = (0.299 * bg_color.red() + 0.587 * bg_color.green() + 0.114 * bg_color.blue())
            return brightness < 128
        return False
    
    def get_color(self, color_name: str) -> str:
        return self.current_theme.get(color_name, '#000000')
    
    def get_stylesheet(self) -> str:
        if self.is_dark_mode:
            return self._get_dark_stylesheet()
        else:
            return self._get_light_stylesheet()
    
    def _get_dark_stylesheet(self) -> str:
        return f"""
        QMainWindow {{
            background-color: {self.get_color('background')};
            color: {self.get_color('text_primary')};
        }}
        
        QFrame {{
            background-color: {self.get_color('surface')};
            border: 1px solid {self.get_color('border')};
            border-radius: 4px;
            color: {self.get_color('text_primary')};
        }}
        
        QListView {{
            background-color: {self.get_color('surface')};
            border: 1px solid {self.get_color('border')};
            border-radius: 4px;
            color: {self.get_color('text_primary')};
            selection-background-color: {self.get_color('primary')};
            selection-color: {self.get_color('on_primary')};
        }}
        
        QListView::item {{
            padding: 8px;
            border-bottom: 1px solid {self.get_color('divider')};
        }}
        
        QListView::item:hover {{
            background-color: {self.get_color('hover')};
        }}
        
        QListView::item:selected {{
            background-color: {self.get_color('primary')};
            color: {self.get_color('on_primary')};
        }}
        
        QTextBrowser {{
            background-color: {self.get_color('surface')};
            border: 1px solid {self.get_color('border')};
            border-radius: 4px;
            color: {self.get_color('text_primary')};
            padding: 8px;
        }}
        
        QLineEdit {{
            background-color: {self.get_color('surface')};
            border: 2px solid {self.get_color('border')};
            border-radius: 4px;
            color: {self.get_color('text_primary')};
            padding: 6px;
            font-size: 14px;
        }}
        
        QLineEdit:focus {{
            border-color: {self.get_color('primary')};
        }}
        
        QLineEdit:disabled {{
            background-color: {self.get_color('disabled')};
            color: {self.get_color('disabled_text')};
        }}
        
        QPushButton {{
            background-color: {self.get_color('primary')};
            border: none;
            border-radius: 4px;
            color: {self.get_color('on_primary')};
            font-weight: bold;
            padding: 8px 16px;
            font-size: 14px;
        }}
        
        QPushButton:hover {{
            background-color: {self.get_color('primary_variant')};
        }}
        
        QPushButton:pressed {{
            background-color: {self.get_color('pressed')};
        }}
        
        QPushButton:disabled {{
            background-color: {self.get_color('disabled')};
            color: {self.get_color('disabled_text')};
        }}
        
        QComboBox {{
            background-color: {self.get_color('surface')};
            border: 2px solid {self.get_color('border')};
            border-radius: 4px;
            color: {self.get_color('text_primary')};
            padding: 6px;
            font-size: 14px;
        }}
        
        QComboBox:focus {{
            border-color: {self.get_color('primary')};
        }}
        
        QComboBox::drop-down {{
            border: none;
            background: transparent;
            width: 20px;
        }}
        
        QComboBox::down-arrow {{
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid {self.get_color('text_primary')};
            margin-right: 6px;
        }}
        
        QComboBox QAbstractItemView {{
            background-color: {self.get_color('surface')};
            border: 1px solid {self.get_color('border')};
            border-radius: 4px;
            color: {self.get_color('text_primary')};
            selection-background-color: {self.get_color('primary')};
            selection-color: {self.get_color('on_primary')};
        }}
        
        QLabel {{
            color: {self.get_color('text_primary')};
            background: transparent;
        }}
        
        QScrollArea {{
            background-color: {self.get_color('background')};
            border: 1px solid {self.get_color('border')};
            border-radius: 4px;
        }}
        
        QScrollBar:vertical {{
            background-color: {self.get_color('surface')};
            width: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {self.get_color('primary')};
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {self.get_color('primary_variant')};
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
        }}
        
        QMenuBar {{
            background-color: {self.get_color('surface')};
            color: {self.get_color('text_primary')};
            border-bottom: 1px solid {self.get_color('border')};
        }}
        
        QMenuBar::item {{
            background: transparent;
            padding: 4px 8px;
        }}
        
        QMenuBar::item:selected {{
            background-color: {self.get_color('hover')};
        }}
        
        QMenu {{
            background-color: {self.get_color('surface')};
            color: {self.get_color('text_primary')};
            border: 1px solid {self.get_color('border')};
            border-radius: 4px;
        }}
        
        QMenu::item {{
            padding: 6px 20px;
        }}
        
        QMenu::item:selected {{
            background-color: {self.get_color('primary')};
            color: {self.get_color('on_primary')};
        }}
        
        QScrollBar:horizontal {{
            background-color: {self.get_color('surface')};
            height: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:horizontal {{
            background-color: {self.get_color('primary')};
            border-radius: 6px;
            min-width: 20px;
        }}
        
        QScrollBar::handle:horizontal:hover {{
            background-color: {self.get_color('primary_variant')};
        }}
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            border: none;
            background: none;
        }}
        
        QDialog {{
            background-color: {self.get_color('background')};
            color: {self.get_color('text_primary')};
        }}
        """
    
    def _get_light_stylesheet(self) -> str:
        return f"""
        QMainWindow {{
            background-color: {self.get_color('background')};
            color: {self.get_color('text_primary')};
        }}
        
        QFrame {{
            background-color: {self.get_color('surface')};
            border: 1px solid {self.get_color('border')};
            border-radius: 4px;
            color: {self.get_color('text_primary')};
        }}
        
        QListView {{
            background-color: {self.get_color('background')};
            border: 1px solid {self.get_color('border')};
            border-radius: 4px;
            color: {self.get_color('text_primary')};
            selection-background-color: {self.get_color('primary')};
            selection-color: {self.get_color('on_primary')};
        }}
        
        QListView::item {{
            padding: 8px;
            border-bottom: 1px solid {self.get_color('divider')};
        }}
        
        QListView::item:hover {{
            background-color: {self.get_color('hover')};
        }}
        
        QListView::item:selected {{
            background-color: {self.get_color('primary')};
            color: {self.get_color('on_primary')};
        }}
        
        QTextBrowser {{
            background-color: {self.get_color('background')};
            border: 1px solid {self.get_color('border')};
            border-radius: 4px;
            color: {self.get_color('text_primary')};
            padding: 8px;
        }}
        
        QLineEdit {{
            background-color: {self.get_color('background')};
            border: 2px solid {self.get_color('border')};
            border-radius: 4px;
            color: {self.get_color('text_primary')};
            padding: 6px;
            font-size: 14px;
        }}
        
        QLineEdit:focus {{
            border-color: {self.get_color('primary')};
        }}
        
        QLineEdit:disabled {{
            background-color: {self.get_color('disabled')};
            color: {self.get_color('disabled_text')};
        }}
        
        QPushButton {{
            background-color: {self.get_color('primary')};
            border: none;
            border-radius: 4px;
            color: {self.get_color('on_primary')};
            font-weight: bold;
            padding: 8px 16px;
            font-size: 14px;
        }}
        
        QPushButton:hover {{
            background-color: {self.get_color('primary_variant')};
        }}
        
        QPushButton:pressed {{
            background-color: {self.get_color('pressed')};
        }}
        
        QPushButton:disabled {{
            background-color: {self.get_color('disabled')};
            color: {self.get_color('disabled_text')};
        }}
        
        QComboBox {{
            background-color: {self.get_color('background')};
            border: 2px solid {self.get_color('border')};
            border-radius: 4px;
            color: {self.get_color('text_primary')};
            padding: 6px;
            font-size: 14px;
        }}
        
        QComboBox:focus {{
            border-color: {self.get_color('primary')};
        }}
        
        QComboBox::drop-down {{
            border: none;
            background: transparent;
            width: 20px;
        }}
        
        QComboBox::down-arrow {{
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid {self.get_color('text_primary')};
            margin-right: 6px;
        }}
        
        QComboBox QAbstractItemView {{
            background-color: {self.get_color('background')};
            border: 1px solid {self.get_color('border')};
            border-radius: 4px;
            color: {self.get_color('text_primary')};
            selection-background-color: {self.get_color('primary')};
            selection-color: {self.get_color('on_primary')};
        }}
        
        QLabel {{
            color: {self.get_color('text_primary')};
            background: transparent;
        }}
        
        QScrollArea {{
            background-color: {self.get_color('background')};
            border: 1px solid {self.get_color('border')};
            border-radius: 4px;
        }}
        
        QScrollBar:vertical {{
            background-color: {self.get_color('surface')};
            width: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {self.get_color('primary')};
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {self.get_color('primary_variant')};
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
        }}
        
        QMenuBar {{
            background-color: {self.get_color('surface')};
            color: {self.get_color('text_primary')};
            border-bottom: 1px solid {self.get_color('border')};
        }}
        
        QMenuBar::item {{
            background: transparent;
            padding: 4px 8px;
        }}
        
        QMenuBar::item:selected {{
            background-color: {self.get_color('hover')};
        }}
        
        QMenu {{
            background-color: {self.get_color('background')};
            color: {self.get_color('text_primary')};
            border: 1px solid {self.get_color('border')};
            border-radius: 4px;
        }}
        
        QMenu::item {{
            padding: 6px 20px;
        }}
        
        QMenu::item:selected {{
            background-color: {self.get_color('primary')};
            color: {self.get_color('on_primary')};
        }}
        
        QScrollBar:horizontal {{
            background-color: {self.get_color('surface')};
            height: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:horizontal {{
            background-color: {self.get_color('primary')};
            border-radius: 6px;
            min-width: 20px;
        }}
        
        QScrollBar::handle:horizontal:hover {{
            background-color: {self.get_color('primary_variant')};
        }}
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            border: none;
            background: none;
        }}
        
        QDialog {{
            background-color: {self.get_color('background')};
            color: {self.get_color('text_primary')};
        }}
        """
    
    def get_attention_colors(self) -> dict:
        """Get attention visualization colors - simplified for manual calculation"""
        return {
            'text_primary': self.get_color('text_primary'),  # For theme detection
            'text_normal': self.get_color('text_primary'),   # Default text color
            'border': self.get_color('border'),
            'background': self.get_color('background')
        }

    def get_key_query_colors(self) -> dict:
        """Get blue colors for Key and Query visualization - simplified for manual calculation"""
        return {
            'text_primary': self.get_color('text_primary'),  # For theme detection
            'text_normal': self.get_color('text_primary'),   # Default text color
            'border': self.get_color('border'),
            'background': self.get_color('background')
        }
