from abc import abstractmethod
from typing import Optional, List, Tuple

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QStackedWidget, QToolButton,
    QSplitter, QFrame
)


from cross_platform.dev.icons_legacy.svg_path import get_icon, IconType
from qtcore.meta import QABCMeta

NAV_LIST_STYLESHEET = """
    QListWidget#navigationList {
        background-color: palette(window);
        border: none;
        border-right: 1px solid palette(mid);
        outline: none;
        padding: 0px;
    }

    QListWidget#navigationList::item {
        background-color: transparent;
        border: none;
        border-left: 3px solid transparent;
        border-radius: 0px;
        padding: 12px 16px;
        margin: 0px;
        color: palette(text);
        font-size: 13px;
    }

    QListWidget#navigationList::item:hover {
        background-color: palette(light);
    }

    QListWidget#navigationList::item:selected {
        background-color: palette(highlight);
        color: palette(highlighted-text);
        border-left: 3px solid palette(accent);
        font-weight: bold;
    }

    QListWidget#navigationList::item:selected:hover {
        background-color: palette(highlight);
    }
"""

NAV_LIST_COLLAPSED_STYLESHEET = """
    QListWidget#navigationList {
        background-color: palette(window);
        border: none;
        border-right: 1px solid palette(mid);
        outline: none;
        padding: 0px;
    }

    QListWidget#navigationList::item {
        background-color: transparent;
        border: none;
        border-radius: 4px;
        padding: 8px;
        margin: 4px;
        color: palette(text);
    }

    QListWidget#navigationList::item:hover {
        background-color: palette(light);
    }

    QListWidget#navigationList::item:selected {
        background-color: palette(highlight);
        color: palette(highlighted-text);
    }

    QListWidget#navigationList::item:selected:hover {
        background-color: palette(highlight);
    }
"""

TOGGLE_BUTTON_STYLESHEET = """
    QPushButton#toggleButton {
        background-color: transparent;
        border: none;
        padding: 8px;
        border-radius: 4px;
    }

    QPushButton#toggleButton:hover {
        background-color: palette(light);
    }

    QPushButton#toggleButton:pressed {
        background-color: palette(mid);
    }
"""

MENU_HEADER_STYLESHEET = """
    QWidget#menuHeader {
        background-color: palette(window);
        border-bottom: 1px solid palette(mid);
    }
"""


def create_group_frame(group_name: str,
                       description: Optional[str] = None) -> QFrame:
    """Create a frame for a group of parameters with optional description."""
    group_frame = QFrame()
    group_frame.setObjectName("groupFrame")
    group_frame.setFrameStyle(QFrame.Shape.NoFrame)
    group_frame.setStyleSheet("""
        QFrame#groupFrame {
            background-color: palette(light);
            border: 1px solid palette(border);
            border-radius: 8px;
            margin: 0px;
        }
    """)
    # group_frame.setMinimumWidth(500)

    layout = QVBoxLayout(group_frame)
    layout.setSpacing(0)
    layout.setContentsMargins(0, 0, 0, 0)

    # Create header container
    header_container = QWidget()
    header_container.setStyleSheet("""
        QWidget {
            background-color: palette(base);
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            border-bottom: 1px solid palette(border);
        }
    """)
    header_layout = QVBoxLayout(header_container)
    header_layout.setContentsMargins(16, 12, 16, 12)
    header_layout.setSpacing(4)

    # Add group title
    title = QLabel(group_name)
    title.setStyleSheet("""
        QLabel {
            font-weight: 600;
            font-size: 14px;
            color: #1a365d;
            background-color: transparent;
            border: none;
        }
    """)
    header_layout.addWidget(title)

    # Add description if provided
    if description:
        desc_label = QLabel(description)
        desc_label.setStyleSheet("""
            QLabel {
                font-weight: 400;
                font-size: 12px;
                color: #64748b;
                background-color: transparent;
                border: none;
            }
        """)
        desc_label.setWordWrap(True)
        header_layout.addWidget(desc_label)

    layout.addWidget(header_container)

    return group_frame


def create_parameter_row(param_name: str,
                         widget: QWidget,
                         show_border: bool = True) -> QWidget:
    """Create a single parameter row within a group."""
    row_widget = QWidget()
    row_widget.setObjectName(f"parameterRow")

    border_style = "border-bottom: 1px solid palette(border);" if show_border \
        else ""
    row_widget.setStyleSheet(f"""
        QWidget#parameterRow {{
            background-color: transparent;
            {border_style}
        }}
    """)

    row_layout = QHBoxLayout(row_widget)
    # row_layout.setContentsMargins(16, 12, 16, 12)
    row_layout.setSpacing(10)

    # Create label
    label_text = param_name
    label = QLabel(label_text)
    label.setStyleSheet("""
        QLabel {
            font-weight: 500;
            color: palette(text);
            font-size: 13px;
        }
    """)
    label.setMinimumWidth(180)
    label.setWordWrap(True)
    row_layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignLeft)
    row_layout.addWidget(widget, alignment=Qt.AlignmentFlag.AlignRight)
    return row_widget


class NavigablePanel(QFrame, metaclass=QABCMeta):
    """Base widget with vertical navigation and stacked content panels."""

    def __init__(self, parent: Optional[QWidget] = None,
                 show_header: bool = False):
        super().__init__(parent)
        self.nav_collapsed = False
        self.nav_items_data: List[Tuple] = []
        self.show_header = show_header

        self._setup_ui()

    def _setup_ui(self):
        """Set up the panel UI with vertical navigation."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Main splitter for fixed widget area and navigation/content
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.setChildrenCollapsible(False)

        # Top fixed widget area (hidden by default)
        self.fixed_widget_container = QFrame()
        self.fixed_widget_layout = QVBoxLayout(self.fixed_widget_container)
        self.fixed_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.fixed_widget_layout.setSpacing(0)
        self.fixed_widget_container.hide()

        # Navigation and content area
        nav_content_widget = QFrame()
        nav_content_layout = QHBoxLayout(nav_content_widget)
        nav_content_layout.setContentsMargins(0, 0, 0, 0)
        nav_content_layout.setSpacing(0)

        # Navigation sidebar container
        nav_container = QFrame()
        nav_container_layout = QVBoxLayout(nav_container)
        nav_container_layout.setContentsMargins(0, 0, 0, 0)
        nav_container_layout.setSpacing(0)

        # Menu header (optional)
        if self.show_header:
            menu_header = self._create_menu_header()
            nav_container_layout.addWidget(menu_header)

        # Navigation list
        self.nav_list = self._create_navigation_list()
        nav_container_layout.addWidget(self.nav_list)

        # Content stack
        self.content_stack = QStackedWidget()
        self.content_stack.setObjectName("contentStack")

        # Add pages (implemented by subclasses)
        self.add_pages()

        # Connect navigation
        self.nav_list.currentRowChanged.connect(
            self.content_stack.setCurrentIndex)
        self.nav_list.setCurrentRow(0)

        nav_content_layout.addWidget(nav_container)
        nav_content_layout.addWidget(self.content_stack, 1)

        nav_content_widget.setObjectName("navContent")
        nav_container.setStyleSheet("border-bottom: 1px solid "
                                    "palette(mid); border-left: none; "
                                    "border-right: none; border-top: none")
        self.content_stack.setStyleSheet("QStackedWidget {border: 1px solid "
                                         "palette(mid); border-left: none; "
                                         "border-top: none;}")

        self.main_splitter.addWidget(nav_content_widget)
        self.main_splitter.addWidget(self.fixed_widget_container)

        main_layout.addWidget(self.main_splitter)

    def _create_menu_header(self) -> QWidget:
        """Create the menu header with title and toggle button."""
        menu_header = QWidget()
        menu_header.setObjectName("menuHeader")
        menu_header.setStyleSheet(MENU_HEADER_STYLESHEET)
        menu_header.setFixedHeight(60)

        header_layout = QHBoxLayout(menu_header)

        # Title area
        self.title_icon_label = QWidget()
        title_icon_layout = QHBoxLayout(self.title_icon_label)
        title_icon_layout.setContentsMargins(0, 0, 0, 0)
        title_icon_layout.setSpacing(4)

        icon_label = QToolButton()
        icon_label.setCheckable(False)
        title_icon = self.get_title_icon()
        icon_label.setIcon(title_icon)
        icon_label.setStyleSheet("border: none; background: transparent;")
        title_icon_layout.addWidget(icon_label)

        title_label = QLabel(self.get_title_text())
        font = title_label.font()
        font.setBold(True)
        title_label.setFont(font)
        title_icon_layout.addWidget(title_label)

        header_layout.addWidget(self.title_icon_label)
        header_layout.addStretch()

        # Toggle button
        self.toggle_button = QPushButton()
        self.toggle_button.setObjectName("toggleButton")
        self.toggle_button.setToolTip("Collapse/Expand Menu")
        menu_icon = get_icon(IconType.MENU_FOLD, QSize(256, 256),
                             self.palette().text().color())
        self.toggle_button.setIcon(menu_icon)
        self.toggle_button.setStyleSheet(TOGGLE_BUTTON_STYLESHEET)
        self.toggle_button.clicked.connect(self._toggle_navigation)

        header_layout.addWidget(self.toggle_button)

        return menu_header

    def _create_navigation_list(self) -> QListWidget:
        """Create the navigation list widget."""
        nav_list = QListWidget()
        nav_list.setObjectName("navigationList")
        nav_list.setMaximumWidth(200)
        nav_list.setMinimumWidth(180)
        nav_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        nav_list.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        nav_list.setSpacing(0)
        nav_list.setStyleSheet(NAV_LIST_STYLESHEET)
        return nav_list

    def _toggle_navigation(self):
        """Toggle between expanded and collapsed navigation."""
        self.nav_collapsed = not self.nav_collapsed
        menu_icon = (IconType.MENU_UNFOLD
                     if self.nav_collapsed else IconType.MENU_FOLD)

        if self.show_header:
            self.toggle_button.setIcon(
                get_icon(menu_icon, QSize(256, 256),
                         self.palette().text().color())
            )

        if self.nav_collapsed:
            self.nav_list.setMaximumWidth(60)
            self.nav_list.setMinimumWidth(60)
            self.nav_list.setStyleSheet(NAV_LIST_COLLAPSED_STYLESHEET)
            if self.show_header:
                self.title_icon_label.hide()

            for i in range(self.nav_list.count()):
                item = self.nav_list.item(i)
                if i < len(self.nav_items_data):
                    icon, text = self.nav_items_data[i]
                    item.setText("")
                    item.setToolTip(text)
                    item.setSizeHint(QSize(60, 48))
        else:
            self.nav_list.setMaximumWidth(200)
            self.nav_list.setMinimumWidth(180)
            self.nav_list.setStyleSheet(NAV_LIST_STYLESHEET)
            if self.show_header:
                self.title_icon_label.show()

            for i in range(self.nav_list.count()):
                item = self.nav_list.item(i)
                if i < len(self.nav_items_data):
                    icon, text = self.nav_items_data[i]
                    item.setText(text)
                    item.setToolTip("")
                    item.setSizeHint(QSize())

    def add_page(self, icon_type: IconType, title: str, widget: QWidget):
        """Add a page to the navigation and content stack.

        Args:
            icon_type: IconType for the navigation item
            title: Display title for the navigation item
            widget: Widget to display when this page is selected
        """
        icon_obj = get_icon(icon_type, QSize(256, 256),
                            self.palette().text().color())
        item = QListWidgetItem(icon_obj, title)
        self.nav_items_data.append((icon_obj, title))
        self.nav_list.addItem(item)
        self.content_stack.addWidget(widget)

    def set_fixed_widget(self, widget: QWidget):
        """Set a widget in the fixed top area.

        Args:
            widget: Widget to display in the fixed area
        """
        while self.fixed_widget_layout.count():
            child = self.fixed_widget_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.fixed_widget_layout.addWidget(widget)

    def remove_fixed_widget(self):
        """Remove the fixed widget and hide the container."""
        while self.fixed_widget_layout.count():
            child = self.fixed_widget_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.fixed_widget_container.hide()


    def set_fixed_widget_visible(self, show:bool):
        self.fixed_widget_container.setVisible(show)

    def set_splitter_sizes(self, fixed_size: int, content_size: int):
        """Set the sizes of the splitter sections."""
        self.main_splitter.setSizes([fixed_size, content_size])

    def current_page_index(self) -> int:
        """Return the current page index."""
        return self.nav_list.currentRow()

    def set_current_page(self, index: int):
        """Set the current page by index."""
        self.nav_list.setCurrentRow(index)

    def page_count(self) -> int:
        """Return the number of pages."""
        return self.content_stack.count()

    @abstractmethod
    def add_pages(self):
        """Add all pages to the panel. Implement in subclass."""
        pass

    def get_title_icon(self):
        """Get title bar icon. Override to customize."""
        return get_icon(
            IconType.SETTINGS,
            QSize(256, 256),
            self.palette().text().color()
        )

    def get_title_text(self) -> str:
        """Get title text. Override to customize."""
        return "Settings"
