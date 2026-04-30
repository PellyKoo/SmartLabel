"""
SmartLabel 主窗口。

布局：
    ┌─────────────────────────────────────────────────────┐
    │  [引擎面板]  │  [Tab 容器：预标注/质检/视频/评估/设置]  │
    │              │                                      │
    │  (左侧固定)   │  (右上：Tab 切换)                    │
    │              ├──────────────────────────────────────┤
    │              │  [日志控制台 (可折叠)]                │
    └─────────────────────────────────────────────────────┘
    [状态栏: 引擎状态 / 最新消息]
"""
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QTabWidget, QStatusBar, QAction, QMessageBox,
)

from src.gui.widgets.engine_panel import EnginePanel
from src.gui.widgets.log_console import LogConsole
from src.gui.widgets.preannotate_tab import PreAnnotateTab
from src.gui.widgets.qualitycheck_tab import QualityCheckTab
from src.gui.widgets.video_tab import VideoTab
from src.gui.widgets.benchmark_tab import BenchmarkTab
from src.gui.widgets.settings_tab import SettingsTab
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QSS_PATH = PROJECT_ROOT / "src" / "gui" / "styles" / "theme.qss"


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SmartLabel — AI 辅助标注与质检平台")
        self.resize(1400, 900)

        self._engine_panel: Optional[EnginePanel] = None
        self._log_console: Optional[LogConsole] = None
        self._tabs: Optional[QTabWidget] = None
        self._preannotate_tab: Optional[PreAnnotateTab] = None
        self._qualitycheck_tab: Optional[QualityCheckTab] = None
        self._video_tab: Optional[VideoTab] = None
        self._benchmark_tab: Optional[BenchmarkTab] = None
        self._settings_tab: Optional[SettingsTab] = None

        self._build_ui()
        self._wire_signals()
        self._apply_theme()

        # 启动后日志控制台抓取后端 logger
        QTimer.singleShot(0, self._attach_logger)

    # ------------- build -------------

    def _build_ui(self):
        self._build_menu()

        central = QWidget()
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._engine_panel = EnginePanel()

        right = QWidget()
        right_v = QVBoxLayout(right)
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.setSpacing(0)

        # Tab 容器
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        self._preannotate_tab = PreAnnotateTab()
        self._qualitycheck_tab = QualityCheckTab()
        self._video_tab = VideoTab()
        self._benchmark_tab = BenchmarkTab()
        self._settings_tab = SettingsTab()

        self._tabs.addTab(self._preannotate_tab, "预标注")
        self._tabs.addTab(self._qualitycheck_tab, "质检")
        self._tabs.addTab(self._video_tab, "视频分类")
        self._tabs.addTab(self._benchmark_tab, "评估")
        self._tabs.addTab(self._settings_tab, "设置")

        self._log_console = LogConsole()

        # Tab 与日志用垂直可调分隔
        v_split = QSplitter(Qt.Vertical)
        v_split.addWidget(self._tabs)
        v_split.addWidget(self._log_console)
        v_split.setStretchFactor(0, 4)
        v_split.setStretchFactor(1, 1)
        v_split.setSizes([700, 200])
        right_v.addWidget(v_split)

        outer.addWidget(self._engine_panel)
        outer.addWidget(right, stretch=1)

        self.setCentralWidget(central)

        # 状态栏
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("就绪")

    def _build_menu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("文件(&F)")

        quit_action = QAction("退出(&Q)", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = menu.addMenu("帮助(&H)")
        about_action = QAction("关于 SmartLabel", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _wire_signals(self):
        panel = self._engine_panel
        panel.primary_engine_changed.connect(self._on_primary_changed)
        panel.vlm_engine_changed.connect(self._on_vlm_changed)
        panel.status_message.connect(
            lambda msg: self.statusBar().showMessage(msg, 5000)
        )

    def _apply_theme(self):
        if QSS_PATH.exists():
            try:
                with open(QSS_PATH, "r", encoding="utf-8") as f:
                    self.setStyleSheet(f.read())
            except OSError as e:
                logger.warning(f"读取主题文件失败: {e}")

    def _attach_logger(self):
        """启动后把 logger 输出挂到 UI 日志控制台"""
        self._log_console.attach_to_logger("smartlabel")
        self._log_console.append("INFO", "SmartLabel GUI 已启动")

    # ------------- slots -------------

    def _on_primary_changed(self, engine):
        """主引擎变化时广播到所有 Tab"""
        for t in (self._preannotate_tab, self._qualitycheck_tab,
                   self._video_tab, self._benchmark_tab, self._settings_tab):
            if t is not None and hasattr(t, "set_primary_engine"):
                t.set_primary_engine(engine)

    def _on_vlm_changed(self, engine):
        for t in (self._preannotate_tab, self._qualitycheck_tab,
                   self._video_tab, self._benchmark_tab, self._settings_tab):
            if t is not None and hasattr(t, "set_vlm_engine"):
                t.set_vlm_engine(engine)

    def _show_about(self):
        QMessageBox.about(
            self, "关于",
            "<b>SmartLabel</b><br>"
            "AI 辅助标注与质检平台<br><br>"
            "双引擎（VLM + 自有模型）+ 三前端（PyQt / CLI / Web）"
        )

    # ------------- lifecycle -------------

    def closeEvent(self, event):
        """关闭前释放引擎资源"""
        logger.info("窗口关闭，释放资源")
        try:
            if self._engine_panel is not None:
                self._engine_panel.shutdown()
            if self._log_console is not None:
                self._log_console.detach("smartlabel")
        except Exception:
            logger.exception("关闭清理出错")
        super().closeEvent(event)
