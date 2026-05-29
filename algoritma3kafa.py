#!/usr/bin/env python3
"""
Head Tracking Mouse Controller with PyQt6 GUI and Auto Calibration
Features:
  - MediaPipe-based head pose estimation
  - Mouse movement control
  - Global hotkey support (1=tracking toggle, 2=preview toggle, 3=debug toggle, 4=auto-calibration, Delete=exit)
  - PyQt6 desktop GUI with camera preview, control panel, and settings
  - Auto-calibration wizard to compute optimal deadzone and sensitivity gains
  - JSON config persistence
"""

import os
import sys
import json
import time
import math
import threading
import traceback
from dataclasses import dataclass, asdict
from collections import deque
from typing import Optional, Tuple, List

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import PoseLandmarker

import ctypes
GetAsyncKeyState = ctypes.windll.user32.GetAsyncKeyState

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox, QComboBox,
    QGroupBox, QGridLayout, QScrollArea, QCheckBox
)

# ============================================================================
# CONSTANTS
# ============================================================================

AUTO_CAL_STEP_SECONDS = 5.0
AUTO_CAL_TRANSITION_SECONDS = 3.0
AUTO_CAL_STATES = ("center", "look_left", "look_right", "look_up", "look_down")

CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "camera_index": 0,
    "camera_width": 1280,
    "camera_height": 720,
    "camera_fps": 30,
    "deadzone": 0.01,
    "temporal_smoothing": 0.2,
    "smoothness": 0.8,
    "x_curve_gain": 1.0,
    "y_curve_gain": 1.0,
    "tracking_enabled": True,
    "preview_enabled": True,
    "debug_enabled": False,
}

# ============================================================================
# HELPERS
# ============================================================================

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def soft_deadzone(value, deadzone):
    if abs(value) < deadzone:
        return 0.0
    return value

def lerp(a, b, t):
    return a + (b - a) * t

def percentile(data, p):
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int((p / 100.0) * (len(sorted_data) - 1))
    return sorted_data[idx]

def robust_std(data):
    if len(data) < 2:
        return 0.0
    p10 = percentile(data, 10)
    p90 = percentile(data, 90)
    trimmed = [x for x in data if p10 <= x <= p90]
    if len(trimmed) < 2:
        return 0.0
    return np.std(trimmed)

def _time_ms():
    return time.time() * 1000.0

# ============================================================================
# CONFIG MANAGER
# ============================================================================

class Config:
    def __init__(self, filename=CONFIG_FILE):
        self.filename = filename
        self.data = DEFAULT_CONFIG.copy()
        self.load_json()

    def load_json(self):
        if os.path.isfile(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    loaded = json.load(f)
                    self.data.update(loaded)
            except Exception as e:
                print(f"Error loading config: {e}")

    def save_json(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value

    def refresh_runtime_config(self):
        # Subclasses override if needed; base is no-op for now
        pass

# ============================================================================
# AUTO CALIBRATOR
# ============================================================================

@dataclass
class AutoCalResult:
    deadzone: float
    temporal_smoothing: float
    x_curve_gain: float
    y_curve_gain: float
    summary_lines: List[str]

class AutoCalibrator:
    def __init__(self):
        self.active = False
        self.step_idx = 0
        self.step_start_time = None
        self.transition_end_time = None
        self.in_transition = False
        
        self.step_samples = {state: {"dx": [], "dy": []} for state in AUTO_CAL_STATES}
        self.result = None

    def start(self):
        self.active = True
        self.step_idx = 0
        self.step_start_time = _time_ms()
        self.transition_end_time = None
        self.in_transition = False
        self.step_samples = {state: {"dx": [], "dy": []} for state in AUTO_CAL_STATES}
        self.result = None

    def is_active(self):
        return self.active

    def update(self, face_found: bool, corrected_dx: float, corrected_dy: float):
        if not self.active:
            return

        now_ms = _time_ms()

        # Handle transition between steps
        if self.in_transition:
            if now_ms >= self.transition_end_time:
                self.in_transition = False
                self.step_idx += 1
                if self.step_idx >= len(AUTO_CAL_STATES):
                    self._finalize()
                    return
                self.step_start_time = now_ms
            else:
                return

        # Collect samples for current step
        elapsed_sec = (now_ms - self.step_start_time) / 1000.0
        if elapsed_sec < AUTO_CAL_STEP_SECONDS and face_found:
            state = AUTO_CAL_STATES[self.step_idx]
            self.step_samples[state]["dx"].append(corrected_dx)
            self.step_samples[state]["dy"].append(corrected_dy)

        # Transition to next step
        if elapsed_sec >= AUTO_CAL_STEP_SECONDS:
            self.in_transition = True
            self.transition_end_time = now_ms + (AUTO_CAL_TRANSITION_SECONDS * 1000.0)

    def poll_result(self):
        result = self.result
        if result:
            self.result = None
        return result

    def _finalize(self):
        # Compute deadzone from center step
        center_dx = self.step_samples["center"]["dx"]
        center_dy = self.step_samples["center"]["dy"]
        
        if center_dx and center_dy:
            # Trim outliers (5th to 95th percentile)
            dx_p5 = percentile(center_dx, 5)
            dx_p95 = percentile(center_dx, 95)
            dy_p5 = percentile(center_dy, 5)
            dy_p95 = percentile(center_dy, 95)
            
            dx_trimmed = [x for x in center_dx if dx_p5 <= x <= dx_p95]
            dy_trimmed = [y for y in center_dy if dy_p5 <= y <= dy_p95]
            
            std_x = robust_std(center_dx)
            std_y = robust_std(center_dy)
            deadzone = clamp(max(std_x, std_y) * 2.2, 0.005, 0.040)
        else:
            deadzone = 0.01

        # Temporal smoothing from overall stability
        temporal_smoothing = clamp(0.2, 0.10, 0.45)

        # Compute gains from left/right and up/down ranges
        left_data = self.step_samples["look_left"]["dx"]
        right_data = self.step_samples["look_right"]["dx"]
        up_data = self.step_samples["look_up"]["dy"]
        down_data = self.step_samples["look_down"]["dy"]

        def compute_gain(data):
            if not data:
                return 1.0
            p10 = percentile(data, 10)
            p90 = percentile(data, 90)
            data_range = abs(p90 - p10)
            if data_range < 0.001:
                return 1.0
            return clamp(0.55 / data_range, 0.3, 2.5)

        x_curve_gain = compute_gain(left_data + right_data)
        y_curve_gain = compute_gain(up_data + down_data)

        summary = [
            f"Deadzone: {deadzone:.4f}",
            f"Temporal Smoothing: {temporal_smoothing:.4f}",
            f"X Gain: {x_curve_gain:.4f}",
            f"Y Gain: {y_curve_gain:.4f}",
        ]

        self.result = AutoCalResult(
            deadzone=deadzone,
            temporal_smoothing=temporal_smoothing,
            x_curve_gain=x_curve_gain,
            y_curve_gain=y_curve_gain,
            summary_lines=summary,
        )
        self.active = False

    def overlay_info(self):
        if not self.active:
            return None, None, None, None

        now_ms = _time_ms()
        state = AUTO_CAL_STATES[self.step_idx]
        
        if self.in_transition:
            elapsed = (now_ms - (self.transition_end_time - AUTO_CAL_TRANSITION_SECONDS * 1000.0)) / 1000.0
            countdown = max(0, AUTO_CAL_TRANSITION_SECONDS - elapsed)
            return state, countdown, self.step_idx, len(AUTO_CAL_STATES)
        
        elapsed = (now_ms - self.step_start_time) / 1000.0
        countdown = max(0, AUTO_CAL_STEP_SECONDS - elapsed)
        return state, countdown, self.step_idx, len(AUTO_CAL_STATES)

# ============================================================================
# SNAPSHOT (threading-safe data structure)
# ============================================================================

@dataclass
class TrackerSnapshot:
    face_found: bool = False
    dx: float = 0.0
    dy: float = 0.0
    fps: float = 0.0
    corrected_dx: float = 0.0
    corrected_dy: float = 0.0
    auto_cal_active: bool = False
    auto_cal_step: str = ""
    auto_cal_countdown: float = 0.0
    auto_cal_progress: Tuple[int, int] = (0, 0)
    auto_cal_summary: List[str] = None

# ============================================================================
# HEAD TRACKER
# ============================================================================

class HeadTracker(threading.Thread):
    def __init__(self, config: Config):
        super().__init__(daemon=True)
        self.config = config
        self.running = False
        self.face_found = False
        self.dx = 0.0
        self.dy = 0.0
        self.frame_times = deque(maxlen=30)
        self.fps = 0.0
        
        self.smoothed_x = 0.0
        self.smoothed_y = 0.0
        
        self.snapshot = TrackerSnapshot()
        self.snapshot_lock = threading.Lock()
        
        self.auto_calibrator = AutoCalibrator()

    def run(self):
        self.running = True
        
        # Initialize MediaPipe PoseLandmarker
        try:
            base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
            options = python.vision.PoseLandmarkerOptions(base_options=base_options, num_poses=1)
            landmarker = python.vision.PoseLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Error loading MediaPipe model: {e}")
            try:
                landmarker = None
            except:
                pass

        cap = cv2.VideoCapture(self.config.get("camera_index", 0))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get("camera_width", 1280))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get("camera_height", 720))
        cap.set(cv2.CAP_PROP_FPS, self.config.get("camera_fps", 30))

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame_time = time.time()
                self.frame_times.append(frame_time)
                
                if len(self.frame_times) > 1:
                    self.fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0] + 1e-6)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, c = frame_rgb.shape
                
                self.face_found = False
                self.dx = 0.0
                self.dy = 0.0

                if landmarker:
                    try:
                        image = python.vision.Image(image_format=python.vision.ImageFormat.SRGB, data=frame_rgb)
                        results = landmarker.detect(image)
                        
                        if results.pose_landmarks and len(results.pose_landmarks) > 0:
                            landmarks = results.pose_landmarks[0]
                            
                            if len(landmarks) >= 11:
                                left_shoulder = landmarks[11]
                                right_shoulder = landmarks[12]
                                nose = landmarks[0]
                                
                                shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2.0
                                shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2.0
                                
                                self.dx = nose.x - shoulder_mid_x
                                self.dy = nose.y - shoulder_mid_y
                                self.face_found = True
                    except Exception as e:
                        print(f"MediaPipe detection error: {e}")

                deadzone = self.config.get("deadzone", 0.01)
                temporal_smoothing = self.config.get("temporal_smoothing", 0.2)
                smoothness = self.config.get("smoothness", 0.8)
                
                corrected_dx = soft_deadzone(self.dx, deadzone)
                corrected_dy = soft_deadzone(self.dy, deadzone)
                
                self.smoothed_x = lerp(self.smoothed_x, corrected_dx, 1.0 - temporal_smoothing)
                self.smoothed_y = lerp(self.smoothed_y, corrected_dy, 1.0 - temporal_smoothing)

                # Update auto-calibrator
                self.auto_calibrator.update(self.face_found, corrected_dx, corrected_dy)
                
                # Check if auto-calibration finished
                auto_result = self.auto_calibrator.poll_result()
                if auto_result:
                    self._apply_auto_calibration(auto_result)

                # Update snapshot
                with self.snapshot_lock:
                    self.snapshot.face_found = self.face_found
                    self.snapshot.dx = self.dx
                    self.snapshot.dy = self.dy
                    self.snapshot.fps = self.fps
                    self.snapshot.corrected_dx = corrected_dx
                    self.snapshot.corrected_dy = corrected_dy
                    
                    if self.auto_calibrator.is_active():
                        state, countdown, step_idx, total_steps = self.auto_calibrator.overlay_info()
                        self.snapshot.auto_cal_active = True
                        self.snapshot.auto_cal_step = state or ""
                        self.snapshot.auto_cal_countdown = countdown or 0.0
                        self.snapshot.auto_cal_progress = (step_idx, total_steps)
                    else:
                        self.snapshot.auto_cal_active = False

                time.sleep(0.01)
        finally:
            cap.release()

    def get_snapshot(self) -> TrackerSnapshot:
        with self.snapshot_lock:
            return TrackerSnapshot(**asdict(self.snapshot))

    def start_auto_calibration(self):
        self.auto_calibrator.start()

    def _apply_auto_calibration(self, result: AutoCalResult):
        self.config.set("deadzone", result.deadzone)
        self.config.set("temporal_smoothing", result.temporal_smoothing)
        self.config.set("x_curve_gain", result.x_curve_gain)
        self.config.set("y_curve_gain", result.y_curve_gain)
        self.config.save_json()
        print("Auto-calibration applied and saved to config.")

    def stop(self):
        self.running = False
        self.join(timeout=2)

# ============================================================================
# MOUSE CONTROLLER
# ============================================================================

class MouseController(threading.Thread):
    def __init__(self, tracker: HeadTracker, config: Config):
        super().__init__(daemon=True)
        self.tracker = tracker
        self.config = config
        self.running = False

    def run(self):
        self.running = True
        try:
            import pyautogui
            pyautogui.FAILSAFE = False
        except ImportError:
            print("Warning: pyautogui not available; mouse control disabled")
            return

        while self.running:
            snapshot = self.tracker.get_snapshot()
            
            if snapshot.face_found and self.config.get("tracking_enabled", True):
                x_gain = self.config.get("x_curve_gain", 1.0)
                y_gain = self.config.get("y_curve_gain", 1.0)
                
                dx_scaled = snapshot.corrected_dx * x_gain * 200
                dy_scaled = snapshot.corrected_dy * y_gain * 200
                
                if abs(dx_scaled) > 0.5 or abs(dy_scaled) > 0.5:
                    try:
                        pyautogui.moveRel(int(dx_scaled), int(dy_scaled), duration=0.01)
                    except Exception as e:
                        pass

            time.sleep(0.016)

    def stop(self):
        self.running = False
        self.join(timeout=2)

# ============================================================================
# HOTKEY WATCHER
# ============================================================================

class HotkeyWatcher:
    def __init__(self):
        self.hotkey_map = {
            '1': 0x31,
            '2': 0x32,
            '3': 0x33,
            '4': 0x34,
            'delete': 0x2E,
        }

    def is_pressed(self, key: str) -> bool:
        if key not in self.hotkey_map:
            return False
        vk_code = self.hotkey_map[key]
        state = GetAsyncKeyState(vk_code)
        return (state >> 15) != 0

# ============================================================================
# PyQt6 GUI
# ============================================================================

class ConfigManager:
    def __init__(self, config: Config):
        self.config = config

    def save(self):
        self.config.save_json()

    def load(self):
        self.config.load_json()

    def reset(self):
        self.config.data = DEFAULT_CONFIG.copy()
        self.config.save_json()

class MainWindow(QMainWindow):
    def __init__(self, tracker: HeadTracker, mouse_controller: MouseController, hotkey_watcher: HotkeyWatcher, config: Config):
        super().__init__()
        self.tracker = tracker
        self.mouse_controller = mouse_controller
        self.hotkey_watcher = hotkey_watcher
        self.config = config
        self.config_manager = ConfigManager(config)
        
        self.setWindowTitle("Head Tracking Mouse Controller")
        self.setGeometry(100, 100, 1400, 700)

        # Main layout: horizontal (left panel + right preview)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()

        # LEFT PANEL (scrollable)
        left_panel = QWidget()
        left_layout = QVBoxLayout()

        # Tracking group
        tracking_group = QGroupBox("Tracking")
        tracking_layout = QGridLayout()
        self.tracking_label = QLabel("Status: Initializing...")
        self.face_detection_label = QLabel("Face: Not detected")
        self.fps_label = QLabel("FPS: 0")
        tracking_layout.addWidget(self.tracking_label, 0, 0)
        tracking_layout.addWidget(self.face_detection_label, 1, 0)
        tracking_layout.addWidget(self.fps_label, 2, 0)
        tracking_group.setLayout(tracking_layout)
        left_layout.addWidget(tracking_group)

        # Calibration group
        calibration_group = QGroupBox("Calibration")
        calibration_layout = QGridLayout()
        self.recalibrate_btn = QPushButton("Manual Recalibration")
        self.auto_cal_btn = QPushButton("Start Auto Calibration")
        self.auto_cal_btn.clicked.connect(self._on_auto_cal_clicked)
        calibration_layout.addWidget(self.recalibrate_btn, 0, 0)
        calibration_layout.addWidget(self.auto_cal_btn, 1, 0)
        calibration_group.setLayout(calibration_layout)
        left_layout.addWidget(calibration_group)

        # Sensitivity group
        sensitivity_group = QGroupBox("Sensitivity")
        sensitivity_layout = QGridLayout()
        
        sensitivity_layout.addWidget(QLabel("Deadzone:"), 0, 0)
        self.deadzone_slider = QSlider(Qt.Orientation.Horizontal)
        self.deadzone_slider.setRange(0, 100)
        self.deadzone_slider.setValue(int(self.config.get("deadzone", 0.01) * 1000))
        self.deadzone_spinbox = QDoubleSpinBox()
        self.deadzone_spinbox.setRange(0.0, 0.1)
        self.deadzone_spinbox.setSingleStep(0.001)
        self.deadzone_spinbox.setValue(self.config.get("deadzone", 0.01))
        self.deadzone_slider.valueChanged.connect(lambda v: self._update_deadzone(v / 1000.0))
        self.deadzone_spinbox.valueChanged.connect(lambda v: self._update_deadzone_spinbox(v))
        sensitivity_layout.addWidget(self.deadzone_slider, 0, 1)
        sensitivity_layout.addWidget(self.deadzone_spinbox, 0, 2)

        sensitivity_layout.addWidget(QLabel("Temporal Smoothing:"), 1, 0)
        self.temporal_slider = QSlider(Qt.Orientation.Horizontal)
        self.temporal_slider.setRange(0, 100)
        self.temporal_slider.setValue(int(self.config.get("temporal_smoothing", 0.2) * 100))
        self.temporal_spinbox = QDoubleSpinBox()
        self.temporal_spinbox.setRange(0.0, 1.0)
        self.temporal_spinbox.setSingleStep(0.01)
        self.temporal_spinbox.setValue(self.config.get("temporal_smoothing", 0.2))
        self.temporal_slider.valueChanged.connect(lambda v: self._update_temporal(v / 100.0))
        self.temporal_spinbox.valueChanged.connect(lambda v: self._update_temporal_spinbox(v))
        sensitivity_layout.addWidget(self.temporal_slider, 1, 1)
        sensitivity_layout.addWidget(self.temporal_spinbox, 1, 2)

        sensitivity_layout.addWidget(QLabel("X Gain:"), 2, 0)
        self.x_gain_spinbox = QDoubleSpinBox()
        self.x_gain_spinbox.setRange(0.1, 5.0)
        self.x_gain_spinbox.setSingleStep(0.1)
        self.x_gain_spinbox.setValue(self.config.get("x_curve_gain", 1.0))
        self.x_gain_spinbox.valueChanged.connect(lambda v: self.config.set("x_curve_gain", v))
        sensitivity_layout.addWidget(self.x_gain_spinbox, 2, 1, 1, 2)

        sensitivity_layout.addWidget(QLabel("Y Gain:"), 3, 0)
        self.y_gain_spinbox = QDoubleSpinBox()
        self.y_gain_spinbox.setRange(0.1, 5.0)
        self.y_gain_spinbox.setSingleStep(0.1)
        self.y_gain_spinbox.setValue(self.config.get("y_curve_gain", 1.0))
        self.y_gain_spinbox.valueChanged.connect(lambda v: self.config.set("y_curve_gain", v))
        sensitivity_layout.addWidget(self.y_gain_spinbox, 3, 1, 1, 2)

        sensitivity_group.setLayout(sensitivity_layout)
        left_layout.addWidget(sensitivity_group)

        # Camera group
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QGridLayout()
        camera_layout.addWidget(QLabel("Camera Index:"), 0, 0)
        self.camera_index_spinbox = QSpinBox()
        self.camera_index_spinbox.setValue(self.config.get("camera_index", 0))
        self.camera_index_spinbox.valueChanged.connect(lambda v: self.config.set("camera_index", v))
        camera_layout.addWidget(self.camera_index_spinbox, 0, 1, 1, 2)
        camera_group.setLayout(camera_layout)
        left_layout.addWidget(camera_group)

        # Controls group
        controls_group = QGroupBox("Controls")
        controls_layout = QGridLayout()
        self.tracking_toggle_btn = QPushButton("Tracking: ON")
        self.tracking_toggle_btn.clicked.connect(self._toggle_tracking)
        self.preview_toggle_btn = QPushButton("Preview: ON")
        self.preview_toggle_btn.clicked.connect(self._toggle_preview)
        controls_layout.addWidget(self.tracking_toggle_btn, 0, 0)
        controls_layout.addWidget(self.preview_toggle_btn, 0, 1)
        controls_group.setLayout(controls_layout)
        left_layout.addWidget(controls_group)

        # Save/Load/Reset
        save_load_group = QGroupBox("Config")
        save_load_layout = QGridLayout()
        save_btn = QPushButton("Save Config")
        load_btn = QPushButton("Load Config")
        reset_btn = QPushButton("Reset Defaults")
        save_btn.clicked.connect(self._save_config)
        load_btn.clicked.connect(self._load_config)
        reset_btn.clicked.connect(self._reset_config)
        save_load_layout.addWidget(save_btn, 0, 0)
        save_load_layout.addWidget(load_btn, 0, 1)
        save_load_layout.addWidget(reset_btn, 1, 0, 1, 2)
        save_load_group.setLayout(save_load_layout)
        left_layout.addWidget(save_load_group)

        left_layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidget(left_panel)
        scroll_area.setWidgetResizable(True)
        left_panel.setLayout(left_layout)

        main_layout.addWidget(scroll_area, 2)

        # RIGHT PANEL (camera preview)
        preview_group = QGroupBox("Camera Preview")
        preview_layout = QVBoxLayout()
        self.camera_label = QLabel("Preview Disabled")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setMinimumWidth(640)
        self.camera_label.setMinimumHeight(480)
        preview_layout.addWidget(self.camera_label)
        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group, 3)

        central_widget.setLayout(main_layout)

        # UI update timer
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self._tick_ui)
        self.ui_timer.start(30)

    def _tick_ui(self):
        # Poll snapshot
        snapshot = self.tracker.get_snapshot()
        
        # Update status labels
        self.tracking_label.setText(f"Status: {'Running' if self.config.get('tracking_enabled') else 'Paused'}")
        self.face_detection_label.setText(f"Face: {'Detected' if snapshot.face_found else 'Not detected'}")
        self.fps_label.setText(f"FPS: {snapshot.fps:.1f}")

        # Update camera preview
        if self.config.get("preview_enabled", True):
            try:
                cap = cv2.VideoCapture(self.config.get("camera_index", 0))
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, c = frame.shape
                    bytes_per_line = 3 * w
                    qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    scaled = pixmap.scaledToWidth(640, Qt.TransformationMode.SmoothTransformation)
                    self.camera_label.setPixmap(scaled)
            except Exception as e:
                self.camera_label.setText(f"Preview Error: {str(e)[:50]}")
        else:
            self.camera_label.setText("Preview Disabled")

        # Poll hotkeys
        if self.hotkey_watcher.is_pressed('1'):
            self._toggle_tracking()
            time.sleep(0.2)
        if self.hotkey_watcher.is_pressed('2'):
            self._toggle_preview()
            time.sleep(0.2)
        if self.hotkey_watcher.is_pressed('4'):
            self._on_auto_cal_clicked()
            time.sleep(0.2)
        if self.hotkey_watcher.is_pressed('delete'):
            self.close()

    def _toggle_tracking(self):
        current = self.config.get("tracking_enabled", True)
        self.config.set("tracking_enabled", not current)
        btn_text = "Tracking: ON" if not current else "Tracking: OFF"
        self.tracking_toggle_btn.setText(btn_text)

    def _toggle_preview(self):
        current = self.config.get("preview_enabled", True)
        self.config.set("preview_enabled", not current)
        btn_text = "Preview: ON" if not current else "Preview: OFF"
        self.preview_toggle_btn.setText(btn_text)

    def _on_auto_cal_clicked(self):
        self.tracker.start_auto_calibration()
        self.auto_cal_btn.setText("Auto-Cal: Running...")
        self.auto_cal_btn.setEnabled(False)

    def _update_deadzone(self, value):
        self.config.set("deadzone", value)
        self.deadzone_spinbox.blockSignals(True)
        self.deadzone_spinbox.setValue(value)
        self.deadzone_spinbox.blockSignals(False)

    def _update_deadzone_spinbox(self, value):
        self.config.set("deadzone", value)
        self.deadzone_slider.blockSignals(True)
        self.deadzone_slider.setValue(int(value * 1000))
        self.deadzone_slider.blockSignals(False)

    def _update_temporal(self, value):
        self.config.set("temporal_smoothing", value)
        self.temporal_spinbox.blockSignals(True)
        self.temporal_spinbox.setValue(value)
        self.temporal_spinbox.blockSignals(False)

    def _update_temporal_spinbox(self, value):
        self.config.set("temporal_smoothing", value)
        self.temporal_slider.blockSignals(True)
        self.temporal_slider.setValue(int(value * 100))
        self.temporal_slider.blockSignals(False)

    def _save_config(self):
        self.config_manager.save()
        self.tracking_label.setText("Status: Config Saved")

    def _load_config(self):
        self.config_manager.load()
        self.tracking_label.setText("Status: Config Loaded")

    def _reset_config(self):
        self.config_manager.reset()
        self.tracking_label.setText("Status: Config Reset")

    def closeEvent(self, event):
        self.ui_timer.stop()
        self.tracker.stop()
        self.mouse_controller.stop()
        event.accept()

# ============================================================================
# MAIN
# ============================================================================

def main():
    config = Config(CONFIG_FILE)
    
    tracker = HeadTracker(config)
    tracker.start()
    
    mouse_controller = MouseController(tracker, config)
    mouse_controller.start()
    
    hotkey_watcher = HotkeyWatcher()
    
    app = QApplication(sys.argv)
    window = MainWindow(tracker, mouse_controller, hotkey_watcher, config)
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
