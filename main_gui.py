#!/usr/bin/env python3
"""
Spectroscopy Fitting GUI - PyQt6 Version
A user-friendly interface for fitting Lorentzian peaks to ion trap spectroscopy data.
"""

import sys
import os
import numpy as np

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
    QPushButton, QLabel, QTextEdit, QTabWidget, QGroupBox,
    QFileDialog, QMessageBox, QListWidget, QInputDialog, QDialog,
    QGridLayout, QDoubleSpinBox, QSpinBox, QSlider, QComboBox, QLineEdit,
    QMenu, QColorDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

# Matplotlib with PyQt6 backend
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Scientific computing
from scipy.signal import find_peaks
from scipy.optimize import least_squares, minimize_scalar, curve_fit
from scipy.special import gammaln, eval_genlaguerre
from scipy.constants import pi, atomic_mass as amu, hbar as HBAR, Boltzmann as KB
import h5py

# Physics constants for thermal analysis
HBAR = 1.0545718e-34
KB = 1.38064852e-23

def lamb_dicke(wavelength, ion_mass, omega, axis='z', laser_angle_deg=45):
    """Calculate Lamb-Dicke parameter for specific axis considering laser geometry."""
    k = 2 * np.pi / wavelength
    angle_rad = np.radians(laser_angle_deg)
    
    if axis.lower() == 'z':
        k_eff = k * np.cos(angle_rad)
    elif axis.lower() in ['x', 'y']:
        k_eff = k * np.sin(angle_rad) / np.sqrt(2)
    else:
        raise ValueError(f"Unknown axis: {axis}. Use 'x', 'y', or 'z'")
    
    eta = k_eff * np.sqrt(HBAR / (2 * ion_mass * omega))
    return eta

def n_max(T, omega, C_param=0.999):
    """Calculate maximum n for thermal state calculation"""
    if not (0 < C_param < 1):
        raise ValueError("C_param must be between 0 and 1 (exclusive).")
    if T <= 0 or omega <= 0:
        return 0
    log_arg = 1 - C_param
    if log_arg <= 0:
        return 0
    val = -(KB * T) / (HBAR * omega) * np.log(log_arg)
    if not np.isfinite(val) or val < 0:
        return 0
    return int(val)

def maxwell_boltzmann_distribution(n_max, omega, T):
    """Calculate thermal state probabilities following Boltzmann distribution"""
    n = np.arange(n_max + 1)
    E_n = HBAR * omega * n
    boltz = np.exp(-E_n / (KB * T))
    Z = boltz.sum()
    if Z == 0:
        probs = np.zeros_like(boltz)
        probs[0] = 1.0
        return probs
    return boltz / Z

def rabi_n_s(n, s, eta):
    """Calculate relative Rabi frequency for sideband transition |n⟩ → |n+s⟩"""
    n = np.asarray(n)
    n_target = n + s
    valid = n_target >= 0
    
    rabi_relative = np.zeros_like(n, dtype=float)
    
    if np.any(valid):
        n_valid = n[valid]
        n_target_valid = n_target[valid]
        
        n_min = np.minimum(n_valid, n_target_valid)
        n_max = np.maximum(n_valid, n_target_valid)
        
        if n_valid.size > 0:
            max_n = max(n_max.max(), abs(s))
            L_all = eval_genlaguerre(np.arange(max_n + 1), abs(s), eta**2)
            L = L_all[n_min]
            
            ratio = np.exp(0.5 * (gammaln(n_min + 1) - gammaln(n_max + 1)))
            rabi_relative[valid] = np.exp(-eta**2/2) * (eta**abs(s)) * ratio * L
    
    return rabi_relative

def calculate_sideband_probability(T, omega, s, wavelength, ion_mass, pulse_time, omega_tau=None, axis='z'):
    """Calculate excitation probability for sideband s at temperature T"""
    eta = lamb_dicke(wavelength, ion_mass, omega, axis=axis)
    max_n = n_max(T, omega)
    n_vals = np.arange(max_n + 1)
    
    # Thermal state probabilities (Boltzmann distribution)
    boltz = maxwell_boltzmann_distribution(max_n, omega, T)
    
    # Relative Rabi frequencies for this sideband (normalized to carrier)
    rabi_relative = rabi_n_s(n_vals, s, eta)
    
    # Use provided pulse area or default to π for π-pulse
    if omega_tau is None:
        Omega0_tau = np.pi  # Default: π-pulse
    else:
        Omega0_tau = omega_tau 
    
    # Effective pulse area for each transition: Ω_{n,n+s} * τ
    pulse_area = rabi_relative * Omega0_tau
    # pulse_area = Omega0_tau
    # Excitation probability for each Fock state: sin²(Ω*τ/2)
    excitation_probs = np.sin(pulse_area / 2) ** 2
    
    # Thermal average: sum over all Fock states weighted by thermal probabilities
    total_prob = np.sum(boltz * excitation_probs)
    
    return total_prob

def temperature_objective(T, target_prob, omega, s, wavelength, ion_mass, pulse_time, omega_tau=None, axis='z'):
    """Objective function for temperature extraction"""
    calc_prob = calculate_sideband_probability(T, omega, s, wavelength, ion_mass, pulse_time, omega_tau, axis)
    return abs(calc_prob - target_prob)

# Import data loading function
try:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from data_importer_methods import import_spectroscopy
except ImportError:
    print("Warning: Could not import data_importer_methods. Some functionality may be limited.")
    import_spectroscopy = None

class AnnotationDialog(QDialog):
    """Dialog for adding custom annotations to plots"""
    
    def __init__(self, plot_widget, parent=None):
        super().__init__(parent)
        self.plot_widget = plot_widget
        self.setWindowTitle("Add Annotation")
        self.setModal(True)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the annotation dialog UI"""
        layout = QVBoxLayout(self)
        
        # Form layout for parameters
        form_layout = QGridLayout()
        
        # Text input
        form_layout.addWidget(QLabel("Text:"), 0, 0)
        self.text_input = QTextEdit()
        self.text_input.setMaximumHeight(80)
        self.text_input.setPlainText("Sample annotation")
        form_layout.addWidget(self.text_input, 0, 1, 1, 2)
        
        # Position inputs
        form_layout.addWidget(QLabel("X Position:"), 1, 0)
        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(-1000, 1000)
        self.x_spin.setValue(0)
        self.x_spin.setDecimals(3)
        form_layout.addWidget(self.x_spin, 1, 1)
        
        form_layout.addWidget(QLabel("Y Position:"), 1, 2)
        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(-1000, 1000)
        self.y_spin.setValue(0.5)
        self.y_spin.setDecimals(3)
        form_layout.addWidget(self.y_spin, 1, 3)
        
        # Appearance
        form_layout.addWidget(QLabel("Font Size:"), 2, 0)
        self.fontsize_spin = QSpinBox()
        self.fontsize_spin.setRange(8, 24)
        self.fontsize_spin.setValue(12)
        form_layout.addWidget(self.fontsize_spin, 2, 1)
        
        form_layout.addWidget(QLabel("Color:"), 2, 2)
        self.color_combo = QComboBox()
        self.color_combo.addItems(['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown'])
        form_layout.addWidget(self.color_combo, 2, 3)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        ok_btn = QPushButton("Add Annotation")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        # Get current plot limits for position defaults
        if self.plot_widget and hasattr(self.plot_widget, 'ax'):
            xlim = self.plot_widget.ax.get_xlim()
            ylim = self.plot_widget.ax.get_ylim()
            self.x_spin.setValue((xlim[0] + xlim[1]) / 2)
            self.y_spin.setValue((ylim[0] + ylim[1]) / 2)
    
    def get_parameters(self):
        """Get the annotation parameters"""
        return {
            'text': self.text_input.toPlainText(),
            'x': self.x_spin.value(),
            'y': self.y_spin.value(),
            'fontsize': self.fontsize_spin.value(),
            'color': self.color_combo.currentText()
        }

class CustomLineDialog(QDialog):
    """Dialog for adding custom lines to plots"""
    
    def __init__(self, plot_widget, parent=None):
        super().__init__(parent)
        self.plot_widget = plot_widget
        self.setWindowTitle("Add Custom Line")
        self.setModal(True)
        self.line_type = 'horizontal'
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the custom line dialog UI"""
        layout = QVBoxLayout(self)
        
        # Line type selection
        type_group = QGroupBox("Line Type")
        type_layout = QVBoxLayout(type_group)
        
        self.horizontal_radio = QPushButton("Horizontal Line")
        self.horizontal_radio.setCheckable(True)
        self.horizontal_radio.setChecked(True)
        self.horizontal_radio.clicked.connect(lambda: self.set_line_type('horizontal'))
        type_layout.addWidget(self.horizontal_radio)
        
        self.vertical_radio = QPushButton("Vertical Line")
        self.vertical_radio.setCheckable(True)
        self.vertical_radio.clicked.connect(lambda: self.set_line_type('vertical'))
        type_layout.addWidget(self.vertical_radio)
        
        self.custom_radio = QPushButton("Custom Points")
        self.custom_radio.setCheckable(True)
        self.custom_radio.clicked.connect(lambda: self.set_line_type('custom'))
        type_layout.addWidget(self.custom_radio)
        
        layout.addWidget(type_group)
        
        # Parameters
        params_group = QGroupBox("Line Parameters")
        params_layout = QGridLayout(params_group)
        
        # Position/value inputs
        params_layout.addWidget(QLabel("Value/Position:"), 0, 0)
        self.value_spin = QDoubleSpinBox()
        self.value_spin.setRange(-1000, 1000)
        self.value_spin.setValue(0)
        self.value_spin.setDecimals(3)
        params_layout.addWidget(self.value_spin, 0, 1)
        
        # Custom points input
        params_layout.addWidget(QLabel("Custom Points (x,y):"), 1, 0)
        self.points_input = QTextEdit()
        self.points_input.setMaximumHeight(60)
        self.points_input.setPlainText("0,0.5\n1,0.7\n2,0.3")
        self.points_input.setEnabled(False)
        params_layout.addWidget(self.points_input, 1, 1, 1, 2)
        
        # Appearance
        params_layout.addWidget(QLabel("Label:"), 2, 0)
        self.label_input = QLineEdit("Custom Line")
        params_layout.addWidget(self.label_input, 2, 1)
        
        params_layout.addWidget(QLabel("Color:"), 3, 0)
        self.color_combo = QComboBox()
        self.color_combo.addItems(['red', 'blue', 'green', 'orange', 'purple', 'brown', 'black'])
        params_layout.addWidget(self.color_combo, 3, 1)
        
        params_layout.addWidget(QLabel("Style:"), 3, 2)
        self.style_combo = QComboBox()
        self.style_combo.addItems(['--', '-', ':', '-.'])
        params_layout.addWidget(self.style_combo, 3, 3)
        
        layout.addWidget(params_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        ok_btn = QPushButton("Add Line")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        # Set default value based on plot limits
        if self.plot_widget and hasattr(self.plot_widget, 'ax'):
            ylim = self.plot_widget.ax.get_ylim()
            self.value_spin.setValue((ylim[0] + ylim[1]) / 2)
    
    def set_line_type(self, line_type):
        """Set the line type and update UI accordingly"""
        self.line_type = line_type
        
        # Update radio button states
        self.horizontal_radio.setChecked(line_type == 'horizontal')
        self.vertical_radio.setChecked(line_type == 'vertical')
        self.custom_radio.setChecked(line_type == 'custom')
        
        # Enable/disable inputs based on type
        self.value_spin.setEnabled(line_type != 'custom')
        self.points_input.setEnabled(line_type == 'custom')
        
        # Update labels
        if line_type == 'horizontal':
            self.findChild(QLabel, "Value/Position:").setText("Y Value:")
        elif line_type == 'vertical':
            self.findChild(QLabel, "Value/Position:").setText("X Value:")
        else:
            self.findChild(QLabel, "Value/Position:").setText("Value/Position:")
    
    def get_parameters(self):
        """Get the line parameters"""
        params = {
            'type': self.line_type,
            'value': self.value_spin.value(),
            'label': self.label_input.text(),
            'color': self.color_combo.currentText(),
            'style': self.style_combo.currentText()
        }
        
        if self.line_type == 'custom':
            # Parse custom points
            try:
                points_text = self.points_input.toPlainText().strip()
                points = []
                for line in points_text.split('\n'):
                    if line.strip():
                        x, y = map(float, line.split(','))
                        points.append((x, y))
                params['points'] = points
            except:
                params['points'] = [(0, 0.5), (1, 0.7)]
        
        return params

class LegendEditDialog(QDialog):
    """Dialog for editing legend entries"""
    
    def __init__(self, plot_widget, parent=None):
        super().__init__(parent)
        self.plot_widget = plot_widget
        self.setWindowTitle("Edit Legend")
        self.setModal(True)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the legend edit dialog UI"""
        layout = QVBoxLayout(self)
        
        # Instructions
        info_label = QLabel("Select a legend entry to edit or remove:")
        layout.addWidget(info_label)
        
        # Legend entries list
        self.entries_list = QListWidget()
        self.load_legend_entries()
        layout.addWidget(self.entries_list)
        
        # Edit controls
        edit_group = QGroupBox("Edit Selected Entry")
        edit_layout = QGridLayout(edit_group)
        
        edit_layout.addWidget(QLabel("New Label:"), 0, 0)
        self.label_input = QLineEdit()
        edit_layout.addWidget(self.label_input, 0, 1)
        
        edit_layout.addWidget(QLabel("Color:"), 1, 0)
        self.color_btn = QPushButton("Choose Color")
        self.color_btn.clicked.connect(self.choose_color)
        edit_layout.addWidget(self.color_btn, 1, 1)
        
        layout.addWidget(edit_group)
        
        # Connect selection change
        self.entries_list.currentItemChanged.connect(self.on_selection_change)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        apply_btn = QPushButton("Apply Changes")
        apply_btn.clicked.connect(self.apply_changes)
        button_layout.addWidget(apply_btn)
        
        remove_btn = QPushButton("Remove Entry")
        remove_btn.clicked.connect(self.remove_entry)
        button_layout.addWidget(remove_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        self.selected_color = 'black'
    
    def load_legend_entries(self):
        """Load current legend entries"""
        self.entries_list.clear()
        
        if not hasattr(self.plot_widget, 'ax') or not self.plot_widget.ax.legend_:
            return
        
        legend = self.plot_widget.ax.legend_
        for text, line in zip(legend.get_texts(), legend.get_lines()):
            label = text.get_text()
            if label and not label.startswith('_'):
                self.entries_list.addItem(label)
    
    def on_selection_change(self, current, previous):
        """Handle legend entry selection change"""
        if current:
            self.label_input.setText(current.text())
            # Try to get color from the actual plot line
            if hasattr(self.plot_widget, 'ax'):
                for line in self.plot_widget.ax.lines:
                    if hasattr(line, 'get_label') and line.get_label() == current.text():
                        color = line.get_color()
                        self.selected_color = color
                        self.color_btn.setStyleSheet(f"background-color: {color}")
                        break
    
    def choose_color(self):
        """Open color chooser dialog"""
        color = QColorDialog.getColor()
        if color.isValid():
            self.selected_color = color.name()
            self.color_btn.setStyleSheet(f"background-color: {color.name()}")
    
    def apply_changes(self):
        """Apply changes to selected legend entry"""
        current_item = self.entries_list.currentItem()
        if not current_item or not hasattr(self.plot_widget, 'ax'):
            return
        
        old_label = current_item.text()
        new_label = self.label_input.text()
        
        # Find and update the corresponding line
        for line in self.plot_widget.ax.lines:
            if hasattr(line, 'get_label') and line.get_label() == old_label:
                line.set_label(new_label)
                line.set_color(self.selected_color)
                break
        
        # Update legend
        self.plot_widget.ax.legend()
        self.plot_widget.canvas.draw()
        
        # Refresh the list
        self.load_legend_entries()
    
    def remove_entry(self):
        """Remove selected legend entry"""
        current_item = self.entries_list.currentItem()
        if not current_item or not hasattr(self.plot_widget, 'ax'):
            return
        
        label = current_item.text()
        
        # Find and remove label from corresponding line
        for line in self.plot_widget.ax.lines:
            if hasattr(line, 'get_label') and line.get_label() == label:
                line.set_label('_nolegend_')
                break
        
        # Update legend
        self.plot_widget.ax.legend()
        self.plot_widget.canvas.draw()
        
        # Refresh the list
        self.load_legend_entries()

class InteractivePlotWidget(QWidget):
    """Enhanced plot widget with interactive capabilities"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI layout with plot"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)  # Increased margins to prevent clipping
        
        # Create matplotlib figure and canvas with better layout
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Setup plot with proper margins
        self.ax = self.figure.add_subplot(111)
        self.ax.grid(True, alpha=0.3)
        
        # Set conservative subplot parameters to prevent clipping
        self.figure.subplots_adjust(left=0.15, bottom=0.15, right=0.92, top=0.88)
        
    def plot_spectroscopy_data(self, x_data, y_data, yerr=None, fit_data=None):
        """Plot spectroscopy data with publication-ready formatting"""
        self.ax.clear()
        
        # Plot experimental data with publication-ready styling
        if yerr is not None:
            self.ax.errorbar(x_data, y_data, yerr=yerr, fmt='o', 
                           markersize=6, alpha=0.8, capsize=4, capthick=1.5,
                           markeredgewidth=1.2, color='#2E86AB', markeredgecolor='darkblue',
                           elinewidth=1.5, label='Experimental Data')
        else:
            self.ax.plot(x_data, y_data, 'o', markersize=6, alpha=0.8,
                        markeredgewidth=1.2, color='#2E86AB', markeredgecolor='darkblue',
                        label='Experimental Data')
        
        # Plot fit if available with professional styling
        if fit_data is not None:
            self.ax.plot(x_data, fit_data, '-', linewidth=3, 
                        color='#A23B72', alpha=0.9,
                        label='Lorentzian Fit')
        
        # Setup plot with publication-ready formatting
        self.ax.set_xlabel('Frequency Detuning (kHz)', fontsize=14, fontweight='bold')
        self.ax.set_ylabel('Excitation Probability', fontsize=14, fontweight='bold')
        
        # Improve tick labels
        self.ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
        self.ax.tick_params(axis='both', which='minor', width=1, length=3)
        
        # Professional legend
        legend = self.ax.legend(fontsize=12, frameon=True, fancybox=True, 
                               shadow=True, framealpha=0.9, edgecolor='black')
        legend.get_frame().set_linewidth(1.2)
        
        # Enhanced grid
        self.ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
        self.ax.set_axisbelow(True)
        
        # Improve spines
        for spine in self.ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')
        
        # Ensure proper layout to prevent clipping - use conservative margins
        self.figure.subplots_adjust(left=0.15, bottom=0.15, right=0.92, top=0.88)
        
        self.canvas.draw()

class PeakDetectionPreviewDialog(QDialog):
    """Dialog for previewing and adjusting peak detection parameters for each H5 file"""
    
    def __init__(self, file_path, filename, probe_delay, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.filename = filename
        self.probe_delay = probe_delay
        self.parent_gui = parent
        
        # Peak detection parameters
        self.height_threshold = 0.05
        self.prominence = 0.02
        self.min_distance = 10
        
        # Data storage
        self.relative_detuning = None
        self.excitation_prob = None
        self.excitation_prob_err = None
        self.detected_peaks = None
        self.peak_positions = None
        self.n_peaks = 0
        
        self.setWindowTitle(f"Peak Detection Preview - {filename}")
        self.setGeometry(100, 100, 1000, 700)
        self.setup_ui()
        self.load_and_plot_data()
        
    def setup_ui(self):
        """Setup the preview dialog UI"""
        layout = QVBoxLayout(self)
        
        # File info
        info_group = QGroupBox("File Information")
        info_layout = QGridLayout(info_group)
        
        info_layout.addWidget(QLabel("Filename:"), 0, 0)
        info_layout.addWidget(QLabel(self.filename), 0, 1)
        info_layout.addWidget(QLabel("Probe Delay:"), 0, 2)
        info_layout.addWidget(QLabel(f"{self.probe_delay:.3f} ms"), 0, 3)
        
        # Peak detection parameters
        params_group = QGroupBox("Peak Detection Parameters")
        params_layout = QGridLayout(params_group)
        
        params_layout.addWidget(QLabel("Height Threshold:"), 0, 0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.001, 1.0)
        self.height_spin.setValue(self.height_threshold)
        self.height_spin.setDecimals(3)
        self.height_spin.setSingleStep(0.005)
        self.height_spin.valueChanged.connect(self.update_peak_detection)
        params_layout.addWidget(self.height_spin, 0, 1)
        
        params_layout.addWidget(QLabel("Prominence:"), 0, 2)
        self.prominence_spin = QDoubleSpinBox()
        self.prominence_spin.setRange(0.001, 1.0)
        self.prominence_spin.setValue(self.prominence)
        self.prominence_spin.setDecimals(3)
        self.prominence_spin.setSingleStep(0.005)
        self.prominence_spin.valueChanged.connect(self.update_peak_detection)
        params_layout.addWidget(self.prominence_spin, 0, 3)
        
        params_layout.addWidget(QLabel("Min Distance (pts):"), 1, 0)
        self.distance_spin = QSpinBox()
        self.distance_spin.setRange(5, 100)
        self.distance_spin.setValue(self.min_distance)
        self.distance_spin.valueChanged.connect(self.update_peak_detection)
        params_layout.addWidget(self.distance_spin, 1, 1)
        
        # Quick preset buttons
        preset_layout = QHBoxLayout()
        strict_btn = QPushButton("Strict (0.05, 0.02)")
        strict_btn.clicked.connect(lambda: self.apply_preset(0.05, 0.02, 25))
        preset_layout.addWidget(strict_btn)
        
        normal_btn = QPushButton("Normal (0.02, 0.01)")
        normal_btn.clicked.connect(lambda: self.apply_preset(0.02, 0.01, 20))
        preset_layout.addWidget(normal_btn)
        
        loose_btn = QPushButton("Loose (0.01, 0.005)")
        loose_btn.clicked.connect(lambda: self.apply_preset(0.01, 0.005, 15))
        preset_layout.addWidget(loose_btn)
        
        preset_layout.addStretch()
        params_layout.addLayout(preset_layout, 1, 2, 1, 2)
        
        # Peak detection results
        results_group = QGroupBox("Detection Results")
        results_layout = QHBoxLayout(results_group)
        
        self.peaks_label = QLabel("Peaks: 0")
        self.peaks_label.setStyleSheet("font-weight: bold; color: blue;")
        results_layout.addWidget(self.peaks_label)
        
        self.frequencies_label = QLabel("Frequencies: -")
        results_layout.addWidget(self.frequencies_label)
        results_layout.addStretch()
        
        # Plot area
        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Set conservative subplot parameters to prevent clipping
        self.figure.subplots_adjust(left=0.15, bottom=0.15, right=0.92, top=0.85)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.accept_btn = QPushButton("Accept Parameters")
        self.accept_btn.clicked.connect(self.accept)
        self.accept_btn.setStyleSheet("background-color: green; color: white; font-weight: bold;")
        button_layout.addWidget(self.accept_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        button_layout.addStretch()
        
        # Auto-fit button
        auto_fit_btn = QPushButton("Auto-Fit Parameters")
        auto_fit_btn.clicked.connect(self.auto_fit_parameters)
        button_layout.addWidget(auto_fit_btn)
        
        # Add all to main layout
        layout.addWidget(info_group)
        layout.addWidget(params_group)
        layout.addWidget(results_group)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addLayout(button_layout)
        
    def load_and_plot_data(self):
        """Load the H5 file data and create initial plot"""
        try:
            if self.parent_gui and hasattr(self.parent_gui, 'log_message'):
                self.parent_gui.log_message(f"Loading preview for {self.filename}")
            
            # Import the spectroscopy data
            from data_importer_methods import import_spectroscopy
            detuning, relative_detuning, excitation_prob, excitation_prob_err, freq_center_MHz = import_spectroscopy(self.file_path)
            
            # Store the data
            self.relative_detuning = np.array(relative_detuning).flatten()
            self.excitation_prob = np.array(excitation_prob).flatten()
            self.excitation_prob_err = np.array(excitation_prob_err).flatten()
            self.freq_center_MHz = freq_center_MHz
            
            # Initial peak detection and plot
            self.update_peak_detection()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load file:\n{str(e)}")
            self.reject()
    
    def update_peak_detection(self):
        """Update peak detection with current parameters and refresh plot"""
        try:
            # Get current parameters
            self.height_threshold = self.height_spin.value()
            self.prominence = self.prominence_spin.value()
            self.min_distance = self.distance_spin.value()
            
            # Perform peak detection
            peaks, properties = find_peaks(
                self.excitation_prob, 
                height=self.height_threshold,
                prominence=self.prominence,
                distance=self.min_distance
            )
            
            self.detected_peaks = peaks
            self.peak_positions = self.relative_detuning[peaks] if len(peaks) > 0 else np.array([])
            self.n_peaks = len(peaks)
            
            # Update results display
            self.peaks_label.setText(f"Peaks: {self.n_peaks}")
            
            if self.n_peaks > 0:
                freq_str = ", ".join([f"{pos/1000:.1f}" for pos in self.peak_positions[:5]])  # Show first 5
                if self.n_peaks > 5:
                    freq_str += "..."
                self.frequencies_label.setText(f"Frequencies (kHz): {freq_str}")
                
                # Set accept button color based on number of peaks
                if self.n_peaks >= 2:
                    self.accept_btn.setStyleSheet("background-color: green; color: white; font-weight: bold;")
                    self.accept_btn.setText(f"Accept Parameters ({self.n_peaks} peaks)")
                else:
                    self.accept_btn.setStyleSheet("background-color: orange; color: white; font-weight: bold;")
                    self.accept_btn.setText(f"Accept Parameters ({self.n_peaks} peaks - may be insufficient)")
            else:
                self.frequencies_label.setText("Frequencies: None detected")
                self.accept_btn.setStyleSheet("background-color: red; color: white; font-weight: bold;")
                self.accept_btn.setText("Accept Parameters (0 peaks)")
            
            # Update plot
            self.plot_data_with_peaks()
            
        except Exception as e:
            if self.parent_gui and hasattr(self.parent_gui, 'log_message'):
                self.parent_gui.log_message(f"Error in peak detection: {str(e)}")
            self.peaks_label.setText("Peaks: Error")
            self.frequencies_label.setText("Frequencies: Error")
    
    def plot_data_with_peaks(self):
        """Plot the data with detected peaks highlighted with publication-ready formatting"""
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Convert to kHz for plotting
            x_data_khz = self.relative_detuning / 1000
            
            # Plot original data with error bars using professional styling
            ax.errorbar(x_data_khz, self.excitation_prob, yerr=self.excitation_prob_err,
                       fmt='o', markersize=5, alpha=0.8, capsize=4, capthick=1.5,
                       markeredgewidth=1.2, color='#2E86AB', markeredgecolor='#1A5276',
                       elinewidth=1.5, label=f'Spectroscopy Data ({len(x_data_khz)} pts)')
            
            # Plot detected peaks with enhanced styling
            if self.n_peaks > 0:
                peak_freqs_khz = self.peak_positions / 1000
                peak_probs = self.excitation_prob[self.detected_peaks]
                
                ax.plot(peak_freqs_khz, peak_probs, 's', markersize=12, 
                       markeredgewidth=2.5, markerfacecolor='#E74C3C', markeredgecolor='#A93226',
                       label=f'Detected Peaks ({self.n_peaks})', zorder=5)
                
                # Add peak labels with enhanced styling
                for i, (freq, prob) in enumerate(zip(peak_freqs_khz, peak_probs)):
                    ax.annotate(f'{i+1}', xy=(freq, prob), xytext=(8, 15), 
                               textcoords='offset points', fontsize=11, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='#F7DC6F', 
                                       alpha=0.9, edgecolor='black'), zorder=6)
            
            # Add threshold lines with professional styling
            ax.axhline(y=self.height_threshold, color='#F39C12', linestyle='--', alpha=0.8,
                      linewidth=2.5, label=f'Height threshold: {self.height_threshold:.3f}')
            
            # Professional plot customization
            ax.set_xlabel('Frequency Detuning (kHz)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Excitation Probability', fontsize=14, fontweight='bold')
            ax.set_title(f'Peak Detection Preview: {self.filename}\n'
                        f'Probe Delay: {self.probe_delay:.3f} ms, Center: {self.freq_center_MHz:.3f} MHz',
                        fontsize=15, fontweight='bold', pad=20)
            
            # Improve tick labels
            ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
            ax.tick_params(axis='both', which='minor', width=1, length=3)
            
            # Professional legend
            legend = ax.legend(fontsize=11, frameon=True, fancybox=True, 
                             shadow=True, framealpha=0.95, edgecolor='black')
            legend.get_frame().set_linewidth(1.2)
            
            # Enhanced grid
            ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
            ax.set_axisbelow(True)
            
            # Improve spines
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color('black')
            
            # Add parameter info as text with enhanced styling
            param_text = f'Height: {self.height_threshold:.3f}, Prominence: {self.prominence:.3f}, Distance: {self.min_distance}'
            ax.text(0.02, 0.98, param_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F4FD', alpha=0.9, edgecolor='black'),
                   verticalalignment='top', fontsize=10, fontweight='bold')
            
            # Ensure proper layout to prevent clipping - use conservative margins
            self.figure.subplots_adjust(left=0.15, bottom=0.15, right=0.92, top=0.85)
            self.canvas.draw()
            
        except Exception as e:
            if self.parent_gui and hasattr(self.parent_gui, 'log_message'):
                self.parent_gui.log_message(f"Error plotting preview: {str(e)}")
    
    def apply_preset(self, height, prominence, distance):
        """Apply preset parameter values"""
        self.height_spin.setValue(height)
        self.prominence_spin.setValue(prominence)
        self.distance_spin.setValue(distance)
        # update_peak_detection will be called automatically by valueChanged signals
    
    def auto_fit_parameters(self):
        """Automatically determine good peak detection parameters"""
        try:
            if self.parent_gui and hasattr(self.parent_gui, 'log_message'):
                self.parent_gui.log_message(f"Auto-fitting parameters for {self.filename}")
            
            # Strategy: Try different parameter combinations and pick the one that gives 2-6 peaks
            height_values = [0.01, 0.02, 0.03, 0.05, 0.07]
            prominence_values = [0.005, 0.01, 0.015, 0.02, 0.03]
            distance_values = [10, 15, 20, 25, 30]
            
            best_params = None
            best_score = -1
            
            for height in height_values:
                for prominence in prominence_values:
                    for distance in distance_values:
                        try:
                            peaks, _ = find_peaks(
                                self.excitation_prob,
                                height=height,
                                prominence=prominence,
                                distance=distance
                            )
                            n_peaks = len(peaks)
                            
                            # Scoring: prefer 2-6 peaks, penalize too few or too many
                            if 2 <= n_peaks <= 6:
                                score = 10
                            elif n_peaks == 1:
                                score = 5
                            elif n_peaks == 7:
                                score = 7
                            else:
                                score = 1
                            
                            # Bonus for reasonable parameter values
                            if 0.01 <= height <= 0.05:
                                score += 2
                            if 0.005 <= prominence <= 0.02:
                                score += 2
                            
                            if score > best_score:
                                best_score = score
                                best_params = (height, prominence, distance, n_peaks)
                        except:
                            continue
            
            if best_params:
                height, prominence, distance, n_peaks = best_params
                self.apply_preset(height, prominence, distance)
                
                if self.parent_gui and hasattr(self.parent_gui, 'log_message'):
                    self.parent_gui.log_message(f"Auto-fit found: {n_peaks} peaks with height={height:.3f}, prominence={prominence:.3f}")
                
                QMessageBox.information(self, "Auto-Fit Complete", 
                                      f"Found optimal parameters:\n"
                                      f"Height: {height:.3f}\n"
                                      f"Prominence: {prominence:.3f}\n"
                                      f"Distance: {distance}\n"
                                      f"Detected peaks: {n_peaks}")
            else:
                QMessageBox.warning(self, "Auto-Fit Failed", 
                                  "Could not find suitable parameters automatically.\n"
                                  "Please adjust parameters manually.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Auto-fit failed:\n{str(e)}")
    
    def get_parameters(self):
        """Get the current peak detection parameters"""
        return {
            'height': self.height_threshold,
            'prominence': self.prominence,
            'distance': self.min_distance,
            'n_peaks': self.n_peaks,
            'peak_positions': self.peak_positions.copy() if self.n_peaks > 0 else np.array([])
        }

class SpectroscopyFitGUI(QMainWindow):
    """Main GUI class for spectroscopy fitting"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ion Trap Spectroscopy Fitting Tool - PyQt6 Edition")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data storage
        self.X = None
        self.y = None
        self.yerr = None
        self.heating_rate_files = []  # List of dictionaries with file info and parameters
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_data_fitting_tab()
        self.create_thermal_tab()
        self.create_heating_rate_tab()
        
        # Status bar
        self.statusBar().showMessage("Ready - PyQt6 Edition")
        
    def create_data_fitting_tab(self):
        """Create the data fitting tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # File loading
        file_group = QGroupBox("File Selection")
        file_layout = QHBoxLayout(file_group)
        
        load_button = QPushButton("Load Data File")
        load_button.clicked.connect(self.load_file)
        file_layout.addWidget(load_button)
        
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: gray;")
        file_layout.addWidget(self.file_label)
        file_layout.addStretch()
        
        # Status display
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(150)
        self.status_text.setFont(QFont("Courier", 9))
        status_layout.addWidget(self.status_text)
        
        # Analysis controls
        controls_group = QGroupBox("Peak Detection & Fitting")
        controls_layout = QHBoxLayout(controls_group)
        
        detect_peaks_button = QPushButton("Detect Peaks")
        detect_peaks_button.clicked.connect(self.detect_peaks)
        controls_layout.addWidget(detect_peaks_button)
        
        fit_peaks_button = QPushButton("Fit Peaks")
        fit_peaks_button.clicked.connect(self.fit_peaks)
        controls_layout.addWidget(fit_peaks_button)
        
        plot_fit_button = QPushButton("Plot Fit")
        plot_fit_button.clicked.connect(self.plot_fit)
        controls_layout.addWidget(plot_fit_button)
        
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset_analysis)
        controls_layout.addWidget(reset_button)
        
        controls_layout.addStretch()
        
        # Plot area
        self.plot_widget = InteractivePlotWidget(self)
        
        # Add to main layout
        layout.addWidget(file_group)
        layout.addWidget(status_group)
        layout.addWidget(controls_group)
        layout.addWidget(self.plot_widget)
        
        self.tab_widget.addTab(tab, "Data & Fitting")
        
    def create_thermal_tab(self):
        """Create thermal analysis tab for individual spectroscopy measurements"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # File loading section
        file_group = QGroupBox("Spectroscopy Data File")
        file_layout = QHBoxLayout(file_group)
        
        load_thermal_button = QPushButton("Load Spectroscopy File")
        load_thermal_button.clicked.connect(self.load_thermal_file)
        file_layout.addWidget(load_thermal_button)
        
        self.thermal_file_label = QLabel("No file selected")
        self.thermal_file_label.setStyleSheet("color: gray;")
        file_layout.addWidget(self.thermal_file_label)
        file_layout.addStretch()
        
        # Analysis parameters section
        thermal_params_group = QGroupBox("Thermal Analysis Parameters (Ba+ Ion Trap)")
        thermal_params_layout = QGridLayout(thermal_params_group)
        
        # Trap frequency parameters
        thermal_params_layout.addWidget(QLabel("X-axis frequency (kHz):"), 0, 0)
        self.thermal_freq_x_spin = QDoubleSpinBox()
        self.thermal_freq_x_spin.setRange(100, 5000)
        self.thermal_freq_x_spin.setValue(330)  # Correct Ba+ trap frequency
        self.thermal_freq_x_spin.setSuffix(" kHz")
        thermal_params_layout.addWidget(self.thermal_freq_x_spin, 0, 1)
        
        thermal_params_layout.addWidget(QLabel("Y-axis frequency (kHz):"), 0, 2)
        self.thermal_freq_y_spin = QDoubleSpinBox()
        self.thermal_freq_y_spin.setRange(100, 5000)
        self.thermal_freq_y_spin.setValue(360)  # Correct Ba+ trap frequency
        self.thermal_freq_y_spin.setSuffix(" kHz")
        thermal_params_layout.addWidget(self.thermal_freq_y_spin, 0, 3)
        
        thermal_params_layout.addWidget(QLabel("Z-axis frequency (kHz):"), 1, 0)
        self.thermal_freq_z_spin = QDoubleSpinBox()
        self.thermal_freq_z_spin.setRange(100, 2000)
        self.thermal_freq_z_spin.setValue(270)  # Correct Ba+ trap frequency
        self.thermal_freq_z_spin.setSuffix(" kHz")
        thermal_params_layout.addWidget(self.thermal_freq_z_spin, 1, 1)
        
        # Ion parameters
        thermal_params_layout.addWidget(QLabel("Ion mass (amu):"), 1, 2)
        self.thermal_ion_mass_spin = QDoubleSpinBox()
        self.thermal_ion_mass_spin.setRange(1, 300)
        self.thermal_ion_mass_spin.setValue(137.905247)  # Ba+ (Barium ion)
        self.thermal_ion_mass_spin.setSuffix(" amu")
        self.thermal_ion_mass_spin.setDecimals(6)
        thermal_params_layout.addWidget(self.thermal_ion_mass_spin, 1, 3)
        
        # Laser parameters
        thermal_params_layout.addWidget(QLabel("Laser wavelength (nm):"), 2, 0)
        self.thermal_wavelength_spin = QDoubleSpinBox()
        self.thermal_wavelength_spin.setRange(350, 2000)
        self.thermal_wavelength_spin.setValue(1762.0)  # Ba+ spectroscopy (to match original GUI)
        self.thermal_wavelength_spin.setSuffix(" nm")
        self.thermal_wavelength_spin.setDecimals(1)
        thermal_params_layout.addWidget(self.thermal_wavelength_spin, 2, 1)
        
        thermal_params_layout.addWidget(QLabel("Pulse area (π units):"), 2, 2)
        self.thermal_pulse_area_spin = QDoubleSpinBox()
        self.thermal_pulse_area_spin.setRange(0.1, 5.0)
        self.thermal_pulse_area_spin.setValue(2.37)  # As specified for the test
        self.thermal_pulse_area_spin.setSuffix(" π")
        self.thermal_pulse_area_spin.setDecimals(2)
        thermal_params_layout.addWidget(self.thermal_pulse_area_spin, 2, 3)
        
        # Analysis controls
        thermal_controls = QHBoxLayout()
        
        analyze_thermal_button = QPushButton("Analyze Temperature")
        analyze_thermal_button.clicked.connect(self.analyze_thermal_temperature)
        thermal_controls.addWidget(analyze_thermal_button)
        
        plot_thermal_model_button = QPushButton("Plot Thermal Model")
        plot_thermal_model_button.clicked.connect(self.plot_thermal_model)
        thermal_controls.addWidget(plot_thermal_model_button)
        
        export_thermal_button = QPushButton("Export Thermal Results")
        export_thermal_button.clicked.connect(self.export_thermal_results)
        thermal_controls.addWidget(export_thermal_button)
        
        thermal_controls.addStretch()
        
        # Results section
        self.thermal_results_text = QTextEdit()
        self.thermal_results_text.setFont(QFont("Courier", 9))
        self.thermal_results_text.setMaximumHeight(200)
        
        # Plot area
        self.thermal_plot_widget = InteractivePlotWidget(self)
        
        # Add all sections to main layout
        layout.addWidget(file_group)
        layout.addWidget(thermal_params_group)
        layout.addLayout(thermal_controls)
        layout.addWidget(self.thermal_results_text)
        layout.addWidget(self.thermal_plot_widget)
        
        # Initialize data storage
        self.thermal_data = None
        self.thermal_analysis_results = {}
        
        self.tab_widget.addTab(tab, "Thermal Analysis")
        
    def create_heating_rate_tab(self):
        """Create heating rate analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # File management
        file_group = QGroupBox("Heating Rate Files")
        file_layout = QVBoxLayout(file_group)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        add_file_button = QPushButton("Add H5 File")
        add_file_button.clicked.connect(self.add_heating_rate_file)
        controls_layout.addWidget(add_file_button)
        
        remove_file_button = QPushButton("Remove Selected")
        remove_file_button.clicked.connect(self.remove_heating_rate_file)
        controls_layout.addWidget(remove_file_button)
        
        clear_files_button = QPushButton("Clear All")
        clear_files_button.clicked.connect(self.clear_heating_rate_files)
        controls_layout.addWidget(clear_files_button)
        
        controls_layout.addStretch()
        
        file_layout.addLayout(controls_layout)
        
        # File list
        self.heating_rate_file_list = QListWidget()
        self.heating_rate_file_list.setMaximumHeight(120)
        file_layout.addWidget(self.heating_rate_file_list)
        
        # Analysis controls
        analysis_layout = QHBoxLayout()
        
        analyze_button = QPushButton("Analyze Heating Rates")
        analyze_button.clicked.connect(self.analyze_heating_rates)
        analysis_layout.addWidget(analyze_button)
        
        fit_button = QPushButton("Fit Heating Rate")
        fit_button.clicked.connect(self.fit_heating_rate)
        analysis_layout.addWidget(fit_button)
        
        analysis_layout.addStretch()
        
        # Results
        self.heating_results_text = QTextEdit()
        self.heating_results_text.setFont(QFont("Courier", 9))
        self.heating_results_text.setMaximumHeight(200)
        
        # Heating plot area
        self.heating_plot_widget = InteractivePlotWidget(self)
        
        # Add to main layout
        layout.addWidget(file_group)
        layout.addLayout(analysis_layout)
        layout.addWidget(self.heating_results_text)
        layout.addWidget(self.heating_plot_widget)
        
        self.tab_widget.addTab(tab, "Heating Rate Analysis")
    
    # Helper methods
    def log_message(self, message):
        """Add a message to the status log"""
        self.status_text.append(message)
        self.status_text.ensureCursorVisible()
        QApplication.processEvents()
    
    def lorentzian(self, x, amplitude, center, width, y0=0):
        """Lorentzian function for fitting"""
        return y0 + amplitude / (1 + ((x - center) / width) ** 2)
    
    def multi_lorentzian(self, x, *params):
        """Multi-peak Lorentzian function for simultaneous fitting"""
        n_peaks = (len(params) - 1) // 3
        x0s = params[:n_peaks]
        widths = params[n_peaks:2*n_peaks]
        amplitudes = params[2*n_peaks:3*n_peaks]
        y0 = params[-1]
        
        y = np.zeros_like(np.array(x, dtype=float))
        for i in range(n_peaks):
            y += amplitudes[i] / (1 + ((x - x0s[i]) / widths[i]) ** 2)
        return y + y0
    
    def detect_peaks_generic(self, x_data, y_data, threshold=0.5, min_distance=10, prominence=0.01):
        """Generic peak detection method"""
        try:
            peaks, properties = find_peaks(
                y_data, 
                height=threshold,
                prominence=prominence,
                distance=min_distance
            )
            return peaks
        except Exception as e:
            if hasattr(self, 'log_message'):
                self.log_message(f"Peak detection error: {str(e)}")
            return np.array([])
    
    def fit_lorentzian_peaks_generic(self, x_data, y_data, peaks):
        """Generic multi-peak Lorentzian fitting method with robust error estimation"""
        try:
            if len(peaks) == 0:
                return []
            
            n_peaks = len(peaks)
            
            # Estimate initial parameters
            x0_inits = [x_data[p] for p in peaks if 0 <= p < len(x_data)]
            A_inits = [y_data[p] for p in peaks if 0 <= p < len(x_data)]
            gamma_inits = [np.abs(x_data.max() - x_data.min()) / (10 * n_peaks) for _ in range(n_peaks)]
            y0_init = 0.0
            
            # Ensure we have valid initial parameters
            if len(x0_inits) != n_peaks:
                return []
            
            p0 = x0_inits + gamma_inits + A_inits + [y0_init]
            
            # Estimate data uncertainties (if not provided)
            yerr = np.ones_like(y_data) * np.sqrt(np.mean(y_data) + 0.001)
            
            # Define residuals function
            def residuals(params, x, y, yerr):
                return (self.multi_lorentzian(x, *params) - y) / yerr
            
            # Set up bounds
            x_min, x_max = x_data.min(), x_data.max()
            lower_bounds = ([x_min] * n_peaks + [10.0] * n_peaks + 
                          [0.0] * n_peaks + [-0.01])
            upper_bounds = ([x_max] * n_peaks + [abs(x_max - x_min)] * n_peaks + 
                          [1.0] * n_peaks + [0.01])
            
            # Perform robust fitting
            result = least_squares(
                residuals, p0, args=(x_data, y_data, yerr),
                bounds=(lower_bounds, upper_bounds),
                max_nfev=10000
            )
            
            if not result.success:
                self.log_message(f"Multi-peak fitting failed: {result.message}")
                # Fall back to individual peak fitting
                return self._fit_individual_peaks(x_data, y_data, peaks)
            
            # Calculate uncertainties
            try:
                residual_variance = np.sum(result.fun**2) / max(1, len(x_data) - len(result.x))
                J = result.jac
                cov = residual_variance * np.linalg.inv(J.T @ J)
                perr = np.sqrt(np.diag(cov))
            except:
                perr = np.zeros_like(result.x)
            
            # Extract parameters
            popt = result.x
            peak_centers = popt[:n_peaks]
            peak_widths = popt[n_peaks:2*n_peaks]
            peak_amplitudes = popt[2*n_peaks:3*n_peaks]
            baseline = popt[-1]
            
            # Prepare results
            results = []
            for i in range(n_peaks):
                amp_err = perr[2*n_peaks + i] if len(perr) > 2*n_peaks + i else 0.0
                center_err = perr[i] if len(perr) > i else 0.0
                width_err = perr[n_peaks + i] if len(perr) > n_peaks + i else 0.0
                
                results.append({
                    'amplitude': peak_amplitudes[i],
                    'center': peak_centers[i],
                    'width': peak_widths[i],
                    'y0': baseline,
                    'amplitude_err': amp_err,
                    'center_err': center_err,
                    'width_err': width_err,
                    'fit_quality': result.cost / len(x_data)
                })
            
            # Store global fit results for plotting
            self._last_fit_result = {
                'popt': popt,
                'perr': perr,
                'n_peaks': n_peaks,
                'x_data': x_data,
                'y_data': y_data,
                'success': True
            }
            
            return results
            
        except Exception as e:
            if hasattr(self, 'log_message'):
                self.log_message(f"Multi-peak fitting error: {str(e)}")
            return self._fit_individual_peaks(x_data, y_data, peaks)
    
    def _fit_individual_peaks(self, x_data, y_data, peaks):
        """Fallback method: fit individual peaks"""
        results = []
        for peak_idx in peaks:
            if 0 <= peak_idx < len(x_data):
                center_guess = x_data[peak_idx]
                amplitude_guess = y_data[peak_idx]
                width_guess = abs(x_data.max() - x_data.min()) / (20 * len(peaks))
                
                # Define a small window around the peak for fitting
                window_size = min(20, len(x_data) // 4)
                start_idx = max(0, peak_idx - window_size)
                end_idx = min(len(x_data), peak_idx + window_size)
                
                x_fit = x_data[start_idx:end_idx]
                y_fit = y_data[start_idx:end_idx]
                
                try:
                    popt, pcov = curve_fit(
                        self.lorentzian, x_fit, y_fit,
                        p0=[amplitude_guess, center_guess, width_guess, 0],
                        bounds=([-np.inf, -np.inf, 0.1, -np.inf], 
                               [np.inf, np.inf, np.inf, np.inf]),
                        maxfev=1000
                    )
                    
                    # Calculate uncertainties
                    perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros(4)
                    
                    results.append({
                        'amplitude': popt[0],
                        'center': popt[1], 
                        'width': popt[2],
                        'y0': popt[3],
                        'amplitude_err': perr[0],
                        'center_err': perr[1],
                        'width_err': perr[2],
                        'fit_quality': 1.0  # Default value for individual fits
                    })
                except:
                    # Fallback: just use peak position
                    results.append({
                        'amplitude': amplitude_guess,
                        'center': center_guess,
                        'width': width_guess,
                        'y0': 0,
                        'amplitude_err': 0.0,
                        'center_err': 0.0,
                        'width_err': 0.0,
                        'fit_quality': 999.0  # High value indicates poor fit
                    })
        
        return results
    
    # Core functionality methods
    def load_file(self):
        """Load a spectroscopy data file"""
        if import_spectroscopy is None:
            QMessageBox.warning(self, "Import Error", 
                              "Data import functionality not available.")
            return
        
        initial_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "used_data-march")
        if not os.path.exists(initial_dir):
            initial_dir = os.path.dirname(__file__)
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Spectroscopy Data File", initial_dir,
            "HDF5 files (*.h5);;All files (*.*)"
        )
        
        if not file_path:
            return
        
        filename = os.path.basename(file_path)
        
        try:
            self.log_message(f"Loading file: {filename}")
            
            # Load spectroscopy data
            detuning, relative_detuning, excitation_prob, excitation_prob_err, freq_center_MHz = import_spectroscopy(file_path)
            
            # Store data
            self.X = np.array(relative_detuning) / 1000  # Convert to kHz
            self.y = np.array(excitation_prob)
            self.yerr = np.array(excitation_prob_err)
            
            # Update UI
            self.file_label.setText(f"Loaded: {filename}")
            self.file_label.setStyleSheet("color: green;")
            
            self.log_message(f"File loaded successfully!")
            self.log_message(f"Data points: {len(self.X)}")
            
            # Plot data
            self.plot_widget.plot_spectroscopy_data(self.X, self.y, self.yerr)
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{str(e)}")
            self.log_message(f"Error loading {filename}: {str(e)}")
    
    # Data fitting tab methods
    def detect_peaks(self):
        """Detect peaks in the loaded data"""
        if self.X is None:
            QMessageBox.warning(self, "No Data", "Please load data first!")
            return
        
        try:
            self.log_message("Detecting peaks...")
            
            # Detect peaks using the generic method
            peaks = self.detect_peaks_generic(self.X * 1000, self.y, threshold=0.02)  # Convert to Hz
            
            if len(peaks) == 0:
                QMessageBox.warning(self, "No Peaks", "No peaks detected. Try adjusting threshold.")
                return
            
            self.peaks = peaks
            self.log_message(f"Detected {len(peaks)} peaks")
            
            # Plot peaks on the data
            self.plot_widget.plot_spectroscopy_data(self.X, self.y, self.yerr)
            
            # Mark peaks on the plot
            peak_freqs = [self.X[p] for p in peaks]
            peak_amps = [self.y[p] for p in peaks]
            
            self.plot_widget.ax.scatter(peak_freqs, peak_amps, color='red', s=50, 
                                      marker='x', linewidth=2, label='Detected Peaks')
            self.plot_widget.ax.legend()
            
            # Ensure proper layout to prevent clipping
            self.plot_widget.figure.tight_layout(pad=1.0)
            self.plot_widget.canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Detection Error", f"Failed to detect peaks:\n{str(e)}")
            self.log_message(f"Peak detection error: {str(e)}")
    
    def fit_peaks(self):
        """Fit Lorentzian peaks to detected peaks"""
        if self.X is None:
            QMessageBox.warning(self, "No Data", "Please load data first!")
            return
        
        if not hasattr(self, 'peaks') or len(self.peaks) == 0:
            QMessageBox.warning(self, "No Peaks", "Please detect peaks first!")
            return
        
        try:
            self.log_message("Fitting peaks...")
            
            # Fit peaks using the improved generic method
            x_data = self.X * 1000  # Convert to Hz
            fit_results = self.fit_lorentzian_peaks_generic(x_data, self.y, self.peaks)
            
            if len(fit_results) == 0:
                QMessageBox.warning(self, "Fit Failed", "Could not fit any peaks.")
                return
            
            self.fit_results = fit_results
            self.log_message(f"Successfully fitted {len(fit_results)} peaks")
            
            # Display fit results
            results_text = "PEAK FITTING RESULTS\n" + "="*30 + "\n\n"
            
            for i, result in enumerate(fit_results):
                center = result['center'] / 1000  # Convert back to kHz
                amplitude = result['amplitude']
                width = result['width'] / 1000  # Convert back to kHz
                amp_err = result.get('amplitude_err', 0)
                fit_quality = result.get('fit_quality', 0)
                
                results_text += f"Peak {i+1}:\n"
                results_text += f"  Center: {center:.2f} kHz\n"
                results_text += f"  Amplitude: {amplitude:.4f} ± {amp_err:.4f}\n"
                results_text += f"  Width: {width:.2f} kHz\n"
                results_text += f"  Fit Quality: {fit_quality:.6f}\n\n"
            
            # Check if multi-peak fitting was successful
            if hasattr(self, '_last_fit_result') and self._last_fit_result.get('success', False):
                results_text += "Multi-peak simultaneous fitting: ✅ SUCCESS\n"
                results_text += f"Global fit with {self._last_fit_result['n_peaks']} peaks\n"
            else:
                results_text += "Multi-peak fitting: ❌ FAILED (using individual fits)\n"
            
            self.status_text.setText(results_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Fit Error", f"Failed to fit peaks:\n{str(e)}")
            self.log_message(f"Peak fitting error: {str(e)}")
    
    def plot_fit(self):
        """Plot the fitted curves with the data using publication-ready formatting"""
        if not hasattr(self, 'fit_results') or len(self.fit_results) == 0:
            QMessageBox.warning(self, "No Fit", "Please fit peaks first!")
            return
        
        try:
            self.log_message("Plotting fitted curves...")
            
            # Plot experimental data with enhanced styling
            self.plot_widget.ax.clear()
            self.plot_widget.ax.errorbar(self.X, self.y, yerr=self.yerr, fmt='o', 
                                        markersize=6, capsize=5, capthick=1.5,
                                        markeredgewidth=1.2, color='#2E86AB', 
                                        markeredgecolor='#1A5276', elinewidth=1.5,
                                        label='Experimental Data')
            
            x_data = self.X * 1000  # Hz
            x_fit = np.linspace(x_data.min(), x_data.max(), 1000)
            
            # Plot fitted curves with professional styling
            if hasattr(self, '_last_fit_result') and self._last_fit_result.get('success', False):
                # Multi-peak fit available with enhanced styling
                fit_data = self._last_fit_result
                y_total = self.multi_lorentzian(x_fit, *fit_data['popt'])
                self.plot_widget.ax.plot(x_fit / 1000, y_total, '-', 
                                       color='#A23B72', linewidth=3, alpha=0.9,
                                       label='Total Fit')
                
                # Plot individual peaks with professional colors
                n_peaks = fit_data['n_peaks']
                peak_centers = fit_data['popt'][:n_peaks]
                peak_widths = fit_data['popt'][n_peaks:2*n_peaks]
                peak_amplitudes = fit_data['popt'][2*n_peaks:3*n_peaks]
                baseline = fit_data['popt'][-1]
                
                colors = ['#E74C3C', '#27AE60', '#3498DB', '#F39C12', '#9B59B6']
                for i in range(n_peaks):
                    y_peak = (peak_amplitudes[i] / (1 + ((x_fit - peak_centers[i]) / peak_widths[i]) ** 2)) + baseline
                    color = colors[i % len(colors)]
                    self.plot_widget.ax.plot(x_fit / 1000, y_peak, '--', alpha=0.8, 
                                           color=color, linewidth=2.5, label=f'Peak {i+1}')
            else:
                # Individual fits with enhanced styling
                colors = ['#E74C3C', '#27AE60', '#3498DB', '#F39C12', '#9B59B6']
                for i, result in enumerate(self.fit_results):
                    y_peak = self.lorentzian(x_fit, result['amplitude'], result['center'], 
                                           result['width'], result['y0'])
                    color = colors[i % len(colors)]
                    self.plot_widget.ax.plot(x_fit / 1000, y_peak, '--', alpha=0.8, 
                                           color=color, linewidth=2.5, label=f'Peak {i+1}')
            
            # Professional formatting
            self.plot_widget.ax.set_xlabel('Relative Detuning (kHz)', fontsize=14, fontweight='bold')
            self.plot_widget.ax.set_ylabel('Excitation Probability', fontsize=14, fontweight='bold')
            self.plot_widget.ax.set_title('Spectroscopy Data with Fitted Peaks', 
                                         fontsize=16, fontweight='bold', pad=20)
            
            # Improve tick labels
            self.plot_widget.ax.tick_params(axis='both', which='major', labelsize=12, 
                                           width=1.5, length=6)
            self.plot_widget.ax.tick_params(axis='both', which='minor', width=1, length=3)
            
            # Professional legend
            legend = self.plot_widget.ax.legend(fontsize=11, frameon=True, fancybox=True, 
                                               shadow=True, framealpha=0.95, edgecolor='black')
            legend.get_frame().set_linewidth(1.2)
            
            # Enhanced grid
            self.plot_widget.ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
            self.plot_widget.ax.set_axisbelow(True)
            
            # Improve spines
            for spine in self.plot_widget.ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color('black')
            
            # Ensure proper layout to prevent clipping - use conservative margins
            self.plot_widget.figure.subplots_adjust(left=0.15, bottom=0.15, right=0.92, top=0.85)
            
            self.plot_widget.canvas.draw()
            self.log_message("Fitted curves plotted successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to plot fit:\n{str(e)}")
            self.log_message(f"Plot fit error: {str(e)}")
    
    def reset_analysis(self):
        """Reset the analysis and clear results"""
        try:
            # Clear analysis results
            if hasattr(self, 'peaks'):
                delattr(self, 'peaks')
            if hasattr(self, 'fit_results'):
                delattr(self, 'fit_results')
            if hasattr(self, '_last_fit_result'):
                delattr(self, '_last_fit_result')
            
            # Clear status text
            self.status_text.clear()
            
            # Replot original data if available
            if self.X is not None:
                self.plot_widget.plot_spectroscopy_data(self.X, self.y, self.yerr)
            
            self.log_message("Analysis reset - ready for new analysis")
            
        except Exception as e:
            self.log_message(f"Reset error: {str(e)}")
    
    # Heating rate analysis methods
    def add_heating_rate_file(self):
        """Add H5 file to heating rate analysis with peak detection preview"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select H5 File for Heating Rate Analysis", 
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "used_data-march"),
            "HDF5 files (*.h5);;All files (*.*)"
        )
        
        if not file_path:
            return
        
        filename = os.path.basename(file_path)
        
        # Get probe delay from user
        probe_delay, ok = QInputDialog.getDouble(
            self, "Probe Delay", f"Enter probe delay for {filename} (ms):", 
            0.0, 0.0, 1000.0, 3
        )
        
        if not ok:
            return
        
        try:
            # Show peak detection preview dialog
            preview_dialog = PeakDetectionPreviewDialog(file_path, filename, probe_delay, self)
            
            if preview_dialog.exec() == QDialog.DialogCode.Accepted:
                # Get the optimized parameters
                peak_params = preview_dialog.get_parameters()
                
                # Add file with per-file parameters
                file_entry = {
                    'file_path': file_path,
                    'filename': filename,
                    'probe_delay': probe_delay,
                    'peak_detection_params': peak_params
                }
                
                self.heating_rate_files.append(file_entry)
                self.update_heating_rate_file_list()
                self.log_message(f"Added {filename} with {peak_params['n_peaks']} peaks detected")
            else:
                self.log_message(f"File addition cancelled for {filename}")
                
        except Exception as e:
            self.log_message(f"Error adding file: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to add file:\n{str(e)}")
    
    def remove_heating_rate_file(self):
        """Remove selected file from heating rate analysis"""
        current_row = self.heating_rate_file_list.currentRow()
        if current_row >= 0 and current_row < len(self.heating_rate_files):
            file_entry = self.heating_rate_files.pop(current_row)
            self.update_heating_rate_file_list()
            self.log_message(f"Removed {file_entry['filename']}")
    
    def clear_heating_rate_files(self):
        """Clear all files from heating rate analysis"""
        self.heating_rate_files.clear()
        self.update_heating_rate_file_list()
        self.heating_results_text.clear()
        self.log_message("Cleared all heating rate files")
    
    def update_heating_rate_file_list(self):
        """Update the file list display"""
        self.heating_rate_file_list.clear()
        for file_entry in self.heating_rate_files:
            item_text = f"{file_entry['filename']} (delay: {file_entry['probe_delay']:.3f} ms, peaks: {file_entry['peak_detection_params']['n_peaks']})"
            self.heating_rate_file_list.addItem(item_text)
    
    def analyze_heating_rates(self):
        """Analyze heating rates from the loaded files using per-file peak detection"""
        if not self.heating_rate_files:
            QMessageBox.warning(self, "No Files", "Please add H5 files first.")
            return
        
        self.log_message(f"Analyzing heating rates for {len(self.heating_rate_files)} files...")
        
        try:
            # Initialize results storage
            heating_analysis_results = {}
            all_temperatures = {'x': [], 'y': [], 'z': []}
            all_delays = []
            
            results_text = "HEATING RATE ANALYSIS RESULTS\\n"
            results_text += "=" * 50 + "\\n\\n"
            
            # Get analysis parameters
            freq_x = self.thermal_freq_x_spin.value() * 1000 * 2 * np.pi  # Convert kHz to rad/s
            freq_y = self.thermal_freq_y_spin.value() * 1000 * 2 * np.pi
            freq_z = self.thermal_freq_z_spin.value() * 1000 * 2 * np.pi
            ion_mass = self.thermal_ion_mass_spin.value() * amu  # Convert amu to kg
            wavelength = self.thermal_wavelength_spin.value() * 1e-9  # Convert nm to m
            pulse_area = self.thermal_pulse_area_spin.value() * np.pi  # Convert to radians
            
            # Process each file
            for i, file_entry in enumerate(self.heating_rate_files):
                file_path = file_entry['file_path']
                filename = file_entry['filename']
                probe_delay = file_entry['probe_delay']
                peak_params = file_entry['peak_detection_params']
                
                results_text += f"File {i+1}: {filename}\\n"
                results_text += f"  Probe delay: {probe_delay:.3f} ms\\n"
                results_text += f"  Detected peaks: {peak_params['n_peaks']}\\n"
                
                try:
                    # Load file data
                    detuning, relative_detuning, excitation_prob, excitation_prob_err, freq_center_MHz = import_spectroscopy(file_path)
                    
                    x_data = np.array(relative_detuning)
                    y_data = np.array(excitation_prob)
                    
                    # Use the stored peak detection parameters
                    peaks = self.detect_peaks_generic(
                        x_data, y_data,
                        threshold=peak_params['height'],
                        min_distance=peak_params['distance'],
                        prominence=peak_params['prominence']
                    )
                    
                    if len(peaks) == 0:
                        results_text += f"  Status: No peaks detected with stored parameters\\n\\n"
                        continue
                    
                    # Fit peaks
                    fit_results = self.fit_lorentzian_peaks_generic(x_data, y_data, peaks)
                    
                    if len(fit_results) == 0:
                        results_text += f"  Status: Peak fitting failed\\n\\n"
                        continue
                    
                    # Analyze temperatures for each axis using axis-specific peak assignment
                    file_temps = {'x': [], 'y': [], 'z': []}
                    axes = [('x', freq_x), ('y', freq_y), ('z', freq_z)]
                    
                    # Get peak centers and amplitudes
                    peak_centers = np.array([peak['center'] for peak in fit_results]) / 1000  # Convert to kHz
                    peak_amps = np.array([peak['amplitude'] for peak in fit_results])
                    
                    for axis_name, omega in axes:
                        omega_khz = omega / (2 * np.pi * 1000)  # Convert to kHz
                        target_freq = -omega_khz  # Red sideband expected at -omega
                        
                        # Find the closest peak to the expected red sideband frequency
                        if len(peak_centers) > 0:
                            idx = (np.abs(peak_centers - target_freq)).argmin()
                            fitted_amp = peak_amps[idx]
                            fitted_freq = peak_centers[idx]
                            freq_diff = abs(fitted_freq - target_freq)
                            
                            # Only use this peak if it's reasonably close to expected frequency
                            if freq_diff < omega_khz * 0.7:  # Within 70% of trap frequency
                                try:
                                    # Define objective function for this specific axis
                                    def model_prob(T):
                                        return calculate_sideband_probability(T, omega, -1, wavelength, ion_mass, 25e-6, pulse_area, axis_name)
                                    
                                    def objective(T):
                                        return abs(model_prob(T) - fitted_amp)
                                    
                                    # Fit temperature
                                    result = minimize_scalar(objective, bounds=(1e-7, 1e-2), method='bounded')
                                    
                                    if result.success:
                                        temp_mK = result.x * 1000  # Convert to mK
                                        file_temps[axis_name].append(temp_mK)
                                        results_text += f"  {axis_name.upper()}-axis: Peak at {fitted_freq:.1f} kHz → {temp_mK:.1f} mK\\n"
                                    else:
                                        results_text += f"  {axis_name.upper()}-axis: Temperature fit failed\\n"
                                except Exception as temp_err:
                                    results_text += f"  {axis_name.upper()}-axis: Error - {str(temp_err)[:30]}\\n"
                            else:
                                results_text += f"  {axis_name.upper()}-axis: No suitable peak (closest: {fitted_freq:.1f} kHz, expected: {target_freq:.1f} kHz)\\n"
                    
                    # Store results for this file
                    file_avg_temps = {}
                    for axis_name in ['x', 'y', 'z']:
                        if file_temps[axis_name]:
                            avg_temp = np.mean(file_temps[axis_name])
                            file_avg_temps[axis_name] = avg_temp
                            all_temperatures[axis_name].append(avg_temp)
                            results_text += f"  {axis_name.upper()}-axis: {avg_temp:.1f} mK\\n"
                        else:
                            results_text += f"  {axis_name.upper()}-axis: No valid measurements\\n"
                    
                    all_delays.append(probe_delay)
                    heating_analysis_results[filename] = {
                        'probe_delay': probe_delay,
                        'temperatures': file_avg_temps,
                        'peaks_detected': len(peaks),
                        'peaks_fitted': len(fit_results)
                    }
                    
                    results_text += f"  Status: Analysis completed\\n\\n"
                    
                except Exception as e:
                    results_text += f"  Status: Error - {str(e)[:50]}\\n\\n"
                    self.log_message(f"Error analyzing {filename}: {str(e)}")
            
            # Store results
            self.heating_rate_results = {
                'file_results': heating_analysis_results,
                'temperature_data': all_temperatures,
                'delays': all_delays,
                'file_info': [(entry['file_path'], entry['filename'], entry['probe_delay']) for entry in self.heating_rate_files]
            }
            
            # Summary
            results_text += "SUMMARY:\\n"
            results_text += "-" * 20 + "\\n"
            results_text += f"Total files analyzed: {len(heating_analysis_results)}\\n"
            for axis_name in ['x', 'y', 'z']:
                n_points = len(all_temperatures[axis_name])
                results_text += f"{axis_name.upper()}-axis: {n_points} temperature measurements\\n"
            
            self.heating_results_text.setText(results_text)
            self.log_message("Heating rate analysis completed")
            
            # Plot temperature vs delay
            if all_delays:
                self.plot_heating_rate_data()
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Failed toanalyze heating rates:\n{str(e)}")
            self.log_message(f"Heating rate analysis error: {str(e)}")
    
    def plot_heating_rate_data(self):
        """Plot temperature vs probe delay data with error bars and motional number plot in separate windows with publication-ready formatting"""
        try:
            if not hasattr(self, 'heating_rate_results'):
                return
            
            temp_data = self.heating_rate_results['temperature_data']
            delays = np.array(self.heating_rate_results['delays'])
            
            colors = ['#E74C3C', '#27AE60', '#3498DB']  # Professional colors
            axes_names = ['x', 'y', 'z']
            
            # Get trap frequencies for motional number calculation
            freq_x = self.thermal_freq_x_spin.value() * 1000 * 2 * np.pi  # rad/s
            freq_y = self.thermal_freq_y_spin.value() * 1000 * 2 * np.pi
            freq_z = self.thermal_freq_z_spin.value() * 1000 * 2 * np.pi
            frequencies = {'x': freq_x, 'y': freq_y, 'z': freq_z}
            
            # Create side-by-side matplotlib figure with publication styling
            import matplotlib.pyplot as plt
            
            # Close any existing plots
            plt.close('all')
            
            # Set publication-ready matplotlib parameters
            plt.rcParams.update({
                'font.size': 12,
                'axes.linewidth': 1.5,
                'lines.linewidth': 2.5,
                'grid.linewidth': 0.8,
                'xtick.major.width': 1.5,
                'ytick.major.width': 1.5,
                'xtick.minor.width': 1,
                'ytick.minor.width': 1,
                'xtick.major.size': 6,
                'ytick.major.size': 6,
                'xtick.minor.size': 3,
                'ytick.minor.size': 3
            })
            
            # Create figure with side-by-side subplots
            fig, (temp_ax, motional_ax) = plt.subplots(1, 2, figsize=(16, 7))
            fig.suptitle('Ion Heating Rate Analysis', fontsize=18, fontweight='bold', y=0.95)
            
            for axis_name, color in zip(axes_names, colors):
                if temp_data[axis_name]:
                    temps = np.array(temp_data[axis_name])
                    
                    # Calculate error bars as standard error if multiple measurements per delay
                    # For now, assume 10% uncertainty as typical for temperature measurements
                    temp_errors = temps * 0.1  # 10% error bars
                    
                    # Plot temperature vs delay with enhanced styling
                    temp_ax.errorbar(delays, temps, yerr=temp_errors, fmt='o-', color=color, 
                                   label=f'{axis_name.upper()}-axis', markersize=10, capsize=6, 
                                   capthick=2.5, linewidth=3, markeredgewidth=2, markeredgecolor='black')
                    
                    # Calculate and plot average motional number vs delay
                    omega = frequencies[axis_name]
                    # Convert temperature from mK to K, then calculate <n> = kT/(ħω) - 1/2
                    temps_K = temps * 1e-3
                    avg_n = (KB * temps_K) / (HBAR * omega) - 0.5
                    avg_n = np.maximum(avg_n, 0)  # Ensure non-negative
                    
                    # Error propagation for motional number
                    n_errors = (KB * temp_errors * 1e-3) / (HBAR * omega)
                    
                    motional_ax.errorbar(delays, avg_n, yerr=n_errors, fmt='s-', color=color,
                                       label=f'{axis_name.upper()}-axis', markersize=10, capsize=6, 
                                       capthick=2.5, linewidth=3, markeredgewidth=2, markeredgecolor='black')
            
            # Format temperature plot with publication styling
            temp_ax.set_xlabel('Probe Delay (ms)', fontsize=14, fontweight='bold')
            temp_ax.set_ylabel('Temperature (mK)', fontsize=14, fontweight='bold')
            temp_ax.set_title('Ion Temperature vs Probe Delay', fontsize=16, fontweight='bold', pad=15)
            temp_ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
            temp_ax.tick_params(axis='both', which='minor', width=1, length=3)
            
            # Professional legend for temperature plot
            legend_temp = temp_ax.legend(fontsize=12, frameon=True, fancybox=True, 
                                       shadow=True, framealpha=0.95, edgecolor='black')
            legend_temp.get_frame().set_linewidth(1.2)
            
            temp_ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
            temp_ax.set_axisbelow(True)
            
            # Improve spines for temperature plot
            for spine in temp_ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color('black')
            
            # Format motional number plot with publication styling
            motional_ax.set_xlabel('Probe Delay (ms)', fontsize=14, fontweight='bold')
            motional_ax.set_ylabel('Average Motional Number ⟨n⟩', fontsize=14, fontweight='bold')
            motional_ax.set_title('Average Motional Number vs Probe Delay', fontsize=16, fontweight='bold', pad=15)
            motional_ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
            motional_ax.tick_params(axis='both', which='minor', width=1, length=3)
            
            # Professional legend for motional number plot
            legend_motional = motional_ax.legend(fontsize=12, frameon=True, fancybox=True, 
                                               shadow=True, framealpha=0.95, edgecolor='black')
            legend_motional.get_frame().set_linewidth(1.2)
            
            motional_ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
            motional_ax.set_axisbelow(True)
            
            # Improve spines for motional number plot
            for spine in motional_ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color('black')
            
            # Adjust layout and show the side-by-side plot
            fig.tight_layout(pad=1.5)
            fig.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.95, wspace=0.25)
            plt.show()
            
            # Also update the main plot widget to show temperature plot for consistency
            self.heating_plot_widget.figure.clear()
            main_ax = self.heating_plot_widget.figure.add_subplot(1, 1, 1)
            
            # Add text overlay indicating external plots with enhanced styling
            main_ax.text(0.02, 0.98, '📊 External Plot Window Active', 
                        transform=main_ax.transAxes, fontsize=14, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F4FD", alpha=0.9, edgecolor='black'),
                        verticalalignment='top')
            
            for axis_name, color in zip(axes_names, colors):
                if temp_data[axis_name]:
                    temps = np.array(temp_data[axis_name])
                    temp_errors = temps * 0.1
                    main_ax.errorbar(delays, temps, yerr=temp_errors, fmt='o-', color=color, 
                                   label=f'{axis_name.upper()}-axis', markersize=8, capsize=5, 
                                   capthick=2, linewidth=2.5, markeredgewidth=1.5, markeredgecolor='black')
            
            # Enhanced styling for main plot widget
            main_ax.set_xlabel('Probe Delay (ms)', fontsize=14, fontweight='bold')
            main_ax.set_ylabel('Temperature (mK)', fontsize=14, fontweight='bold')
            main_ax.set_title('Ion Temperature vs Probe Delay', fontsize=16, fontweight='bold', pad=15)
            main_ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
            main_ax.tick_params(axis='both', which='minor', width=1, length=3)
            
            # Professional legend for main plot
            legend_main = main_ax.legend(fontsize=12, frameon=True, fancybox=True, 
                                       shadow=True, framealpha=0.95, edgecolor='black')
            legend_main.get_frame().set_linewidth(1.2)
            
            main_ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
            main_ax.set_axisbelow(True)
            
            # Improve spines for main plot
            for spine in main_ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color('black')
            
            # Ensure proper layout to prevent clipping - use conservative margins
            self.heating_plot_widget.figure.subplots_adjust(left=0.15, bottom=0.15, right=0.92, top=0.85)
            self.heating_plot_widget.canvas.draw()
            
            # Store references to the external figure for fit plotting
            self.temp_figure = fig
            self.motional_figure = fig  # Same figure now
            self.temp_ax = temp_ax
            self.motional_ax = motional_ax
            
        except Exception as e:
            self.log_message(f"Plot error: {str(e)}")
    
    def fit_heating_rate(self):
        """Fit heating rates from the analyzed temperature data"""
        if not hasattr(self, 'heating_rate_results'):
            QMessageBox.warning(self, "No Data", "Please analyze heating rates first.")
            return
        
        try:
            temp_data = self.heating_rate_results['temperature_data']
            delays = np.array(self.heating_rate_results['delays'])
            
            if len(delays) < 2:
                QMessageBox.warning(self, "Insufficient Data", "Need at least 2 data points for fitting.")
                return
            
            heating_fits = {}
            results_text = self.heating_results_text.toPlainText()
            results_text += "\n\n" + "="*60 + "\n"
            results_text += "HEATING RATE ANALYSIS RESULTS\n"
            results_text += "="*60 + "\n\n"
            
            # Get trap frequencies for motional number calculation
            freq_x = self.thermal_freq_x_spin.value() * 1000 * 2 * np.pi  # rad/s
            freq_y = self.thermal_freq_y_spin.value() * 1000 * 2 * np.pi
            freq_z = self.thermal_freq_z_spin.value() * 1000 * 2 * np.pi
            frequencies = {'x': freq_x, 'y': freq_y, 'z': freq_z}
            
            for axis_name in ['x', 'y', 'z']:
                if temp_data[axis_name] and len(temp_data[axis_name]) >= 2:
                    temps = np.array(temp_data[axis_name])
                    
                    # Linear fit: T = T0 + rate * delay
                    try:
                        coeffs = np.polyfit(delays, temps, 1)
                        heating_rate = coeffs[0]  # mK/ms
                        T0 = coeffs[1]  # mK
                        
                        # Calculate R-squared
                        fit_temps = np.polyval(coeffs, delays)
                        ss_res = np.sum((temps - fit_temps) ** 2)
                        ss_tot = np.sum((temps - np.mean(temps)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        
                        # Calculate motional number heating rate
                        omega = frequencies[axis_name]
                        omega_khz = omega / (2 * np.pi * 1000)
                        
                        # Convert temperature heating rate to motional number heating rate
                        # dn/dt = (1/ħω) * (kB * dT/dt)
                        temp_rate_K_s = heating_rate * 1e-3 * 1000  # Convert mK/ms to K/s
                        n_rate = (KB * temp_rate_K_s) / (HBAR * omega)  # quanta/s
                        
                        # Initial motional number
                        T0_K = T0 * 1e-3  # Convert to K
                        n0 = (KB * T0_K) / (HBAR * omega) - 0.5
                        n0 = max(n0, 0)  # Ensure non-negative
                        
                        heating_fits[axis_name] = {
                            'T0': T0 / 1000,  # Convert to K
                            'rate': heating_rate / 1000,  # Convert to K/ms
                            'r_squared': r_squared,
                            'delays': delays,
                            'temperatures': temps / 1000,  # Convert to K
                            'n0': n0,
                            'n_rate': n_rate,
                            'heating_rate_mK_ms': heating_rate,
                            'heating_rate_uK_s': heating_rate * 1000
                        }
                        
                        results_text += f"🔥 {axis_name.upper()}-AXIS HEATING ANALYSIS (ω = {omega_khz:.0f} kHz)\n"
                        results_text += "─" * 50 + "\n"
                        results_text += f"📊 Fit Quality: R² = {r_squared:.4f} ({len(temps)} data points)\n\n"
                        
                        results_text += "🌡️  TEMPERATURE ANALYSIS:\n"
                        results_text += f"   • Initial temperature: T₀ = {T0:.2f} mK\n"
                        results_text += f"   • Heating rate: {heating_rate:.3f} mK/ms = {heating_rate*1000:.1f} μK/s\n\n"
                        
                        results_text += "🔢 MOTIONAL NUMBER ANALYSIS:\n"
                        results_text += f"   • Initial motional number: ⟨n₀⟩ = {n0:.2f}\n"
                        results_text += f"   • Motional number heating rate: {n_rate:.1f} quanta/s\n"
                        results_text += f"   • Heating time constant: τ = {1/n_rate:.1f} s (to gain 1 quantum)\n\n"
                        
                        # Add quality assessment
                        if r_squared > 0.9:
                            quality = "🟢 EXCELLENT"
                        elif r_squared > 0.7:
                            quality = "🟡 GOOD"
                        elif r_squared > 0.5:
                            quality = "🟠 FAIR"
                        else:
                            quality = "🔴 POOR"
                        
                        results_text += f"✅ Fit Quality Assessment: {quality}\n"
                        results_text += "=" * 50 + "\n\n"
                        
                    except Exception as e:
                        results_text += f"❌ {axis_name.upper()}-axis: Fit failed - {str(e)}\n\n"
                else:
                    results_text += f"⚠️  {axis_name.upper()}-axis: Insufficient data points (need ≥2)\n\n"
            
            # Store fit results
            self.heating_rate_results['heating_fits'] = heating_fits
            
            # Update display
            self.heating_results_text.setText(results_text)
            
            # Update plot with fits
            if heating_fits:
                self.plot_heating_rate_fits(heating_fits)
            
            self.log_message("Heating rate fitting completed")
            
        except Exception as e:
            QMessageBox.critical(self, "Fit Error", f"Failed to fit heating rates:\n{str(e)}")
            self.log_message(f"Heating rate fit error: {str(e)}")
    
    def plot_heating_rate_fits(self, heating_fits):
        """Plot heating rate fits over the data on both external and main widget plots with publication-ready formatting"""
        try:
            delays = np.array(self.heating_rate_results['delays'])
            delay_range = np.linspace(delays.min(), delays.max(), 100)
            
            colors = ['#E74C3C', '#27AE60', '#3498DB']  # Professional colors
            axes_names = ['x', 'y', 'z']
            
            # Get trap frequencies for motional number calculation
            freq_x = self.thermal_freq_x_spin.value() * 1000 * 2 * np.pi  # rad/s
            freq_y = self.thermal_freq_y_spin.value() * 1000 * 2 * np.pi
            freq_z = self.thermal_freq_z_spin.value() * 1000 * 2 * np.pi
            frequencies = {'x': freq_x, 'y': freq_y, 'z': freq_z}
            
            # Plot fits on external figure if available
            if hasattr(self, 'temp_ax') and hasattr(self, 'motional_ax'):
                for axis_name, color in zip(axes_names, colors):
                    if axis_name in heating_fits:
                        fit_data = heating_fits[axis_name]
                        T0 = fit_data['T0'] * 1000  # Convert to mK for plotting
                        rate = fit_data['rate'] * 1000  # Convert to mK/ms
                        
                        # Temperature fit line on external plot with enhanced styling
                        fit_line = T0 + rate * delay_range
                        self.temp_ax.plot(delay_range, fit_line, '--', 
                                       color=color, linewidth=4, alpha=0.9,
                                       label=f'{axis_name.upper()}-axis fit\n'
                                             f'Rate: {fit_data["rate"]:.2e} K/s')
                        
                        # Motional number fit line on external plot
                        omega = frequencies[axis_name]
                        fit_line_K = fit_line * 1e-3  # Convert to K
                        n_fit_line = (KB * fit_line_K) / (HBAR * omega) - 0.5
                        n_fit_line = np.maximum(n_fit_line, 0)  # Ensure non-negative
                        
                        self.motional_ax.plot(delay_range, n_fit_line, '--',
                                           color=color, linewidth=4, alpha=0.9,
                                           label=f'{axis_name.upper()}-axis fit\n'
                                                 f'Rate: {fit_data["rate"]:.2e} K/s')
                
                # Update legends with professional styling and redraw external plots
                legend_temp = self.temp_ax.legend(fontsize=11, frameon=True, fancybox=True, 
                                               shadow=True, framealpha=0.95, edgecolor='black')
                legend_temp.get_frame().set_linewidth(1.2)
                
                legend_motional = self.motional_ax.legend(fontsize=11, frameon=True, fancybox=True, 
                                                       shadow=True, framealpha=0.95, edgecolor='black')
                legend_motional.get_frame().set_linewidth(1.2)
                
                self.temp_figure.canvas.draw()
            
            # Also plot fits on main widget for consistency with enhanced styling
            main_ax = self.heating_plot_widget.figure.get_axes()[0]
            for axis_name, color in zip(axes_names, colors):
                if axis_name in heating_fits:
                    fit_data = heating_fits[axis_name]
                    T0 = fit_data['T0'] * 1000  # Convert to mK for plotting
                    rate = fit_data['rate'] * 1000  # Convert to mK/ms
                    
                    # Temperature fit line on main widget with enhanced styling
                    fit_line = T0 + rate * delay_range
                    main_ax.plot(delay_range, fit_line, '--', 
                               color=color, linewidth=3, alpha=0.9,
                               label=f'{axis_name.upper()}-axis fit\n'
                                     f'Rate: {fit_data["rate"]:.2e} K/s')
            
            # Professional legend for main plot
            legend_main = main_ax.legend(fontsize=11, frameon=True, fancybox=True, 
                                       shadow=True, framealpha=0.95, edgecolor='black')
            legend_main.get_frame().set_linewidth(1.2)
            
            # Ensure proper layout to prevent clipping
            self.heating_plot_widget.figure.tight_layout(pad=1.0)
            self.heating_plot_widget.canvas.draw()
            
        except Exception as e:
            self.log_message(f"Plot fit error: {str(e)}")
    
    def export_heating_results(self):
        """Export heating rate analysis results"""
        if not hasattr(self, 'heating_rate_results'):
            QMessageBox.warning(self, "No Results", "No heating rate results to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Heating Rate Analysis Results", "", "Text files (*.txt);;All files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.heating_results_text.toPlainText())
                    f.write("\\n\\n" + "="*60 + "\\n")
                    f.write("DETAILED ANALYSIS DATA\\n")
                    f.write("="*60 + "\\n\\n")
                    
                    # Write file information
                    if 'file_info' in self.heating_rate_results:
                        f.write("ANALYZED FILES:\\n")
                        f.write("-" * 20 + "\\n")
                        file_info = self.heating_rate_results['file_info']
                        for i, (filepath, filename, delay) in enumerate(file_info):
                            f.write(f"File {i+1}: {filename}\\n")
                            f.write(f"  Probe delay: {delay:.3f} ms\\n")
                            f.write(f"  Path: {filepath}\\n\\n")
                    
                    # Write heating rate fits if available
                    if 'heating_fits' in self.heating_rate_results:
                        f.write("HEATING RATE FITS:\\n")
                        f.write("-" * 30 + "\\n")
                        heating_fits = self.heating_rate_results['heating_fits']
                        for axis, data in heating_fits.items():
                            f.write(f"{axis}-axis:\\n")
                            f.write(f"  Initial temperature: {data.get('T0', 0)*1e6:.3f} μK\\n")
                            f.write(f"  Heating rate: {data.get('rate', 0)*1e6:.3f} μK/s\\n")
                            f.write(f"  R-squared: {data.get('r_squared', 0):.6f}\\n")
                            f.write(f"  Data points: {len(data.get('delays', []))}\\n\\n")
                        
                    QMessageBox.information(self, "Success", f"Heating rate results exported to {file_path}")
                    self.log_message(f"Exported heating rate results to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export results:\n{str(e)}")
    
    # Thermal analysis methods
    def load_thermal_file(self):
        """Load a spectroscopy file for thermal analysis"""
        if import_spectroscopy is None:
            QMessageBox.warning(self, "Import Error", 
                              "Data import functionality not available.")
            return
        
        initial_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "used_data-march")
        if not os.path.exists(initial_dir):
            initial_dir = os.path.dirname(__file__)
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Spectroscopy Data File for Thermal Analysis", initial_dir,
            "HDF5 files (*.h5);;All files (*.*)"
        )
        
        if not file_path:
            return
        
        filename = os.path.basename(file_path)
        
        try:
            self.log_message(f"Loading thermal analysis file: {filename}")
            
            # Load spectroscopy data
            detuning, relative_detuning, excitation_prob, excitation_prob_err, freq_center_MHz = import_spectroscopy(file_path)
            
            # Store thermal data
            self.thermal_data = {
                'x_data': np.array(relative_detuning) / 1000,  # Convert to kHz
                'y_data': np.array(excitation_prob),
                'yerr': np.array(excitation_prob_err),
                'freq_center_MHz': freq_center_MHz,
                'file_path': file_path,
                'filename': filename
            }
            
            # Update UI
            self.thermal_file_label.setText(f"Loaded: {filename}")
            self.thermal_file_label.setStyleSheet("color: green;")
            
            self.log_message(f"Thermal file loaded successfully!")
            self.log_message(f"Data points: {len(self.thermal_data['x_data'])}")
            
            # Plot data
            self.thermal_plot_widget.plot_spectroscopy_data(
                self.thermal_data['x_data'], 
                self.thermal_data['y_data'], 
                self.thermal_data['yerr']
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load thermal file:\n{str(e)}")
            self.log_message(f"Error loading thermal file {filename}: {str(e)}")
    
    def analyze_thermal_temperature(self):
        """Analyze thermal temperature from spectroscopy data (robust, axis-specific, with error propagation)"""
        if self.thermal_data is None:
            QMessageBox.warning(self, "No Data", "Please load a spectroscopy file first.")
            return
        
        self.log_message("Analyzing thermal temperature...")
        
        try:
            # Get analysis parameters
            freq_x = self.thermal_freq_x_spin.value() * 1000 * 2 * np.pi  # kHz to rad/s
            freq_y = self.thermal_freq_y_spin.value() * 1000 * 2 * np.pi
            freq_z = self.thermal_freq_z_spin.value() * 1000 * 2 * np.pi
            ion_mass = self.thermal_ion_mass_spin.value() * amu
            wavelength = self.thermal_wavelength_spin.value() * 1e-9
            pulse_area = self.thermal_pulse_area_spin.value() * np.pi
            pulse_time = 25e-6  # Default pulse time (s), adjust if needed

            # Detect and fit peaks
            x_data = self.thermal_data['x_data'] * 1000  # Hz
            y_data = self.thermal_data['y_data']
            peaks = self.detect_peaks_generic(x_data, y_data, threshold=0.02)
            if len(peaks) == 0:
                QMessageBox.warning(self, "No Peaks", "No peaks detected in the data. Try adjusting detection parameters.")
                return
            fit_results = self.fit_lorentzian_peaks_generic(x_data, y_data, peaks)
            if len(fit_results) == 0:
                QMessageBox.warning(self, "Fit Failed", "Could not fit any peaks.")
                return

            # Prepare for axis-specific analysis
            axes = [('x', freq_x), ('y', freq_y), ('z', freq_z)]
            peak_centers = np.array([peak['center'] for peak in fit_results]) / 1000  # kHz
            peak_amps = np.array([peak['amplitude'] for peak in fit_results])
            peak_amp_errs = np.array([peak.get('amplitude_err', 0.0) for peak in fit_results])

            results_text = "MULTI-AXIS TEMPERATURE EXTRACTION\n" + "="*50 + "\n\n"
            results_text += f"Physical Parameters:\n• Ion mass: {ion_mass/amu:.1f} amu\n• Laser wavelength: {wavelength*1e9:.1f} nm\n• Pulse area: {self.thermal_pulse_area_spin.value():.2f}π = {pulse_area:.3f} rad\n\n"
            temperature_results = {}
            fitted_peaks = []

            for axis_name, omega in axes:
                omega_khz = omega / (2 * np.pi * 1000)
                target_freq = -omega_khz  # Red sideband expected at -omega
                idx = (np.abs(peak_centers - target_freq)).argmin()
                fitted_amp = peak_amps[idx]
                fitted_freq = peak_centers[idx]
                fitted_amp_err = peak_amp_errs[idx]
                freq_diff = abs(fitted_freq - target_freq)
                results_text += f"{axis_name.upper()}-AXIS ANALYSIS (omega = {omega_khz:.1f} kHz):\n" + "-"*30 + "\n"
                if freq_diff > omega_khz * 0.5:
                    results_text += f"⚠️  Warning: Peak at {fitted_freq:.1f} kHz is far from expected {target_freq:.1f} kHz\n"
                def model_prob(T):
                    return calculate_sideband_probability(T, omega, -1, wavelength, ion_mass, pulse_time, pulse_area, axis_name)
                def objective(T):
                    return abs(model_prob(T) - fitted_amp)
                try:
                    res = minimize_scalar(objective, bounds=(1e-7, 1e-2), method='bounded')
                    temperature_k = res.x
                    temperature_mk = temperature_k * 1e3
                    model_amp = model_prob(temperature_k)
                    temp_err_k = 0.0
                    temp_err_mk = 0.0
                    if fitted_amp_err > 0:
                        dT = temperature_k * 0.001
                        if temperature_k + dT < 1e-2:
                            P_plus = model_prob(temperature_k + dT)
                            P_minus = model_prob(temperature_k - dT)
                            dP_dT = (P_plus - P_minus) / (2 * dT)
                            if abs(dP_dT) > 1e-10:
                                temp_err_k = fitted_amp_err / abs(dP_dT)
                                temp_err_mk = temp_err_k * 1e3
                            else:
                                results_text += f"   ⚠️  Warning: Very small temperature sensitivity - error calculation unreliable\n"
                        else:
                            results_text += f"   ⚠️  Warning: Temperature too close to upper bound for error calculation\n"
                    mean_n = 1 / (np.exp(HBAR * omega / (KB * temperature_k)) - 1)
                    mean_n_err = 0.0
                    if temp_err_k > 0:
                        x = HBAR * omega / (KB * temperature_k)
                        exp_x = np.exp(x)
                        dn_dT = (x / temperature_k) * exp_x / (exp_x - 1)**2
                        mean_n_err = abs(dn_dT * temp_err_k)
                    eta = lamb_dicke(wavelength, ion_mass, omega, axis=axis_name)
                    temperature_results[axis_name] = {
                        'temperature_k': temperature_k,
                        'temperature_mk': temperature_mk,
                        'temperature_err_k': temp_err_k,
                        'temperature_err_mk': temp_err_mk,
                        'mean_n': mean_n,
                        'mean_n_err': mean_n_err,
                        'fitted_amp': fitted_amp,
                        'fitted_amp_err': fitted_amp_err,
                        'model_amp': model_amp,
                        'fitted_freq': fitted_freq,
                        'expected_freq': target_freq,
                        'eta': eta,
                        'omega_khz': omega_khz
                    }
                    fitted_peaks.append({'axis': axis_name, 'fitted': True, 'fitted_freq': fitted_freq, 'expected_freq': target_freq, 'fitted_amp': fitted_amp, 'fitted_amp_err': fitted_amp_err})
                    results_text += f"✅ Temperature extracted successfully:\n"
                    results_text += f"   Peak: {fitted_freq:.1f} kHz (amplitude: {fitted_amp:.4f} ± {fitted_amp_err:.4f})\n"
                    results_text += f"   Expected: {target_freq:.1f} kHz\n"
                    if temp_err_mk > 0:
                        results_text += f"   Temperature: {temperature_mk:.2f} ± {temp_err_mk:.2f} mK\n"
                    else:
                        results_text += f"   Temperature: {temperature_mk:.2f} mK\n"
                    results_text += f"   Mean n: {mean_n:.2f} ± {mean_n_err:.2f}\n   η: {eta:.4f}\n\n"
                except Exception as e:
                    results_text += f"   ❌ Error extracting temperature: {str(e)[:40]}\n"
            # Store results
            self.thermal_analysis_results = {
                'temperatures': temperature_results,
                'fit_results': fit_results,
                'peaks': peaks,
                'fitted_peaks': fitted_peaks,
                'parameters': {
                    'frequencies': {'x': freq_x, 'y': freq_y, 'z': freq_z},
                    'ion_mass': ion_mass,
                    'wavelength': wavelength,
                    'pulse_area': pulse_area
                }
            }
            
            # Plot fitted curves along with experimental data
            self.plot_fitted_thermal_data(x_data, y_data, fit_results, temperature_results)
            
            self.thermal_results_text.setText(results_text)
            self.log_message("Thermal temperature analysis completed")
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Failed to analyze thermal temperature:\n{str(e)}")
            self.log_message(f"Thermal analysis error: {str(e)}")
    
    def plot_fitted_thermal_data(self, x_data, y_data, fit_results, temperature_results):
        """Plot experimental data with fitted Lorentzian curves with publication-ready formatting"""
        try:
            # Clear and set up the plot
            self.thermal_plot_widget.ax.clear()
            
            # Plot experimental data with publication-ready styling
            x_plot = x_data / 1000  # Convert to kHz for plotting
            self.thermal_plot_widget.ax.errorbar(x_plot, y_data, yerr=self.thermal_data['yerr'], 
                                                fmt='o', markersize=7, capsize=5, capthick=2,
                                                markeredgewidth=1.5, color='#2E86AB', 
                                                markeredgecolor='#1A5276', elinewidth=2,
                                                label='Experimental Data')
            
            # Plot fitted curves if multi-peak fit was successful
            if hasattr(self, '_last_fit_result') and self._last_fit_result.get('success', False):
                fit_data = self._last_fit_result
                x_fit = np.linspace(x_data.min(), x_data.max(), 1000)
                
                # Plot total fit with professional styling
                y_total = self.multi_lorentzian(x_fit, *fit_data['popt'])
                self.thermal_plot_widget.ax.plot(x_fit / 1000, y_total, '-', 
                                                color='#A23B72', linewidth=3, alpha=0.9,
                                                label='Total Fit')
                
                # Plot individual peaks with axis assignments
                n_peaks = fit_data['n_peaks']
                peak_centers = fit_data['popt'][:n_peaks]
                peak_widths = fit_data['popt'][n_peaks:2*n_peaks]
                peak_amplitudes = fit_data['popt'][2*n_peaks:3*n_peaks]
                baseline = fit_data['popt'][-1]
                
                colors = ['#E74C3C', '#27AE60', '#3498DB']  # Professional red, green, blue
                axes_names = ['x', 'y', 'z']
                
                for i in range(n_peaks):
                    # Calculate individual peak
                    y_peak = (peak_amplitudes[i] / (1 + ((x_fit - peak_centers[i]) / peak_widths[i]) ** 2)) + baseline
                    
                    # Determine which axis this peak belongs to based on temperature results
                    assigned_axis = None
                    assigned_color = '#95A5A6'  # Professional gray
                    
                    for axis_name, color in zip(axes_names, colors):
                        if axis_name in temperature_results:
                            expected_freq = temperature_results[axis_name]['expected_freq'] * 1000  # Hz
                            if abs(peak_centers[i] - expected_freq) < abs(peak_centers[i] + temperature_results[axis_name]['omega_khz'] * 1000) * 0.3:
                                assigned_axis = axis_name
                                assigned_color = color
                                break
                    
                    label = f'Peak {i+1}' + (f' ({assigned_axis}-axis)' if assigned_axis else '')
                    self.thermal_plot_widget.ax.plot(x_fit / 1000, y_peak, '--', 
                                                    color=assigned_color, alpha=0.8, 
                                                    linewidth=2.5, label=label)
            else:
                # Fallback: plot individual fitted peaks from fit_results
                x_fit = np.linspace(x_data.min(), x_data.max(), 1000)
                colors = ['#E74C3C', '#27AE60', '#3498DB', '#F39C12', '#9B59B6']
                for i, peak in enumerate(fit_results):
                    y_peak = self.lorentzian(x_fit, peak['amplitude'], peak['center'], 
                                           peak['width'], peak['y0'])
                    color = colors[i % len(colors)]
                    self.thermal_plot_widget.ax.plot(x_fit / 1000, y_peak, '--', 
                                                    color=color, alpha=0.8, linewidth=2.5, 
                                                    label=f'Peak {i+1}')
            
            # Professional formatting
            self.thermal_plot_widget.ax.set_xlabel('Relative Detuning (kHz)', fontsize=14, fontweight='bold')
            self.thermal_plot_widget.ax.set_ylabel('Excitation Probability', fontsize=14, fontweight='bold')
            self.thermal_plot_widget.ax.set_title('Thermal Analysis: Data with Fitted Peaks', 
                                                 fontsize=16, fontweight='bold', pad=20)
            
            # Improve tick labels
            self.thermal_plot_widget.ax.tick_params(axis='both', which='major', labelsize=12, 
                                                   width=1.5, length=6)
            self.thermal_plot_widget.ax.tick_params(axis='both', which='minor', width=1, length=3)
            
            # Professional legend
            legend = self.thermal_plot_widget.ax.legend(fontsize=11, frameon=True, fancybox=True, 
                                                       shadow=True, framealpha=0.95, edgecolor='black')
            legend.get_frame().set_linewidth(1.2)
            
            # Enhanced grid
            self.thermal_plot_widget.ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
            self.thermal_plot_widget.ax.set_axisbelow(True)
            
            # Improve spines
            for spine in self.thermal_plot_widget.ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color('black')
            
            # Ensure proper layout to prevent clipping - use conservative margins
            self.thermal_plot_widget.figure.subplots_adjust(left=0.15, bottom=0.15, right=0.92, top=0.85)
            
            # Update the canvas
            self.thermal_plot_widget.canvas.draw()
            
        except Exception as e:
            self.log_message(f"Error plotting fitted data: {str(e)}")
    
    def plot_thermal_model(self):
        """Plot thermal model comparison showing measured red sideband excitation probability on theoretical thermal distribution curves"""
        if not self.thermal_analysis_results:
            QMessageBox.warning(self, "No Results", "Please analyze temperature first!")
            return
        
        try:
            self.log_message("Plotting thermal model...")
            
            # Get experimental data and analysis results
            temperatures = self.thermal_analysis_results.get('temperatures', {})
            if not temperatures:
                QMessageBox.warning(self, "No Temperature Results", "No temperature analysis results available!")
                return
            
            # Get analysis parameters
            params = self.thermal_analysis_results['parameters']
            ion_mass = params['ion_mass']
            wavelength = params['wavelength']
            pulse_area = params['pulse_area']
            
            # Clear the plot
            self.thermal_plot_widget.ax.clear()
            
            # Create temperature range for theoretical curves (log scale from 0.01 to 100 mK)
            temp_range_mk = np.logspace(-2, 2, 100)  # 0.01 to 100 mK
            temp_range_k = temp_range_mk * 1e-3  # Convert to K
            
            # Plot theoretical curves for each axis with professional styling
            colors = ['#E74C3C', '#27AE60', '#3498DB']  # Professional red, green, blue
            axes_names = ['x', 'y', 'z']
            
            for i, (axis_name, color) in enumerate(zip(axes_names, colors)):
                if axis_name in temperatures:
                    # Get temperature and measured data for this axis
                    temp_mk = temperatures[axis_name]['temperature_mk']  # Already in mK
                    temp_err_mk = temperatures[axis_name]['temperature_err_mk']
                    fitted_amp = temperatures[axis_name]['fitted_amp']
                    fitted_amp_err = temperatures[axis_name]['fitted_amp_err']
                    omega = params['frequencies'][axis_name]
                    
                    # Calculate theoretical curve for red sideband (-1) excitation probability vs temperature
                    theoretical_probs = []
                    for temp_k in temp_range_k:
                        prob = calculate_sideband_probability(
                            temp_k, omega, -1, wavelength, ion_mass, 25e-6, pulse_area, axis_name
                        )
                        theoretical_probs.append(prob)
                    
                    # Plot theoretical curve with enhanced styling
                    self.thermal_plot_widget.ax.loglog(temp_range_mk, theoretical_probs, 
                                                     '-', color=color, linewidth=3, alpha=0.9,
                                                     label=f'{axis_name.upper()}-axis theoretical')
                    
                    # Plot the measured excitation probability at the extracted temperature
                    # This is the key point: plot measured red sideband probability at the temperature where it should occur
                    if temp_err_mk > 0 and fitted_amp_err > 0:
                        # Plot with both temperature and amplitude error bars - enhanced styling
                        self.thermal_plot_widget.ax.errorbar(temp_mk, fitted_amp, 
                                                           xerr=temp_err_mk, yerr=fitted_amp_err,
                                                           fmt='o', color=color, markersize=12, 
                                                           markeredgecolor='black', markeredgewidth=2,
                                                           capsize=6, capthick=2.5, elinewidth=2.5,
                                                           label=f'{axis_name.upper()}-axis measured', zorder=5)
                        
                        # Add uncertainty band for temperature
                        temp_lower = max(temp_mk - temp_err_mk, 0.01)  # Don't go below 0.01 mK
                        temp_upper = temp_mk + temp_err_mk
                        
                        try:
                            # Calculate uncertainty band
                            prob_lower = calculate_sideband_probability(temp_lower*1e-3, omega, -1, wavelength, ion_mass, 25e-6, pulse_area, axis_name)
                            prob_upper = calculate_sideband_probability(temp_upper*1e-3, omega, -1, wavelength, ion_mass, 25e-6, pulse_area, axis_name)
                            
                            # Add shaded uncertainty region with better styling
                            self.thermal_plot_widget.ax.fill_between([temp_lower, temp_upper], 
                                                                   [prob_lower, prob_lower], [prob_upper, prob_upper],                                                           color=color, alpha=0.25, 
                                                           label=f'{axis_name.upper()}-axis uncertainty')
                        except Exception:
                            pass  # Skip uncertainty band if calculation fails
                    else:
                        # Plot without error bars
                        self.thermal_plot_widget.ax.loglog(temp_mk, fitted_amp, 'o', color=color, markersize=12, 
                                                          markeredgecolor='black', markeredgewidth=2,
                                                          label=f'{axis_name.upper()}-axis measured', zorder=5)
            
            # Professional plot customization
            self.thermal_plot_widget.ax.set_xlabel('Temperature (mK)', fontsize=14, fontweight='bold')
            self.thermal_plot_widget.ax.set_ylabel('First Red Sideband Excitation Probability', fontsize=14, fontweight='bold')
            self.thermal_plot_widget.ax.set_title('Theoretical Thermal Distribution Curves', 
                                                 fontsize=16, fontweight='bold', pad=20)
            
            # Improve tick labels
            self.thermal_plot_widget.ax.tick_params(axis='both', which='major', labelsize=12, 
                                                   width=1.5, length=6)
            self.thermal_plot_widget.ax.tick_params(axis='both', which='minor', width=1, length=3)
            
            # Professional legend
            legend = self.thermal_plot_widget.ax.legend(fontsize=11, frameon=True, fancybox=True, 
                                                       shadow=True, framealpha=0.95, edgecolor='black')
            legend.get_frame().set_linewidth(1.2)
            
            # Enhanced grid
            self.thermal_plot_widget.ax.grid(True, alpha=0.4, which='both', linestyle='-', linewidth=0.8)
            self.thermal_plot_widget.ax.set_axisbelow(True)
            
            # Set axis limits and improve spines
            self.thermal_plot_widget.ax.set_xlim(0.01, 100)
            self.thermal_plot_widget.ax.set_ylim(1e-6, 1)
            
            for spine in self.thermal_plot_widget.ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color('black')
            
            # Ensure proper layout to prevent clipping - use conservative margins
            self.thermal_plot_widget.figure.subplots_adjust(left=0.15, bottom=0.15, right=0.92, top=0.85)
            
            self.thermal_plot_widget.canvas.draw()
            self.log_message("Thermal model plot completed - showing measured red sideband probability on theoretical curves")
            
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to plot thermal model:\n{str(e)}")
            self.log_message(f"Thermal model plot error: {str(e)}")
    
    def export_thermal_results(self):
        """Export thermal analysis results to a file"""
        if not self.thermal_analysis_results:
            QMessageBox.warning(self, "No Results", "No thermal analysis results to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Thermal Analysis Results", "", "Text files (*.txt);;All files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.thermal_results_text.toPlainText())
                
                QMessageBox.information(self, "Success", f"Thermal results exported to {file_path}")
                self.log_message(f"Exported thermal results to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export results:\n{str(e)}")

def main():
    """Main function to run the GUI"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Ion Trap Spectroscopy Fitting Tool")
    app.setApplicationVersion("2.0")
    
    # Create and show the main window
    window = SpectroscopyFitGUI()
    window.show()
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()