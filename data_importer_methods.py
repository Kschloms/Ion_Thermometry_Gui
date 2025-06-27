import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
# Helper functions to import and plot data from Rabi scan and spectroscopy HDF5 files
def import_rabi_scan(filepath):
    """
    Import data from a Rabi scan HDF5 file.
    """
    with h5py.File(filepath, 'r') as f:
        dataset = f['datasets']
        n_points = dataset['n_points'][:]
        pulse_length = np.array(dataset['time_shelv_ns'][0,0,0]) / 1e3 # ns to us
        dark_counts = np.array(dataset['dark'][:])
        excitation_prob = dark_counts / n_points
        excitation_prob_err = np.sqrt(excitation_prob * (1 - excitation_prob) / n_points)
        return pulse_length, excitation_prob, excitation_prob_err
    
def plot_rabi_scan(ax: plt.axes, pulse_length, excitation_prob, excitation_prob_err, title=None):
    """
    Plot the Rabi scan data.
    """
    ax.errorbar(pulse_length, excitation_prob, yerr=excitation_prob_err, fmt='o', capsize=2, ls='--')
    ax.set_xlabel('Pulse Length ($\mu s$)')
    ax.set_ylabel('Excitation Probability')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Rabi Scan Data')
    ax.legend()
    ax.grid()

def import_spectroscopy(filepath):
    """
    Import data from a spectroscopy HDF5 file.
    """
    with h5py.File(filepath, 'r') as f:
        dataset = f['datasets']
        n_points = dataset['n_points'][:]
        frequency = np.array(dataset['freqs'][:]) # Hz
        dark_counts = np.array(dataset['dark'][:])
        excitation_prob = dark_counts / n_points
        excitation_prob_err = np.sqrt(excitation_prob * (1 - excitation_prob) / n_points)

        # Suppose 'expid_bytes' is your value from f['expid'][()]
        expid_bytes = f['expid'][()]  # This is a bytes object

        # Decode and load as dict
        expid_dict = json.loads(expid_bytes.decode('utf-8'))

        arguments = expid_dict["arguments"]
        freq_carrier2_MHz = arguments.get("freq_carrier2_MHz", None)
        freq_center_MHz = arguments.get("freq_center_MHz", None)

        detuning = (freq_carrier2_MHz * 1e6) - frequency
        relative_detuning = detuning - freq_center_MHz * 1e6

        relative_detuning = relative_detuning * 2
        return detuning, relative_detuning, excitation_prob, excitation_prob_err, freq_center_MHz
    
def plot_spectroscopy(ax: plt.axes, detuning, excitation_prob, excitation_prob_err, relative_center, title=None):
    ax.set_xlabel(f'Detuning relative to {relative_center} MHz (kHz) ')

    detuning /= 1e3  # Convert to kHz
    """
    Plot the spectroscopy data.
    """
    ax.errorbar(detuning, excitation_prob, yerr=excitation_prob_err, fmt='o', capsize=2, ls='', markersize=2, label='Data')
    ax.set_ylabel('Excitation Probability')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Spectroscopy Data')
    ax.grid()