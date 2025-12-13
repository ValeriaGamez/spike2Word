import numpy as np
from sim_params import sim_dict
import os

# Absolute path to the directory of this script
here = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(here, "data")
file_path = os.path.join(data_dir, "train_t12.2022.04.28_sentence_4.npy")

# --------------------------------------------------------------------
# Load stimulus data
# --------------------------------------------------------------------
sentence_data = np.load(file_path)

# Ensure data is 2D: (num_bins, num_channels)
if sentence_data.ndim == 1:
    # If the file only had a single column originally
    sentence_data = sentence_data[:, np.newaxis]

sentence_data = sentence_data.astype(float)

num_bins, num_channels = sentence_data.shape

# --------------------------------------------------------------------
# Build time axis and convert to rates
# --------------------------------------------------------------------
bin_width_ms = 20.0  # ms per bin

t_presim = float(sim_dict.get('t_presim', 0.0))

# Time points for each bin (same for all channels)
rate_times_ms = np.arange(num_bins, dtype=float) * bin_width_ms + t_presim

# Convert counts per bin into Hz:
#    (spikes per bin) * (1000 ms / bin_width_ms) = spikes per second (Hz)
rate_values_hz = sentence_data * (1000.0 / bin_width_ms)  # shape: (num_bins, num_channels)

# Total simulation time contributed by the stimulus (excluding t_presim)
t_sim = num_bins * bin_width_ms


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------
def getNumChannels():
    """Return the number of stimulus channels (columns)."""
    return num_channels


def getRateTimes(channel=None):
    """
    Return the time points in ms for the bins.
    The time axis is the same for all channels, so `channel` is ignored.
    """
    return rate_times_ms


def getRateValues(channel=0):
    """
    Return the rate values (Hz) for a given channel index.
    channel: int, 0-based index (0 <= channel < num_channels)
    """
    return rate_values_hz[:, channel]


def getTotalms():
    """Total simulation time in ms including presimulation period."""
    return t_sim + t_presim


def getStartms():
    """Start time in ms for the stimulus."""
    return t_presim


def update_sim_dict():
    """Update sim_dict with the total simulation time contributed by the stimulus."""
    sim_dict.update({'t_sim': t_sim})
