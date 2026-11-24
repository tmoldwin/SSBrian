"""Shared helpers for raw recurrent-dynamics sweeps."""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Dict, Iterable, Tuple

import numpy as np
from brian2 import ms, prefs, second
from scipy.stats import pearsonr

DEFAULT_TIME_WINDOW_MS = 50.0


def setup_numpy_device() -> None:
    """Force Brian to use the NumPy codegen backend."""
    prefs.codegen.target = "numpy"


@contextmanager
def suppress_stdout():
    """Temporarily silence stdout (used for large sweeps)."""
    original = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        os.dup2(original, 1)
        os.close(devnull)
        os.close(original)


def compute_firing_rate_correlation(
    spike_times_e: np.ndarray,
    spike_times_i: np.ndarray,
    time_window_ms: float = DEFAULT_TIME_WINDOW_MS,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Correlation between E/I population firing rates."""
    if len(spike_times_e) == 0 or len(spike_times_i) == 0:
        return 0.0, np.array([]), np.array([])

    t_start = min(spike_times_e.min(), spike_times_i.min())
    t_end = max(spike_times_e.max(), spike_times_i.max())
    bins = np.arange(t_start, t_end + time_window_ms, time_window_ms)

    hist_e, _ = np.histogram(spike_times_e, bins=bins)
    hist_i, _ = np.histogram(spike_times_i, bins=bins)

    time_window_s = time_window_ms / 1000.0
    rates_e = hist_e / time_window_s
    rates_i = hist_i / time_window_s

    if len(rates_e) > 1 and len(rates_i) > 1:
        corr, _ = pearsonr(rates_e, rates_i)
        if not np.isnan(corr):
            return corr, rates_e, rates_i
    return 0.0, rates_e, rates_i


def compute_pairwise_correlations(
    spike_times_by_neuron: Dict[int, np.ndarray],
    max_neurons: int = 50,
    time_window_ms: float = DEFAULT_TIME_WINDOW_MS,
) -> Iterable[float]:
    """Pairwise spike-count correlations between neurons."""
    if not spike_times_by_neuron:
        return []

    neuron_ids = list(spike_times_by_neuron.keys())
    if len(neuron_ids) > max_neurons:
        neuron_ids = random.sample(neuron_ids, max_neurons)

    all_times = np.concatenate([spike_times_by_neuron[n] for n in neuron_ids])
    if len(all_times) == 0:
        return []

    bins = np.arange(all_times.min(), all_times.max() + time_window_ms, time_window_ms)
    spike_counts = np.array(
        [np.histogram(spike_times_by_neuron[n], bins=bins)[0] for n in neuron_ids]
    )

    correlations = []
    for i in range(len(neuron_ids)):
        for j in range(i + 1, len(neuron_ids)):
            if np.std(spike_counts[i]) > 0 and np.std(spike_counts[j]) > 0:
                corr, _ = pearsonr(spike_counts[i], spike_counts[j])
                if not np.isnan(corr):
                    correlations.append(corr)
    return correlations


def analyze_network_activity(
    network,
    weight_rec: np.ndarray,
    n_rec_e: int,
    n_rec_i: int,
) -> Dict[str, float]:
    """Compute firing rates, CVs, balance, and correlations for a run."""
    total_time_s = network.T / second
    e_spikes = network.rec_spikes.count[:n_rec_e].sum()
    i_spikes = network.rec_spikes.count[n_rec_e:].sum()

    e_rate = e_spikes / (n_rec_e * total_time_s) if n_rec_e > 0 else 0.0
    i_rate = i_spikes / (n_rec_i * total_time_s) if n_rec_i > 0 else 0.0

    cv_e_list = []
    cv_i_list = []
    for neuron_id in range(n_rec_e + n_rec_i):
        spike_times = network.rec_spikes.spike_trains()[neuron_id]
        if len(spike_times) > 2:
            isis = np.diff(spike_times)
            if np.mean(isis) > 0:
                cv = np.std(isis) / np.mean(isis)
                (cv_e_list if neuron_id < n_rec_e else cv_i_list).append(cv)

    cv_e = float(np.mean(cv_e_list)) if cv_e_list else 0.0
    cv_i = float(np.mean(cv_i_list)) if cv_i_list else 0.0

    try:
        if (
            getattr(network, "rec_current_E", None) is not None
            and getattr(network, "rec_current_I", None) is not None
        ):
            mean_current_e = np.mean(np.abs(network.rec_current_E.I_syn_rec_E))
            mean_current_i = np.mean(np.abs(network.rec_current_I.I_syn_rec_I))
            balance_ratio = mean_current_i / mean_current_e if mean_current_e > 0 else 0.0
        else:
            balance_ratio = 0.0
    except Exception:
        balance_ratio = 0.0

    e_outgoing_weights = weight_rec[:, :n_rec_e].flatten()
    i_outgoing_weights = weight_rec[:, n_rec_e:].flatten()
    mean_e_weight = float(np.mean(e_outgoing_weights))
    mean_i_weight = float(np.mean(i_outgoing_weights))

    spike_times_all = None
    spike_neurons_all = None
    rate_corr = 0.0
    mean_pairwise_corr_e = 0.0
    mean_pairwise_corr_i = 0.0

    if getattr(network, "rec_spikes", None) is not None:
        spike_times_all = np.array(network.rec_spikes.t / ms)
        spike_neurons_all = np.array(network.rec_spikes.i)

        e_spike_mask = spike_neurons_all < n_rec_e
        i_spike_mask = ~e_spike_mask
        spike_times_e = spike_times_all[e_spike_mask]
        spike_times_i = spike_times_all[i_spike_mask]

        spike_times_by_neuron_e = {
            neuron_id: spike_times_all[spike_neurons_all == neuron_id]
            for neuron_id in range(n_rec_e)
            if np.any(spike_neurons_all == neuron_id)
        }
        spike_times_by_neuron_i = {
            neuron_id: spike_times_all[spike_neurons_all == neuron_id]
            for neuron_id in range(n_rec_e, n_rec_e + n_rec_i)
            if np.any(spike_neurons_all == neuron_id)
        }

        rate_corr, _, _ = compute_firing_rate_correlation(spike_times_e, spike_times_i)
        pairwise_corrs_e = compute_pairwise_correlations(spike_times_by_neuron_e)
        pairwise_corrs_i = compute_pairwise_correlations(spike_times_by_neuron_i)
        mean_pairwise_corr_e = float(np.mean(pairwise_corrs_e)) if pairwise_corrs_e else 0.0
        mean_pairwise_corr_i = float(np.mean(pairwise_corrs_i)) if pairwise_corrs_i else 0.0

    return {
        "e_rate": e_rate,
        "i_rate": i_rate,
        "cv_e": cv_e,
        "cv_i": cv_i,
        "balance_ratio": balance_ratio,
        "mean_e_weight": mean_e_weight,
        "mean_i_weight": mean_i_weight,
        "rate_correlation": rate_corr,
        "mean_pairwise_corr_e": mean_pairwise_corr_e,
        "mean_pairwise_corr_i": mean_pairwise_corr_i,
        "spike_times_all": spike_times_all,
        "spike_neurons_all": spike_neurons_all,
    }

