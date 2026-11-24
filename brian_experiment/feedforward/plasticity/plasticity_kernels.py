"""
Modular plasticity kernel system for temporal spike-timing dependent plasticity.

This module provides a flexible framework for implementing various plasticity rules
as functions of the temporal difference between pre- and post-synaptic spikes.
"""

from brian2 import *
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

# Note: clip is available in Brian2's namespace for synapse equations


class PlasticityKernel(ABC):
    """
    Base class for plasticity kernels.
    
    A plasticity kernel defines how synaptic weights change based on the temporal
    difference between pre-synaptic and post-synaptic spikes (delta_t = t_post - t_pre).
    
    Subclasses should implement the kernel() method which returns the weight change
    as a function of the temporal difference.
    """
    
    def __init__(self, learning_rate=1.0, weight_bounds=None):
        """
        Initialize plasticity kernel.
        
        Parameters:
        -----------
        learning_rate : float
            Global scaling factor for weight changes
        weight_bounds : tuple (min, max) or None
            Optional bounds on weight values. If None, no bounds are applied.
        """
        self.learning_rate = learning_rate
        self.weight_bounds = weight_bounds  # (min, max) or None
        
    @abstractmethod
    def kernel(self, delta_t):
        """
        Compute weight change as a function of temporal difference.
        
        Parameters:
        -----------
        delta_t : Quantity (time)
            Temporal difference: t_post - t_pre (positive when post fires after pre)
            
        Returns:
        --------
        float
            Weight change (will be scaled by learning_rate)
        """
        pass
    
    def apply_bounds(self, weight):
        """Apply weight bounds if specified."""
        if self.weight_bounds is not None:
            w_min, w_max = self.weight_bounds
            return clip(weight, w_min, w_max)
        return weight
    
    def get_brian2_equations(self):
        """
        Get Brian2 equations string for synapse model.
        This should be overridden by subclasses that need additional state variables.
        
        Returns:
        --------
        str
            Brian2 equations string (can be empty if no state variables needed)
        """
        return ''
    
    def get_brian2_on_pre(self):
        """
        Get Brian2 on_pre code for weight updates.
        This should be overridden by subclasses that need custom update logic.
        
        Returns:
        --------
        str
            Brian2 on_pre code string
        """
        return ''
    
    def get_brian2_on_post(self):
        """
        Get Brian2 on_post code for weight updates.
        This should be overridden by subclasses that need custom update logic.
        
        Returns:
        --------
        str
            Brian2 on_post code string
        """
        return ''
    
    def get_brian2_namespace(self):
        """
        Get namespace dictionary for Brian2 equations.
        This should be overridden by subclasses that need to pass parameters.
        
        Returns:
        --------
        dict
            Dictionary of parameter names and values
        """
        return {}
    
    def visualize(self, delta_t_range=(-50, 50), n_points=200, ax=None, label=None, **plot_kwargs):
        """
        Visualize the plasticity kernel as a function of temporal difference.
        
        Parameters:
        -----------
        delta_t_range : tuple (min_ms, max_ms)
            Range of temporal differences to plot (in milliseconds)
        n_points : int
            Number of points to evaluate
        ax : matplotlib.axes.Axes or None
            Axes to plot on. If None, creates a new figure.
        label : str or None
            Label for the plot. If None, uses kernel class name.
        **plot_kwargs
            Additional arguments to pass to plt.plot()
            
        Returns:
        --------
        matplotlib.axes.Axes
            The axes object used for plotting
        """
        # Create time array
        delta_t_ms = np.linspace(delta_t_range[0], delta_t_range[1], n_points)
        delta_t = delta_t_ms * ms
        
        # Evaluate kernel
        weight_changes = self.kernel(delta_t)
        
        # Convert to numpy array if needed
        if hasattr(weight_changes, 'magnitude'):
            weight_changes = weight_changes.magnitude
        else:
            weight_changes = np.array(weight_changes)
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot
        if label is None:
            label = self.__class__.__name__
        
        default_kwargs = {'linewidth': 2, 'alpha': 0.8}
        default_kwargs.update(plot_kwargs)
        ax.plot(delta_t_ms, weight_changes, label=label, **default_kwargs)
        
        # Add zero line
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Labels
        ax.set_xlabel(r'$\Delta t$ (ms) [t_post - t_pre]', fontsize=12)
        ax.set_ylabel('Weight Change', fontsize=12)
        ax.set_title(f'Plasticity Kernel: {label}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax


class STDPKernel(PlasticityKernel):
    """
    Spike-Timing Dependent Plasticity (STDP) kernel.
    
    Classic STDP rule:
    - If pre fires before post (delta_t > 0): LTP (Long-Term Potentiation)
    - If post fires before pre (delta_t < 0): LTD (Long-Term Depression)
    
    Weight change: A_pre * exp(-delta_t/tau_pre) for delta_t > 0
                   A_post * exp(delta_t/tau_post) for delta_t < 0
    """
    
    def __init__(self, eta_pre=0.5, eta_post=0.525, tau_pre=20*ms, tau_post=20*ms,
                 weight_bounds=None):
        """
        Initialize STDP kernel.
        
        Parameters:
        -----------
        eta_pre : float in [0,1]
            LTP scale applied on pre spikes
        eta_post : float in [0,1]
            LTD scale applied on post spikes
        tau_pre : Quantity (time)
            Time constant for LTP decay
        tau_post : Quantity (time)
            Time constant for LTD decay
        weight_bounds : tuple (min, max) or None
            Optional weight bounds
        """
        # learning_rate is absorbed into pre/post; keep global scale = 1.0
        super().__init__(learning_rate=1.0, weight_bounds=weight_bounds)
        self.eta_pre = eta_pre
        self.eta_post = eta_post
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        
    def kernel(self, delta_t):
        """STDP kernel function."""
        # Convert to numpy array if needed for vectorized operations
        if isinstance(delta_t, np.ndarray) or hasattr(delta_t, '__len__'):
            # Handle Brian2 Quantity arrays – always operate in milliseconds
            if hasattr(delta_t, 'magnitude'):
                # Work in seconds consistently to avoid unit scale mistakes
                delta_t_vals = np.asarray(delta_t / second, dtype=float)
            else:
                delta_t_vals = np.array([float(dt) if hasattr(dt, '__float__') else dt for dt in delta_t], dtype=float)

            tau_pre_val = float(self.tau_pre / second)
            tau_post_val = float(self.tau_post / second)
            
            result = np.zeros_like(delta_t_vals, dtype=float)
            mask_positive = delta_t_vals >= 0
            mask_negative = delta_t_vals < 0
            result[mask_positive] = self.eta_pre * np.exp(-delta_t_vals[mask_positive] / tau_pre_val)
            result[mask_negative] = self.eta_post * np.exp(delta_t_vals[mask_negative] / tau_post_val)
            return result * self.learning_rate
        else:
            # Scalar case
            if hasattr(delta_t, 'magnitude'):
                delta_t_val = float(delta_t / second)
            else:
                delta_t_val = float(delta_t) if hasattr(delta_t, '__float__') else delta_t
            
            tau_pre_val = float(self.tau_pre / second)
            tau_post_val = float(self.tau_post / second)
            
            if delta_t_val > 0:
                return self.learning_rate * self.eta_pre * np.exp(-delta_t_val / tau_pre_val)
            elif delta_t_val < 0:
                return self.learning_rate * self.eta_post * np.exp(delta_t_val / tau_post_val)
            else:
                return 0.0
    
    def get_brian2_equations(self):
        """Get Brian2 equations for STDP with trace variables."""
        return '''
        dapre/dt = -apre/taupre : 1 (clock-driven)
        dapost/dt = -apost/taupost : 1 (clock-driven)
        '''
    
    def get_brian2_on_pre(self):
        """Get Brian2 on_pre code for STDP with soft bounds."""
        if self.weight_bounds:
            w_min = float(self.weight_bounds[0] / amp)
            w_max = float(self.weight_bounds[1] / amp)
            return f'''
        apre += {self.eta_pre}
        dw = apost * {self.eta_pre} * {self.learning_rate} * ({w_max}*amp - w_syn)
        w_syn = clip(w_syn + dw, {w_min}*amp, {w_max}*amp)
        '''
        else:
            return f'''
        apre += {self.eta_pre}
        w_syn = w_syn + apost * {self.eta_pre} * {self.learning_rate} * amp
        '''
    
    def get_brian2_on_post(self):
        """Get Brian2 on_post code for STDP with soft bounds."""
        if self.weight_bounds:
            w_min = float(self.weight_bounds[0] / amp)
            w_max = float(self.weight_bounds[1] / amp)
            return f'''
        apost += {self.eta_post}
        dw = - apre * {self.eta_post} * {self.learning_rate} * (w_syn - {w_min}*amp)
        w_syn = clip(w_syn + dw, {w_min}*amp, {w_max}*amp)
        '''
        else:
            return f'''
        apost += {self.eta_post}
        w_syn = w_syn - apre * {self.eta_post} * {self.learning_rate} * amp
        '''
    
    def get_brian2_namespace(self):
        """Get namespace for STDP parameters."""
        return {
            'taupre': self.tau_pre,
            'taupost': self.tau_post
        }
    
    def visualize(self, delta_t_range=(-100, 100), n_points=1000, ax=None, **plot_kwargs):
        """
        Visualize STDP kernel with LTP and LTD regions highlighted.
        Shows smooth curved bumps: up bump (LTP) for positive delta_t, down bump (LTD) for negative delta_t.
        
        Parameters:
        -----------
        delta_t_range : tuple (min_ms, max_ms)
            Range of temporal differences to plot (in milliseconds)
        n_points : int
            Number of points to evaluate (more points = smoother curves)
        ax : matplotlib.axes.Axes or None
            Axes to plot on. If None, creates a new figure.
        **plot_kwargs
            Additional arguments to pass to plt.plot()
        
        Returns:
        --------
        matplotlib.axes.Axes
            The axes object used for plotting
        """
        # Create time array with many points for smooth curves
        # Use more points for smoother curves
        delta_t_ms = np.linspace(delta_t_range[0], delta_t_range[1], n_points)
        delta_t = delta_t_ms * ms
        
        # Evaluate kernel - this should produce smooth exponential curves
        weight_changes = self.kernel(delta_t)
        
        # Convert to numpy array
        if hasattr(weight_changes, 'magnitude'):
            weight_changes = weight_changes.magnitude
        else:
            weight_changes = np.array(weight_changes)
        
        # Ensure we have a smooth continuous array (no NaN or inf)
        weight_changes = np.nan_to_num(weight_changes, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot smooth continuous curve
        default_kwargs = {'linewidth': 2.5, 'color': 'blue', 'label': 'STDP'}
        default_kwargs.update(plot_kwargs)
        ax.plot(delta_t_ms, weight_changes, **default_kwargs)
        
        # Add zero lines
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Labels
        ax.set_xlabel(r'$\Delta t$ (ms) [t_post - t_pre]', fontsize=12)
        ax.set_ylabel('Weight Change', fontsize=12)
        ax.set_title(f'STDP Kernel (τ_pre={self.tau_pre/ms:.1f}ms, τ_post={self.tau_post/ms:.1f}ms)', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax


class HebbianKernel(PlasticityKernel):
    """
    Hebbian plasticity: "neurons that fire together, wire together; out of sync lose your link"
    
    Mexican hat function: potentiation for synchronous firing, depression for asynchronous firing.
    - Narrow positive bump for in-sync firing (LTP)
    - Wider negative region for out-of-sync firing (LTD)
    """
    
    def __init__(self, A_hebb=0.01, tau_hebb=20*ms, tau_hebb_wide=40*ms, learning_rate=1.0, weight_bounds=None):
        """
        Initialize Hebbian kernel.
        
        Parameters:
        -----------
        A_hebb : float
            Amplitude for Hebbian potentiation (narrow bump)
        tau_hebb : Quantity (time)
            Time constant for narrow potentiation window
        tau_hebb_wide : Quantity (time)
            Time constant for wider depression window
        learning_rate : float
            Global scaling factor
        weight_bounds : tuple (min, max) or None
            Optional weight bounds
        """
        super().__init__(learning_rate, weight_bounds)
        self.A_hebb = A_hebb
        self.tau_hebb = tau_hebb
        self.tau_hebb_wide = tau_hebb_wide
        
    def kernel(self, delta_t):
        """Hebbian kernel: Mexican hat (narrow positive bump, wider negative surround)."""
        if isinstance(delta_t, np.ndarray) or hasattr(delta_t, '__len__'):
            # Handle Brian2 Quantity arrays
            if hasattr(delta_t, 'magnitude'):
                delta_t_vals = np.asarray(delta_t / second, dtype=float)
            else:
                delta_t_vals = np.array([float(dt) if hasattr(dt, '__float__') else dt for dt in delta_t], dtype=float)
            
            tau_narrow_val = float(self.tau_hebb / second)
            tau_wide_val = float(self.tau_hebb_wide / second)
            
            # Mexican hat: narrow positive - wider negative
            # Positive bump for synchronized firing
            narrow_bump = self.A_hebb * np.exp(-np.abs(delta_t_vals) / tau_narrow_val)
            # Negative surround for asynchronous firing (balanced at 0.5)
            wide_surround = (self.A_hebb * 0.5) * np.exp(-np.abs(delta_t_vals) / tau_wide_val)
            
            result = narrow_bump - wide_surround
            return result * self.learning_rate
        else:
            # Scalar case
            if hasattr(delta_t, 'magnitude'):
                delta_t_val = float(delta_t / second)
            else:
                delta_t_val = float(delta_t) if hasattr(delta_t, '__float__') else delta_t
            
            tau_narrow_val = float(self.tau_hebb / second)
            tau_wide_val = float(self.tau_hebb_wide / second)
            
            # Mexican hat (balanced at 0.5)
            narrow_bump = self.A_hebb * np.exp(-np.abs(delta_t_val) / tau_narrow_val)
            wide_surround = (self.A_hebb * 0.5) * np.exp(-np.abs(delta_t_val) / tau_wide_val)
            
            return self.learning_rate * (narrow_bump - wide_surround)
    
    def get_brian2_equations(self):
        """Get Brian2 equations for Hebbian plasticity - Mexican hat with two traces."""
        return '''
        dapost_narrow/dt = -apost_narrow/tauhebb : 1 (clock-driven)
        dapost_wide/dt = -apost_wide/tauhebbwide : 1 (clock-driven)
        '''
    
    def get_brian2_on_pre(self):
        """Get Brian2 on_pre code for Hebbian (no change on pre spike)."""
        return ''
    
    def get_brian2_on_post(self):
        """Get Brian2 on_post code for Hebbian with soft bounds - Mexican hat."""
        # Mexican hat: increment traces with correct amplitudes
        # Narrow gets full amplitude, wide gets 0.5 amplitude
        # Then net signal = narrow - wide (no additional scaling needed)
        if self.weight_bounds:
            w_min = float(self.weight_bounds[0] / amp)
            w_max = float(self.weight_bounds[1] / amp)
            return f'''
        apost_narrow += {self.A_hebb}
        apost_wide += {0.5 * self.A_hebb}
        dw_hebb = (apost_narrow - apost_wide) * {self.learning_rate}
        dw = dw_hebb * int(dw_hebb > 0) * ({w_max}*amp - w_syn) + dw_hebb * int(dw_hebb < 0) * (w_syn - {w_min}*amp)
        w_syn = clip(w_syn + dw, {w_min}*amp, {w_max}*amp)
        '''
        else:
            return f'''
        apost_narrow += {self.A_hebb}
        apost_wide += {0.5 * self.A_hebb}
        dw_hebb = (apost_narrow - apost_wide) * {self.learning_rate}
        w_syn = w_syn + dw_hebb * amp
        '''
    
    def get_brian2_namespace(self):
        """Get namespace for Hebbian parameters."""
        return {
            'tauhebb': self.tau_hebb,
            'tauhebbwide': self.tau_hebb_wide
        }
    
    def visualize(self, delta_t_range=(-100, 100), n_points=1000, ax=None, **plot_kwargs):
        """
        Visualize Hebbian kernel (Mexican hat: narrow positive, wide negative).
        Shows potentiation for synchronous firing and depression for asynchronous firing.
        
        Parameters:
        -----------
        delta_t_range : tuple (min_ms, max_ms)
            Range of temporal differences to plot (in milliseconds)
        n_points : int
            Number of points to evaluate (more points = smoother curves)
        ax : matplotlib.axes.Axes or None
            Axes to plot on. If None, creates a new figure.
        **plot_kwargs
            Additional arguments to pass to plt.plot()
        
        Returns:
        --------
        matplotlib.axes.Axes
            The axes object used for plotting
        """
        # Create time array with many points for smooth curves
        delta_t_ms = np.linspace(delta_t_range[0], delta_t_range[1], n_points)
        delta_t = delta_t_ms * ms
        
        # Evaluate kernel
        weight_changes = self.kernel(delta_t)
        
        # Convert to numpy array
        if hasattr(weight_changes, 'magnitude'):
            weight_changes = weight_changes.magnitude
        else:
            weight_changes = np.array(weight_changes)
        
        # Ensure we have a smooth continuous array (no NaN or inf)
        weight_changes = np.nan_to_num(weight_changes, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot smooth continuous curve
        default_kwargs = {'linewidth': 2.5, 'color': 'green', 'label': 'Hebbian (Mexican Hat)'}
        default_kwargs.update(plot_kwargs)
        ax.plot(delta_t_ms, weight_changes, **default_kwargs)
        
        # Add zero lines
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Labels
        ax.set_xlabel(r'$\Delta t$ (ms) [t_post - t_pre]', fontsize=12)
        ax.set_ylabel('Weight Change', fontsize=12)
        ax.set_title(f'Hebbian Kernel (Mexican Hat: τ_narrow={self.tau_hebb/ms:.1f}ms, τ_wide={self.tau_hebb_wide/ms:.1f}ms)', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax


class AntiHebbianKernel(PlasticityKernel):
    """
    Anti-Hebbian plasticity: "neurons that fire together, unwire"
    
    Opposite of Hebbian: weight decreases when pre and post fire together.
    """
    
    def __init__(self, A_anti=0.01, tau_anti=20*ms, learning_rate=1.0, weight_bounds=None):
        """
        Initialize Anti-Hebbian kernel.
        
        Parameters:
        -----------
        A_anti : float
            Amplitude for anti-Hebbian depression (typically positive, but weight change is negative)
        tau_anti : Quantity (time)
            Time constant for anti-Hebbian window
        learning_rate : float
            Global scaling factor
        weight_bounds : tuple (min, max) or None
            Optional weight bounds
        """
        super().__init__(learning_rate, weight_bounds)
        self.A_anti = A_anti
        self.tau_anti = tau_anti
        
    def kernel(self, delta_t):
        """Anti-Hebbian kernel: symmetric negative changes for both directions."""
        if isinstance(delta_t, np.ndarray) or hasattr(delta_t, '__len__'):
            # Handle Brian2 Quantity arrays – always operate in milliseconds
            if hasattr(delta_t, 'magnitude'):
                delta_t_vals = np.asarray(delta_t / second, dtype=float)
            else:
                delta_t_vals = np.array([float(dt) if hasattr(dt, '__float__') else dt for dt in delta_t], dtype=float)
            
            tau_anti_val = float(self.tau_anti / second)
            
            # Symmetric: use absolute value of delta_t, always negative
            result = -self.A_anti * np.exp(-np.abs(delta_t_vals) / tau_anti_val)
            return result * self.learning_rate
        else:
            # Scalar case
            if hasattr(delta_t, 'magnitude'):
                delta_t_val = float(delta_t / second)
            else:
                delta_t_val = float(delta_t) if hasattr(delta_t, '__float__') else delta_t
            
            tau_anti_val = float(self.tau_anti / second)
            
            # Symmetric: use absolute value, always negative
            return -self.learning_rate * self.A_anti * np.exp(-np.abs(delta_t_val) / tau_anti_val)
    
    def get_brian2_equations(self):
        """Get Brian2 equations for Anti-Hebbian plasticity."""
        return '''
        dapost/dt = -apost/tauanti : 1 (clock-driven)
        '''
    
    def get_brian2_on_pre(self):
        """Get Brian2 on_pre code for Anti-Hebbian (no change on pre spike)."""
        return ''
    
    def get_brian2_on_post(self):
        """Get Brian2 on_post code for Anti-Hebbian with soft bounds."""
        if self.weight_bounds:
            w_min = float(self.weight_bounds[0] / amp)
            w_max = float(self.weight_bounds[1] / amp)
            # Soft bounds: depression scales by (w - w_min) for asymptotic approach to minimum
            return f'''
        apost += {self.A_anti}
        dw = -apost * {self.learning_rate} * (w_syn - {w_min}*amp)
        w_syn = clip(w_syn + dw, {w_min}*amp, {w_max}*amp)
        '''
        else:
            return f'''
        apost += {self.A_anti}
        w_syn = w_syn - apost * {self.learning_rate} * amp
        '''
    
    def get_brian2_namespace(self):
        """Get namespace for Anti-Hebbian parameters."""
        return {
            'tauanti': self.tau_anti
        }
    
    def visualize(self, delta_t_range=(-100, 100), n_points=1000, ax=None, **plot_kwargs):
        """
        Visualize Anti-Hebbian kernel (symmetric inverted bump centered at delta_t=0).
        Shows a smooth curved inverted bump that peaks (most negative) at delta_t=0 and decays symmetrically.
        
        Parameters:
        -----------
        delta_t_range : tuple (min_ms, max_ms)
            Range of temporal differences to plot (in milliseconds)
        n_points : int
            Number of points to evaluate (more points = smoother curves)
        ax : matplotlib.axes.Axes or None
            Axes to plot on. If None, creates a new figure.
        **plot_kwargs
            Additional arguments to pass to plt.plot()
        
        Returns:
        --------
        matplotlib.axes.Axes
            The axes object used for plotting
        """
        # Create time array with many points for smooth curves
        # Use more points for smoother curves
        delta_t_ms = np.linspace(delta_t_range[0], delta_t_range[1], n_points)
        delta_t = delta_t_ms * ms
        
        # Evaluate kernel - this should produce smooth exponential curves
        weight_changes = self.kernel(delta_t)
        
        # Convert to numpy array
        if hasattr(weight_changes, 'magnitude'):
            weight_changes = weight_changes.magnitude
        else:
            weight_changes = np.array(weight_changes)
        
        # Ensure we have a smooth continuous array (no NaN or inf)
        weight_changes = np.nan_to_num(weight_changes, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot smooth continuous curve
        default_kwargs = {'linewidth': 2.5, 'color': 'red', 'label': 'Anti-Hebbian'}
        default_kwargs.update(plot_kwargs)
        ax.plot(delta_t_ms, weight_changes, **default_kwargs)
        
        # Add zero lines
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Labels
        ax.set_xlabel(r'$\Delta t$ (ms) [t_post - t_pre]', fontsize=12)
        ax.set_ylabel('Weight Change', fontsize=12)
        ax.set_title(f'Anti-Hebbian Kernel (Symmetric Inverted Bump, τ={self.tau_anti/ms:.1f}ms)', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax


class CustomKernel(PlasticityKernel):
    """
    Custom plasticity kernel that accepts an arbitrary function.
    
    This allows users to define their own plasticity rules as Python functions.
    """
    
    def __init__(self, kernel_function, learning_rate=1.0, weight_bounds=None, 
                 brian2_equations='', brian2_on_pre='', brian2_on_post=''):
        """
        Initialize custom kernel.
        
        Parameters:
        -----------
        kernel_function : callable
            Function that takes delta_t (time) and returns weight change
        learning_rate : float
            Global scaling factor
        weight_bounds : tuple (min, max) or None
            Optional weight bounds
        brian2_equations : str
            Optional Brian2 equations string for state variables
        brian2_on_pre : str
            Optional Brian2 on_pre code
        brian2_on_post : str
            Optional Brian2 on_post code
        """
        super().__init__(learning_rate, weight_bounds)
        self.kernel_function = kernel_function
        self._brian2_equations = brian2_equations
        self._brian2_on_pre = brian2_on_pre
        self._brian2_on_post = brian2_on_post
        
    def kernel(self, delta_t):
        """Apply custom kernel function."""
        result = self.kernel_function(delta_t)
        return result * self.learning_rate
    
    def get_brian2_equations(self):
        """Get custom Brian2 equations."""
        return self._brian2_equations
    
    def get_brian2_on_pre(self):
        """Get custom Brian2 on_pre code."""
        return self._brian2_on_pre
    
    def get_brian2_on_post(self):
        """Get custom Brian2 on_post code."""
        return self._brian2_on_post


# Convenience function to create standard kernels
def create_stdp_kernel(eta_pre=0.5, eta_post=0.525, tau_pre=20*ms, tau_post=20*ms, 
                       weight_bounds=None):
    """Create an STDP kernel with default parameters (eta_pre/eta_post in [0,1])."""
    return STDPKernel(eta_pre, eta_post, tau_pre, tau_post, weight_bounds)


def create_hebbian_kernel(A_hebb=0.01, tau_hebb=20*ms, learning_rate=1.0, weight_bounds=None):
    """Create a Hebbian kernel with default parameters."""
    return HebbianKernel(A_hebb, tau_hebb, learning_rate, weight_bounds)


def create_anti_hebbian_kernel(A_anti=0.01, tau_anti=20*ms, learning_rate=1.0, weight_bounds=None):
    """Create an Anti-Hebbian kernel with default parameters."""
    return AntiHebbianKernel(A_anti, tau_anti, learning_rate, weight_bounds)

