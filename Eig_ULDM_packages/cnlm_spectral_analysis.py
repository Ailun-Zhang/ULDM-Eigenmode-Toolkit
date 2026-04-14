"""
Spectral analysis tools for c_{nlm}(t) time series.

This module provides functions for:
- FFT-based spectral analysis of complex time series
- Peak detection with proper boundary handling
- 4-panel EDA visualization
- PRD-ready figure export

Author: Auto-generated from compute_c_nlm_updated.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SpectralAnalysisResult:
    """Container for spectral analysis results."""
    # Time and signal data
    t: np.ndarray
    x: np.ndarray
    x_complex: np.ndarray
    
    # FFT results
    freq: np.ndarray
    mag: np.ndarray
    
    # Sampling info
    N: int
    dt_med: float
    Tspan: float
    fft_note: str
    
    # Peak detection results
    peaks_pos: List[Tuple[float, float]]
    peaks_neg: List[Tuple[float, float]]
    dc_mag: float
    dc_is_peak: bool
    
    # Pattern classification
    peak_pattern: str
    significant_peaks: List[Tuple[float, float]]
    strong_peaks_2: List[Tuple[float, float]]
    
    # Envelope analysis
    f_env: float
    
    # Phase analysis (for complex signals)
    phase: np.ndarray
    phase_slope: float
    phase_intercept: float
    f_phase: float
    phase_lin: np.ndarray
    phase_available: bool
    
    # Metadata
    key: Tuple[int, int, int]
    mode_label: str
    is_real_signal: bool
    source_tag: str
    norm_tag: str


def top_peaks_improved(freq_arr: np.ndarray, mag_arr: np.ndarray, 
                       k: int = 6, rel_thr: float = 0.05, 
                       check_boundary: bool = True) -> List[Tuple[float, float]]:
    """
    Detect peaks in spectrum with proper handling of boundary peaks.
    
    Key improvements:
    1. Uses scipy.signal.find_peaks for interior peaks
    2. For boundary elements: only considers them as peaks if they are TRUE local maxima
       - First element: must be > second element (not just >=)
       - Last element: must be > second-to-last element
       This avoids false positives at monotonic transitions.
    
    Parameters
    ----------
    freq_arr : array-like
        Frequency array
    mag_arr : array-like
        Magnitude array (|FFT|)
    k : int
        Maximum number of peaks to return
    rel_thr : float
        Relative threshold (fraction of max magnitude)
    check_boundary : bool
        Whether to check boundary elements as potential peaks
    
    Returns
    -------
    list of (freq, mag) tuples, sorted by magnitude descending
    """
    if freq_arr.size == 0:
        return []
    maxv = float(np.max(mag_arr))
    if not np.isfinite(maxv) or maxv <= 0:
        return []
    thr = maxv * rel_thr
    
    out = []
    
    # Try scipy.signal.find_peaks for interior peaks
    try:
        from scipy.signal import find_peaks
        peaks_idx, props = find_peaks(mag_arr, height=thr)
        if peaks_idx.size > 0:
            for i in peaks_idx:
                out.append((float(freq_arr[i]), float(mag_arr[i])))
    except Exception:
        # Fallback for interior peaks: simple local maxima detection
        for i in range(1, len(mag_arr) - 1):
            if mag_arr[i] >= thr and mag_arr[i] > mag_arr[i-1] and mag_arr[i] > mag_arr[i+1]:
                out.append((float(freq_arr[i]), float(mag_arr[i])))
    
    # Check boundary elements as potential peaks (stricter criterion)
    if check_boundary and len(mag_arr) >= 3:
        if mag_arr[0] >= thr and mag_arr[0] > mag_arr[1]:
            out.append((float(freq_arr[0]), float(mag_arr[0])))
        if mag_arr[-1] >= thr and mag_arr[-1] > mag_arr[-2]:
            out.append((float(freq_arr[-1]), float(mag_arr[-1])))
    
    # Remove duplicates and sort by amplitude descending
    seen_freqs = set()
    unique_out = []
    for (f, a) in out:
        f_rounded = round(f, 10)
        if f_rounded not in seen_freqs:
            seen_freqs.add(f_rounded)
            unique_out.append((f, a))
    
    unique_out.sort(key=lambda z: z[1], reverse=True)
    return unique_out[:k]


def analyze_cnlm_spectrum(
    t_data: np.ndarray,
    c_data: np.ndarray,
    key: Tuple[int, int, int],
    analysis_mode: str = 'complex',
    normalization: Optional[float] = None,
    apply_hann_window: bool = True,
    detrend_mean: bool = True,
    peak_rel_threshold: float = 0.05,
    top_k_peaks: int = 6,
) -> SpectralAnalysisResult:
    """
    Perform comprehensive spectral analysis on a c_{nlm}(t) time series.
    
    Parameters
    ----------
    t_data : np.ndarray
        Time array
    c_data : np.ndarray
        Complex coefficient array c_{nlm}(t)
    key : tuple
        (n, ℓ, m) indices
    analysis_mode : str
        'complex' (full c), 'real' (Re[c]), or 'imag' (Im[c])
    normalization : float, optional
        If provided, divide c by sqrt(normalization)
    apply_hann_window : bool
        Apply Hann window before FFT
    detrend_mean : bool
        Remove mean before FFT
    peak_rel_threshold : float
        Relative threshold for peak detection
    top_k_peaks : int
        Max peaks to detect per region
    
    Returns
    -------
    SpectralAnalysisResult
        Container with all analysis results
    """
    # Clean and sort data
    t = np.asarray(t_data).copy()
    x_complex = np.asarray(c_data, dtype=np.complex128).copy()
    
    order = np.argsort(t)
    t = t[order]
    x_complex = x_complex[order]
    
    mask = np.isfinite(t) & np.isfinite(x_complex.real) & np.isfinite(x_complex.imag)
    t = t[mask]
    x_complex = x_complex[mask]
    
    if t.size < 8:
        raise ValueError(f"Too few samples (N={t.size}) for stable spectral analysis.")
    
    # Normalization
    norm_tag = ''
    if normalization is not None and normalization > 0:
        x_complex = x_complex / np.sqrt(normalization)
        norm_tag = ' (normalized)'
    
    # Select analysis target
    if analysis_mode == 'real':
        x = x_complex.real.astype(np.float64)
        mode_label = 'Re[c]'
        is_real_signal = True
    elif analysis_mode == 'imag':
        x = x_complex.imag.astype(np.float64)
        mode_label = 'Im[c]'
        is_real_signal = True
    else:
        x = x_complex
        mode_label = 'c (complex)'
        is_real_signal = False
    
    # Sampling info
    dt_arr = np.diff(t)
    dt_med = float(np.median(dt_arr))
    if dt_med <= 0:
        raise ValueError("Time series not strictly increasing.")
    
    rel_nonuniform = float(np.max(np.abs(dt_arr - dt_med)) / dt_med)
    needs_resample = rel_nonuniform > 1e-3
    
    t_fft = t
    x_fft = x
    fft_note = ""
    
    if needs_resample:
        t_fft = np.linspace(t[0], t[-1], t.size, endpoint=True)
        if is_real_signal:
            x_fft = np.interp(t_fft, t, x)
        else:
            xr = np.interp(t_fft, t, x.real)
            xi = np.interp(t_fft, t, x.imag)
            x_fft = xr + 1j * xi
        fft_note = f"FFT: resampled (non-uniformity ≈ {rel_nonuniform:.2e})"
    else:
        fft_note = f"FFT: direct (non-uniformity ≈ {rel_nonuniform:.2e})"
    
    N = int(t_fft.size)
    Tspan = float(t_fft[-1] - t_fft[0])
    
    # FFT with optional windowing
    x_spec = x_fft.copy()
    if detrend_mean:
        x_spec = x_spec - np.mean(x_spec)
    if apply_hann_window:
        x_spec = x_spec * np.hanning(N)
    
    freq = np.fft.fftshift(np.fft.fftfreq(N, d=dt_med))
    X = np.fft.fftshift(np.fft.fft(x_spec))
    mag = np.abs(X)
    
    # DC component
    zero_idx = np.argmin(np.abs(freq))
    dc_mag = float(mag[zero_idx]) if zero_idx < len(mag) else 0.0
    
    # Peak detection
    if is_real_signal:
        nonneg = freq >= 0
        peaks_all = top_peaks_improved(freq[nonneg], mag[nonneg], k=top_k_peaks, rel_thr=peak_rel_threshold)
        peaks_pos = [(f, a) for (f, a) in peaks_all if f > 0]
        peaks_neg = []
    else:
        pos = freq > 0
        neg = freq < 0
        peaks_pos = top_peaks_improved(freq[pos], mag[pos], k=top_k_peaks, rel_thr=peak_rel_threshold)
        peaks_neg = top_peaks_improved(freq[neg], mag[neg], k=top_k_peaks, rel_thr=peak_rel_threshold)
    
    # Check if DC is a true peak
    global_max_mag = float(np.max(mag)) if mag.size > 0 else 1.0
    dc_is_peak = False
    if dc_mag >= global_max_mag * peak_rel_threshold:
        if zero_idx > 0 and zero_idx < len(mag) - 1:
            left_val = float(mag[zero_idx - 1])
            right_val = float(mag[zero_idx + 1])
            if dc_mag > left_val and dc_mag > right_val:
                dc_is_peak = True
    
    # Combine all peaks
    all_peaks = list(peaks_pos) + list(peaks_neg)
    if dc_is_peak:
        all_peaks.append((0.0, dc_mag))
    all_peaks.sort(key=lambda z: z[1], reverse=True)
    
    # Classify peak pattern
    peak_pattern = 'none'
    significant_peaks = []
    if all_peaks:
        max_amp = all_peaks[0][1]
        significant_peaks = [(f, a) for (f, a) in all_peaks if (a / max_amp) >= 0.30]
        
        if len(significant_peaks) == 0:
            peak_pattern = 'none'
        elif len(significant_peaks) == 1:
            f_dom = significant_peaks[0][0]
            peak_pattern = 'dc_only' if abs(f_dom) < 1e-10 else 'single'
        elif len(significant_peaks) == 2:
            peak_pattern = 'two'
        else:
            peak_pattern = 'multiple'
    
    strong_peaks_2 = significant_peaks[:2] if len(significant_peaks) >= 2 else significant_peaks
    
    # Envelope analysis
    amp = np.abs(x_fft)
    amp_spec = amp.copy()
    if detrend_mean:
        amp_spec = amp_spec - np.mean(amp_spec)
    if apply_hann_window:
        amp_spec = amp_spec * np.hanning(N)
    A = np.fft.fftshift(np.fft.fft(amp_spec))
    amp_mag = np.abs(A)
    pos_mask = freq > 0
    amp_peaks_pos = top_peaks_improved(freq[pos_mask], amp_mag[pos_mask], k=3, rel_thr=0.10)
    f_env = amp_peaks_pos[0][0] if len(amp_peaks_pos) else np.nan
    
    # Phase analysis
    if is_real_signal:
        phase = np.zeros_like(t)
        phase_slope = 0.0
        phase_intercept = 0.0
        f_phase = np.nan
        phase_lin = np.zeros_like(t)
        phase_available = False
    else:
        phase = np.unwrap(np.angle(x))
        amp_thr = max(1e-14, float(np.max(np.abs(x))) * 1e-3)
        phase_fit_mask = np.abs(x) > amp_thr
        if np.sum(phase_fit_mask) < 8:
            phase_fit_mask = np.ones_like(t, dtype=bool)
        p = np.polyfit(t[phase_fit_mask], phase[phase_fit_mask], deg=1)
        phase_slope = float(p[0])
        phase_intercept = float(p[1])
        f_phase = phase_slope / (2.0 * np.pi)
        phase_lin = phase_slope * t + phase_intercept
        phase_available = True
    
    return SpectralAnalysisResult(
        t=t, x=x, x_complex=x_complex,
        freq=freq, mag=mag,
        N=N, dt_med=dt_med, Tspan=Tspan, fft_note=fft_note,
        peaks_pos=peaks_pos, peaks_neg=peaks_neg,
        dc_mag=dc_mag, dc_is_peak=dc_is_peak,
        peak_pattern=peak_pattern,
        significant_peaks=significant_peaks,
        strong_peaks_2=strong_peaks_2,
        f_env=f_env,
        phase=phase, phase_slope=phase_slope, phase_intercept=phase_intercept,
        f_phase=f_phase, phase_lin=phase_lin, phase_available=phase_available,
        key=key, mode_label=mode_label, is_real_signal=is_real_signal,
        source_tag='', norm_tag=norm_tag,
    )


def plot_4panel_eda(result: SpectralAnalysisResult, figsize: Tuple[float, float] = (13, 9)) -> plt.Figure:
    """
    Create 4-panel EDA visualization for spectral analysis results.
    
    Parameters
    ----------
    result : SpectralAnalysisResult
        Output from analyze_cnlm_spectrum()
    figsize : tuple
        Figure size
    
    Returns
    -------
    matplotlib.Figure
    """
    t = result.t
    x = result.x
    freq = result.freq
    mag = result.mag
    
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.subplots_adjust(hspace=0.28, wspace=0.30)
    
    # (1) Time domain
    ax = axs[0, 0]
    if result.is_real_signal:
        ax.plot(t, x, color='C0', linewidth=2.0, label=rf'{result.mode_label}')
        y_max_1 = float(np.max(np.abs(x))) if x.size else 1.0
    else:
        ax.plot(t, np.abs(x), color='black', linewidth=2.6, label=r'$|c(t)|$')
        ax.plot(t, x.real, color='C0', alpha=0.55, linewidth=1.2, label=r'Re$[c(t)]$')
        ax.plot(t, x.imag, color='C1', alpha=0.55, linewidth=1.2, label=r'Im$[c(t)]$')
        y_max_1 = float(np.max(np.abs(x))) if x.size else 1.0
    
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'1) Time-domain decomposition [{result.mode_label}]')
    ax.grid(True, alpha=0.25)
    
    if np.isfinite(y_max_1) and y_max_1 > 0:
        ax.set_ylim(bottom=-1.15 * y_max_1, top=+1.65 * y_max_1)
    
    if np.isfinite(result.f_env):
        ax.plot([], [], ' ', label=rf'$f_{{\mathrm{{env}}}}\approx {result.f_env:.6g}$')
    ax.legend(loc='upper left', fontsize=9, frameon=False, ncol=2)
    
    # (2) Spectrum
    ax = axs[0, 1]
    mag_plot = np.clip(mag.copy(), 1e-30, None)
    ax.plot(freq, mag_plot, color='blue', linewidth=1.4)
    ax.set_yscale('log')
    
    min_span = 20.0 / max(result.Tspan, 1e-30)
    
    if result.peak_pattern == 'two' and len(result.strong_peaks_2) == 2:
        f1, f2 = sorted([result.strong_peaks_2[0][0], result.strong_peaks_2[1][0]])
        df_peaks = float(abs(f2 - f1))
        span = max(5.0 * df_peaks, min_span)
        f_center = 0.5 * (f1 + f2)
        ax.set_xlim(f_center - 0.5 * span, f_center + 0.5 * span)
        ax.axvline(f1, color='red', linestyle='--', linewidth=1.2, alpha=0.9)
        ax.axvline(f2, color='red', linestyle='--', linewidth=1.2, alpha=0.9)
        ax.annotate('', xy=(f1, 0.90), xytext=(f2, 0.90),
                   xycoords=('data', 'axes fraction'),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=1.2))
        ax.text(0.5*(f1+f2), 0.93, rf'$\Delta f\approx {df_peaks:.6g}$',
               color='red', ha='center', va='bottom',
               transform=ax.get_xaxis_transform(), fontsize=9)
    elif result.peak_pattern == 'single' and len(result.significant_peaks) >= 1:
        f1 = float(result.significant_peaks[0][0])
        span = max(4.0 * max(abs(f1), 1e-30), min_span)
        ax.set_xlim(f1 - 0.5 * span, f1 + 0.5 * span)
        ax.axvline(f1, color='red', linestyle='--', linewidth=1.2, alpha=0.9)
    elif result.peak_pattern == 'dc_only':
        ax.set_xlim(-0.5 * min_span, 0.5 * min_span)
        ax.axvline(0.0, color='red', linestyle='--', linewidth=1.2, alpha=0.9)
    elif result.peak_pattern == 'multiple' and len(result.significant_peaks) > 2:
        freqs_sig = [f for (f, a) in result.significant_peaks]
        f_min, f_max = min(freqs_sig), max(freqs_sig)
        f_center = 0.5 * (f_min + f_max)
        span = max(3.0 * (f_max - f_min), min_span)
        ax.set_xlim(f_center - 0.5 * span, f_center + 0.5 * span)
        for (f_pk, a_pk) in result.significant_peaks:
            ax.axvline(f_pk, color='red', linestyle='--', linewidth=1.0, alpha=0.8)
    elif len(result.peaks_pos) >= 1:
        f1 = float(result.peaks_pos[0][0])
        span = max(4.0 * max(abs(f1), 1e-30), min_span)
        ax.set_xlim(f1 - 0.5 * span, f1 + 0.5 * span)
        ax.axvline(f1, color='red', linestyle='--', linewidth=1.2, alpha=0.9)
    
    # Auto ylim based on visible window
    xlo, xhi = ax.get_xlim()
    win_mask = (freq >= xlo) & (freq <= xhi)
    mag_win = mag_plot[win_mask]
    mag_win = mag_win[np.isfinite(mag_win)]
    if mag_win.size >= 8:
        y_win_max = float(np.max(mag_win))
        y_bg = float(np.percentile(mag_win, 10))
        y_bg = max(y_bg, 1e-30)
        if result.peak_pattern == 'two' and len(result.strong_peaks_2) >= 2:
            f1, f2 = sorted([result.strong_peaks_2[0][0], result.strong_peaks_2[1][0]])
            i1 = int(np.argmin(np.abs(freq - f1)))
            i2 = int(np.argmin(np.abs(freq - f2)))
            ilo, ihi = (i1, i2) if i1 <= i2 else (i2, i1)
            if (ihi - ilo) >= 2:
                y_valley = float(np.min(mag_plot[ilo:ihi + 1]))
                y_ref_low = max(y_valley, y_bg)
            else:
                y_ref_low = y_bg
            y_top = y_win_max * 1.40
            y_bottom = max(y_ref_low * 0.85, y_win_max / 60.0, 1e-30)
        else:
            y_top = y_win_max * 1.40
            y_bottom = max(y_bg * 0.85, y_win_max / 80.0, 1e-30)
        if np.isfinite(y_bottom) and np.isfinite(y_top) and (y_top > y_bottom):
            ax.set_ylim(y_bottom, y_top)
    
    ax.set_xlabel(r'Frequency $f$ (cycles per time unit)')
    ax.set_ylabel(r'$|\mathrm{FFT}|$')
    ax.set_title(f'2) Spectrum [{result.mode_label}]')
    ax.grid(True, which='both', alpha=0.25)
    
    # (3) Argand / trajectory
    ax = axs[1, 0]
    if result.is_real_signal:
        ax.plot(t, x, color='teal', alpha=0.6, linewidth=1.2)
        sc = ax.scatter(t, x, c=t, s=8, cmap='viridis', alpha=0.7)
        ax.scatter([t[0]], [x[0]], color='green', s=60, marker='o', label='start')
        ax.scatter([t[-1]], [x[-1]], color='red', s=60, marker='x', label='end')
        ax.set_xlabel('Time $t$')
        ax.set_ylabel(f'{result.mode_label}')
        ax.set_title(f'3) Signal trajectory [{result.mode_label}]')
    else:
        ax.plot(x.real, x.imag, color='teal', alpha=0.6, linewidth=1.2)
        sc = ax.scatter(x.real, x.imag, c=t, s=8, cmap='viridis', alpha=0.7)
        ax.scatter([x.real[0]], [x.imag[0]], color='green', s=60, marker='o', label='start')
        ax.scatter([x.real[-1]], [x.imag[-1]], color='red', s=60, marker='x', label='end')
        ax.set_xlabel(r'Re$[c(t)]$')
        ax.set_ylabel(r'Im$[c(t)]$')
        ax.set_title('3) Argand trajectory (complex plane)')
        ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.25)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Time $t$')
    ax.legend(loc='best', fontsize=9)
    
    # (4) Phase dynamics
    ax = axs[1, 1]
    if result.phase_available:
        ax.plot(t, result.phase, color='darkred', linewidth=1.6, label='Unwrapped phase')
        ax.plot(t, result.phase_lin, color='black', linewidth=1.1, alpha=0.7, linestyle='--', label='Linear fit')
        ax.set_xlabel('Time $t$')
        ax.set_ylabel('Phase (rad)')
        ax.set_title('4) Phase dynamics (unwrap)')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='best', fontsize=9)
        phase_annot = (f"Slope ≈ {result.phase_slope:.6g} rad/unit\n"
                      + rf"$\Rightarrow\; f_{{\mathrm{{phase}}}}\approx {result.f_phase:.6g}$")
        ax.text(0.02, 0.95, phase_annot, transform=ax.transAxes, va='top', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Phase analysis not available\nfor real-valued signals',
               transform=ax.transAxes, ha='center', va='center', fontsize=12, color='gray')
        ax.set_title('4) Phase dynamics (N/A for real signal)')
        ax.grid(True, alpha=0.25)
    
    fig.suptitle(f"Signal EDA for key={result.key} [{result.mode_label}] | "
                f"N={result.N}, dt≈{result.dt_med:.6g}, T≈{result.Tspan:.6g}{result.norm_tag}", fontsize=13)
    
    return fig


def print_spectral_summary(result: SpectralAnalysisResult, top_k: int = 5) -> None:
    """
    Print formatted spectral analysis summary.
    
    Parameters
    ----------
    result : SpectralAnalysisResult
        Output from analyze_cnlm_spectrum()
    top_k : int
        Number of peaks to print
    """
    print("\n===== 4-panel quick readout (math-only) =====")
    print(f"Analysis mode: {result.mode_label}")
    print(result.fft_note)
    
    print("\n[1] Time-domain decomposition:")
    if result.is_real_signal:
        print(f"  - Plotted {result.mode_label} directly.")
    else:
        print("  - Black line |c(t)| is the envelope; blue/orange are Re/Im.")
    
    print("\n[2] Frequency domain:")
    global_max = float(np.max(result.mag)) if result.mag.size > 0 else 1.0
    
    print(f"\n  Spectral peaks (threshold = 5% of max):")
    print(f"\n    Positive frequency peaks ({len(result.peaks_pos)} detected):")
    if result.peaks_pos:
        for i, (f, a) in enumerate(result.peaks_pos[:top_k], 1):
            rel = a / global_max if global_max > 0 else 0
            print(f"      {i}. f = {f: .6g},  |FFT| = {a: .6g},  rel_amp = {rel: .4f}")
    else:
        print("      (none)")
    
    if not result.is_real_signal:
        print(f"\n    Negative frequency peaks ({len(result.peaks_neg)} detected):")
        if result.peaks_neg:
            for i, (f, a) in enumerate(result.peaks_neg[:top_k], 1):
                rel = a / global_max if global_max > 0 else 0
                print(f"      {i}. f = {f: .6g},  |FFT| = {a: .6g},  rel_amp = {rel: .4f}")
        else:
            print("      (none)")
    
    if result.dc_is_peak:
        rel_dc = result.dc_mag / global_max if global_max > 0 else 0
        print(f"\n    DC peak (f=0): |FFT| = {result.dc_mag: .6g}, rel_amp = {rel_dc: .4f}")
    
    print(f"\n  Peak pattern detected: {result.peak_pattern}")
    
    if result.significant_peaks:
        print(f"  Significant peaks (rel_amp >= 0.30):")
        for i, (f, a) in enumerate(result.significant_peaks[:top_k], 1):
            rel = a / global_max if global_max > 0 else 0
            print(f"    {i}. f = {f: .6g},  |FFT| = {a: .6g},  rel_amp = {rel: .4f}")
    
    print("\n[3] Trajectory plot:")
    if result.is_real_signal:
        print("  - Shows signal value vs time.")
    else:
        print("  - Near-circle: single frequency; spiral/petal: multi-frequency.")
    
    print("\n[4] Phase dynamics:")
    if result.phase_available:
        print(f"  - Phase-derived frequency f_phase ≈ {result.f_phase:.6g}")
    else:
        print("  - Not available for real signals.")
    
    # Summary
    print("\n===== Spectral Analysis Summary =====")
    if result.peak_pattern == 'dc_only':
        print("CONCLUSION: DC-dominated (approximately constant)")
    elif result.peak_pattern == 'single':
        f_dom = result.significant_peaks[0][0]
        print(f"CONCLUSION: Single-frequency dominated, f ≈ {f_dom:.6g}")
    elif result.peak_pattern == 'two':
        f1, f2 = sorted([result.significant_peaks[0][0], result.significant_peaks[1][0]])
        print(f"CONCLUSION: Two-frequency beating, f1 ≈ {f1:.6g}, f2 ≈ {f2:.6g}")
        print(f"  - Δf ≈ {abs(f2-f1):.6g}")
    elif result.peak_pattern == 'multiple':
        print(f"CONCLUSION: Multi-frequency superposition ({len(result.significant_peaks)} peaks)")
    else:
        print("CONCLUSION: No clear peak structure")
    
    if np.isfinite(result.f_env):
        print(f"\nEnvelope frequency f_env ≈ {result.f_env:.6g}")


def export_prd_figure(
    result: SpectralAnalysisResult,
    save_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (7.6, 3.1),
    time_xlim: Optional[Tuple[float, float]] = None,
    save_pdf: bool = True,
) -> Path:
    """
    Export PRD-ready 1×2 figure (time domain + spectrum).
    
    Parameters
    ----------
    result : SpectralAnalysisResult
        Output from analyze_cnlm_spectrum()
    save_path : Path, optional
        Output base path. If None, auto-generated.
        The function always writes EPS to <stem>.eps and, if save_pdf=True, PDF to <stem>.pdf.
    figsize : tuple
        Figure size
    time_xlim : tuple(float, float), optional
        X-range for the left time-domain panel. If provided, left-panel y-limits
        are auto-scaled using only data inside this range.
    save_pdf : bool
        Whether to additionally export a PDF next to EPS.
    
    Returns
    -------
    Path to saved EPS file
    """
    n, ell, m = result.key
    
    if save_path is None:
        mode_suffix = ''
        if result.mode_label == 'Re[c]':
            mode_suffix = '_Re'
        elif result.mode_label == 'Im[c]':
            mode_suffix = '_Im'
        save_path = Path(f"PRD_cnlm_time_spectrum_n{n}_ell{ell}_m{m}{mode_suffix}.eps")
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    base_path = save_path.with_suffix('') if save_path.suffix else save_path
    eps_path = base_path.with_suffix('.eps')
    pdf_path = base_path.with_suffix('.pdf')
    
    def _sci_tex(x: float, digits: int = 2) -> str:
        if not np.isfinite(x):
            return r"\mathrm{NaN}"
        if x == 0:
            return "0"
        s = f"{x:.{digits}e}"
        mant, exp = s.split('e')
        exp_i = int(exp)
        mant = mant.rstrip('0').rstrip('.')
        return rf"{mant}\times10^{{{exp_i}}}"
    
    c_nlm_tex = rf"c_{{{n},{ell},{m}}}"
    c_freq_tex = rf"\tilde{{c}}_{{{n},{ell},{m}}}(f)"
    
    t = result.t
    x = result.x
    freq = result.freq
    mag = result.mag
    
    with plt.rc_context({'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 8.8}):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.93, wspace=0.36)
        
        # Time domain
        if result.is_real_signal:
            ax1.plot(t, x, color='C0', linewidth=2.0, label=rf"${result.mode_label}$")
        else:
            ax1.plot(t, np.abs(x), color='black', linewidth=2.3, label=rf"$|{c_nlm_tex}|$")
            ax1.plot(t, x.real, color='C0', linewidth=1.15, linestyle='--', label=rf"$\mathrm{{Re}}\,[{c_nlm_tex}]$")
            ax1.plot(t, x.imag, color='C1', linewidth=1.15, linestyle=':', label=rf"$\mathrm{{Im}}\,[{c_nlm_tex}]$")
        
        ax1.set_xlabel(r"Time $t$ (Myr)")
        ax1.set_ylabel("Amplitude")

        # Optional envelope-frequency text in legend
        if np.isfinite(result.f_env):
            f_env_tex = _sci_tex(float(result.f_env), digits=2)
            ax1.plot([], [], ' ', label=rf"$f_{{\mathrm{{env}}}}\approx {f_env_tex}\,\mathrm{{Myr}}^{{-1}}$")

        ax1.legend(
            loc='upper left',
            bbox_to_anchor=(0.01, 0.995),
            frameon=False,
            ncol=2,
            columnspacing=0.9,
            handletextpad=0.55,
            handlelength=1.6,
            borderaxespad=0.0,
            labelspacing=0.25,
        )

        def _set_left_ylim(y_abs_max: float) -> None:
            # Reserve generous headroom for 4-item legend and keep curves away from text.
            if (not np.isfinite(y_abs_max)) or (y_abs_max <= 0):
                y_abs_max = 1.0
            if result.is_real_signal:
                ax1.set_ylim(bottom=-1.25 * y_abs_max, top=+1.55 * y_abs_max)
            else:
                ax1.set_ylim(bottom=-1.25 * y_abs_max, top=+2.45 * y_abs_max)

        # Optional left-panel time window (for publication crops)
        if time_xlim is not None:
            t0, t1 = float(time_xlim[0]), float(time_xlim[1])
            if t1 <= t0:
                raise ValueError(f"Invalid time_xlim={time_xlim}: require time_xlim[1] > time_xlim[0]")
            ax1.set_xlim(t0, t1)

            if result.is_real_signal:
                y_ref = np.asarray(x, dtype=float)
            else:
                y_ref = np.concatenate([
                    np.asarray(np.abs(x), dtype=float),
                    np.asarray(x.real, dtype=float),
                    np.asarray(x.imag, dtype=float),
                ])

            m = (t >= t0) & (t <= t1)
            if np.any(m):
                if result.is_real_signal:
                    y_win = np.asarray(x[m], dtype=float)
                else:
                    y_win = np.concatenate([
                        np.asarray(np.abs(x[m]), dtype=float),
                        np.asarray(x.real[m], dtype=float),
                        np.asarray(x.imag[m], dtype=float),
                    ])
                y_abs_max = float(np.max(np.abs(y_win))) if y_win.size else 0.0
            else:
                y_abs_max = 0.0

            if (not np.isfinite(y_abs_max)) or (y_abs_max <= 0):
                y_abs_max = float(np.max(np.abs(y_ref))) if y_ref.size else 1.0
            _set_left_ylim(y_abs_max)
        
        y_max = float(np.max(np.abs(x))) if np.size(x) else 1.0
        if (time_xlim is None) and np.isfinite(y_max) and y_max > 0:
            _set_left_ylim(y_max)
        
        # Spectrum
        mag_plot = np.clip(mag.copy(), 1e-30, None)
        ax2.plot(freq, mag_plot, color='blue', linewidth=1.25)
        ax2.set_yscale('log')
        
        min_span = 20.0 / max(result.Tspan, 1e-30)
        
        if result.peak_pattern == 'two' and len(result.significant_peaks) >= 2:
            f1, f2 = sorted([result.significant_peaks[0][0], result.significant_peaks[1][0]])
            df = abs(f2 - f1)
            span = max(5.0 * df, min_span)
            ax2.set_xlim(0.5*(f1+f2) - 0.5*span, 0.5*(f1+f2) + 0.5*span)
            ax2.axvline(f1, color='red', linestyle='--', linewidth=1.2)
            ax2.axvline(f2, color='red', linestyle='--', linewidth=1.2)
            df_tex = _sci_tex(df, digits=2)
            y_frac = 0.70
            ax2.annotate('', xy=(f1, y_frac), xytext=(f2, y_frac),
                        xycoords=('data', 'axes fraction'),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.25))
            xlo_now, xhi_now = ax2.get_xlim()
            span_now = max(float(xhi_now - xlo_now), 1e-30)
            x_margin = 0.12 * span_now
            x_text = 0.5 * (f1 + f2)
            x_text = min(max(x_text, float(xlo_now) + x_margin), float(xhi_now) - x_margin)
            ax2.text(x_text, y_frac + 0.06, rf"$\Delta f\approx {df_tex}\,\mathrm{{Myr}}^{{-1}}$",
                    color='red', ha='center', va='bottom',
                transform=ax2.get_xaxis_transform(), fontsize=9.2,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
        elif result.peak_pattern == 'single' and len(result.significant_peaks) >= 1:
            f1 = float(result.significant_peaks[0][0])
            span = max(4.0 * max(abs(f1), 1e-30), min_span)
            ax2.set_xlim(f1 - 0.5*span, f1 + 0.5*span)
            ax2.axvline(f1, color='red', linestyle='--', linewidth=1.2)
        
        # Auto ylim based on visible window
        xlo, xhi = ax2.get_xlim()
        win_mask = (freq >= xlo) & (freq <= xhi)
        mag_win = mag_plot[win_mask]
        mag_win = mag_win[np.isfinite(mag_win)]
        if mag_win.size >= 8:
            y_peak = float(np.max(mag_win))
            y_bg = float(np.percentile(mag_win, 10))
            y_bg = max(y_bg, 1e-30)
            if result.peak_pattern == 'two' and len(result.significant_peaks) >= 2:
                f1, f2 = sorted([result.significant_peaks[0][0], result.significant_peaks[1][0]])
                i1 = int(np.argmin(np.abs(freq - f1)))
                i2 = int(np.argmin(np.abs(freq - f2)))
                ilo, ihi = (i1, i2) if i1 <= i2 else (i2, i1)
                if (ihi - ilo) >= 2:
                    y_valley = float(np.min(mag_plot[ilo:ihi + 1]))
                    y_ref_low = max(y_valley, y_bg)
                else:
                    y_ref_low = y_bg
            else:
                y_ref_low = y_bg

            # Log-scale vertical framing: keep selected peaks around the middle-upper region
            # and avoid sticking to top border.
            y_bottom = max(y_ref_low * 0.90, y_peak / 220.0, 1e-30)
            if y_peak > y_bottom:
                peak_frac = 0.62  # desired vertical position of peak in axes fraction (log scale)
                y_top_candidate = y_bottom * np.exp((np.log(y_peak) - np.log(y_bottom)) / peak_frac)
                y_top = min(max(y_top_candidate, y_peak * 1.30), y_peak * 35.0)
            else:
                y_top = y_peak * 2.0
            if np.isfinite(y_bottom) and np.isfinite(y_top) and (y_top > y_bottom):
                ax2.set_ylim(y_bottom, y_top)
        
        ax2.set_xlabel(r"Frequency $f$ (Myr$^{-1}$)")
        ax2.set_ylabel(rf"$|{c_freq_tex}|$")
        
        fig.savefig(eps_path, format='eps', bbox_inches='tight')
        if save_pdf:
            fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.show()
    
    print(f"Saved EPS: {eps_path}")
    if save_pdf:
        print(f"Saved PDF: {pdf_path}")
    return eps_path


# ============================================================================
# Additional plotting utilities for c_{nlm}(t) visualization
# ============================================================================

def plot_cnlm_comparison(
    t_original: np.ndarray,
    c_original: np.ndarray,
    t_dense: np.ndarray,
    c_dense: np.ndarray,
    key: Tuple[int, int, int],
    normalization: float = 1.0,
    figsize: Tuple[float, float] = (8, 4),
) -> plt.Figure:
    """
    Plot comparison between original and interpolated c_{nlm}(t) time series.
    
    Parameters
    ----------
    t_original : np.ndarray
        Original time array
    c_original : np.ndarray
        Original complex coefficient array
    t_dense : np.ndarray
        Dense (interpolated) time array
    c_dense : np.ndarray
        Dense (interpolated) complex coefficient array
    key : tuple
        (n, ℓ, m) indices
    normalization : float
        Normalization factor (divide |c|^2 by this)
    figsize : tuple
        Figure size
    
    Returns
    -------
    matplotlib.Figure
    """
    n, ell, m = key
    
    abs_c2_orig = np.abs(c_original)**2 / normalization
    abs_c2_dense = np.abs(c_dense)**2 / normalization
    
    fig = plt.figure(figsize=figsize)
    plt.plot(t_original, abs_c2_orig, label='Original sampling', linewidth=3)
    plt.plot(t_dense, abs_c2_dense, label='Interpolated sampling', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel(r'$|c_{nlm}(t)|^2$')
    plt.title(rf'$|c_{{{n}{ell}{m}}}(t)|^2$: original vs interpolated')
    plt.legend()
    plt.grid(True)
    
    return fig


def plot_cnlm_prd_single(
    t: np.ndarray,
    c: np.ndarray,
    key: Tuple[int, int, int],
    normalization: float = 1.0,
    save_path: Optional[Path] = None,
    xlim: Tuple[float, float] = (0.0, 3000.0),
    figsize: Tuple[float, float] = (3.5, 2.6),
    color: str = 'red',
) -> Path:
    """
    Create PRD-ready single-mode |c_{nlm}(t)|^2 plot.
    
    Parameters
    ----------
    t : np.ndarray
        Time array
    c : np.ndarray
        Complex coefficient array
    key : tuple
        (n, ℓ, m) indices
    normalization : float
        Normalization factor (divide |c|^2 by this)
    save_path : Path, optional
        Output path. If None, auto-generated.
    xlim : tuple
        Time axis limits
    figsize : tuple
        Figure size
    color : str
        Line color
    
    Returns
    -------
    Path to saved file
    """
    import matplotlib as mpl
    
    n, ell, m = key
    y = np.abs(c)**2 / normalization
    
    # Auto-generate save path
    if save_path is None:
        if m < 0:
            m_tag = f"mneg{abs(m)}"
        elif m > 0:
            m_tag = f"mpos{m}"
        else:
            m_tag = "m0"
        save_path = Path(f"cnlm_n{n}_l{ell}_{m_tag}_original.eps")
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Publication styling
    style = {
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.linewidth': 1.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
        'xtick.major.size': 4.0,
        'ytick.major.size': 4.0,
        'xtick.minor.size': 2.0,
        'ytick.minor.size': 2.0,
        'ps.useafm': True,
        'pdf.use14corefonts': True,
        'text.usetex': False,
    }
    
    y_label = rf'$|c_{{{n},{ell},{m}}}(t)|^2$'
    
    with mpl.rc_context(style):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(t, y, color=color, linewidth=2.0)
        
        ax.set_xlabel(r'Time $t$ (Myr)')
        ax.set_ylabel(y_label)
        ax.set_xlim(xlim)
        
        ax.grid(False)
        ax.minorticks_on()
        ax.tick_params(top=True, right=True, which='both')
        
        fig.tight_layout(pad=0.2)
        fig.savefig(save_path, format='eps', bbox_inches='tight', pad_inches=0.02, facecolor='white')
        plt.show()
    
    print(f"Saved EPS to: {save_path}")
    return save_path


def plot_cnlm_multimode(
    t: np.ndarray,
    c_dict: Dict[Tuple[int, int, int], np.ndarray],
    n_select: int,
    l_min: int,
    l_max: int,
    normalization: float = 1.0,
    figsize: Tuple[float, float] = (9.5, 5.5),
    log_scale: bool = True,
) -> plt.Figure:
    """
    Plot |c_{nlm}(t)|^2 for multiple modes with color/linestyle encoding.
    
    Colors encode ℓ (rainbow gradient), linestyles encode |m|, 
    line widths encode sign of m.
    
    Parameters
    ----------
    t : np.ndarray
        Time array
    c_dict : dict
        Dictionary mapping (n, ℓ, m) -> complex coefficient array
    n_select : int
        Radial index n to plot
    l_min : int
        Minimum ℓ to include
    l_max : int
        Maximum ℓ to include
    normalization : float
        Normalization factor (divide |c|^2 by this)
    figsize : tuple
        Figure size
    log_scale : bool
        Use log scale for y-axis
    
    Returns
    -------
    matplotlib.Figure
    """
    import matplotlib as mpl
    
    if l_min < 0:
        raise ValueError("l_min must be >= 0")
    if l_min > l_max:
        raise ValueError("l_min must be <= l_max")
    
    # Color map for ℓ
    cmap = plt.get_cmap('rainbow')
    
    def color_for_ell(ell: int) -> tuple:
        if l_max == l_min:
            v = 0.0
        else:
            v = (ell - l_min) / (l_max - l_min)
            v = min(max(v, 0.0), 1.0)
        v = 1.0 - v  # invert: l_min -> red, l_max -> violet
        return cmap(v)
    
    color_map = {ell: color_for_ell(ell) for ell in range(l_min, l_max + 1)}
    
    # Linestyle patterns for |m|
    patterns_by_abs_m = {
        1: (8, 4),
        2: (5, 2, 1, 2),
        3: (3, 1, 1, 1),
        4: (1, 1),
        5: (9, 3, 2, 3),
    }
    fallback_patterns = [
        (6, 3), (4, 2, 1, 2), (2, 2), (10, 2, 2, 2), (3, 2, 3, 2, 1, 2)
    ]
    
    def dashseq_for_abs_m(k: int):
        if k in patterns_by_abs_m:
            return patterns_by_abs_m[k]
        return fallback_patterns[(k - 1) % len(fallback_patterns)]
    
    # Line width policy
    LW_ZERO = 3.5
    LW_NEG = 2.5
    LW_POS = 1.5
    
    def linestyle_for_m(m: int):
        if m == 0:
            return '-'
        return (0, dashseq_for_abs_m(abs(m)))
    
    def linewidth_for_m(m: int):
        if m == 0:
            return LW_ZERO
        return LW_POS if m > 0 else LW_NEG
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(right=0.76)
    
    min_pos = np.inf
    max_val = 0.0
    missing = []
    
    for ell in range(l_min, l_max + 1):
        for m in range(-ell, ell + 1):
            key = (n_select, ell, m)
            if key not in c_dict:
                missing.append(key)
                continue
            c_series = c_dict[key]
            y_vals = np.abs(c_series)**2 / normalization
            y_safe = np.clip(y_vals, 1e-18, None)
            ax.plot(
                t,
                y_safe,
                color=color_map[ell],
                linestyle=linestyle_for_m(m),
                linewidth=linewidth_for_m(m),
            )
            if np.any(y_vals > 0):
                min_pos = min(min_pos, float(np.min(y_vals[y_vals > 0])))
                max_val = max(max_val, float(np.max(y_vals)))
    
    # Y axis scaling
    if log_scale:
        ax.set_yscale('log')
        y_bottom = 1e-12 if not np.isfinite(min_pos) else max(1e-25, min_pos * 0.8)
        y_top_candidate = (max_val * 10) if max_val > 0 else 1.1
        y_top = max(1.05, y_top_candidate) if l_min == 0 else y_top_candidate
        if y_top <= y_bottom:
            y_top = y_bottom * 10
        ax.set_ylim(y_bottom, y_top)
    
    ax.set_xlabel('Time $t$', fontsize=12)
    ax.set_ylabel(r'$|c_{nlm}(t)|^2$', fontsize=12)
    ax.set_title(
        rf'Time-resolved coefficients $|c_{{nlm}}(t)|^2$ for $n={n_select}$ and ${l_min} \leq \ell \leq {l_max}$',
        fontsize=13,
    )
    
    ax.grid(True, which='major', alpha=0.35)
    ax.grid(True, which='minor', linestyle=':', alpha=0.15)
    ax.minorticks_on()
    
    # Legend
    ell_handles = [
        mpl.lines.Line2D([0], [0], color=color_map[ell], lw=2.2, linestyle='-')
        for ell in range(l_min, l_max + 1)
    ]
    ell_labels = [rf'$\ell={ell}$' for ell in range(l_min, l_max + 1)]
    
    m_handles = [
        mpl.lines.Line2D([0], [0], color='black', linestyle=linestyle_for_m(0), lw=LW_ZERO)
    ]
    m_labels = [r'$m=0$']
    for k in range(1, l_max + 1):
        m_handles.append(mpl.lines.Line2D([0], [0], color='black', linestyle=linestyle_for_m(+k), lw=LW_POS))
        m_labels.append(rf'$m=+{k}$')
        m_handles.append(mpl.lines.Line2D([0], [0], color='black', linestyle=linestyle_for_m(-k), lw=LW_NEG))
        m_labels.append(rf'$m=-{k}$')
    
    handles = ell_handles + m_handles
    labels = ell_labels + m_labels
    
    ax.legend(
        handles,
        labels,
        title=r'Degree $\ell$ (color)' + '\n' + r'Order $m$ (Style/Width)' + '\n' + r'Width: $m=0>m_+>m_-$',
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
        title_fontsize=10,
        handlelength=1.6,
        handletextpad=0.6,
        labelspacing=0.35,
        columnspacing=0.8,
        ncol=1,
    )
    
    plt.tight_layout()
    
    if missing:
        print(f"Skipped {len(missing)} missing coefficient series, e.g., first missing: {missing[0]}")
    
    return fig
