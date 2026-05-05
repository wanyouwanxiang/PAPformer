"""
Sliding-Window Variational Mode Decomposition (SMVMD)

Decomposes a long time series into Intrinsic Mode Functions (IMFs) via
overlapping sliding windows + VMD, with linear cross-fade blending in
overlap regions and spectral-centroid sorting for mode alignment.

Pipeline:
    1. Standardize the signal
    2. Slice into overlapping windows
    3. VMD decomposition per window + spectral-centroid sorting
    4. Linear cross-fade merge across windows
    5. Reconstruct signal from merged IMFs
    6. Verify reconstruction accuracy
"""

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from vmdpy import VMD

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ==============================================================================
# Pipeline Steps
# ==============================================================================


def standardize(data):
    """Zero-mean, unit-variance standardization."""
    data = np.array(data).reshape(-1, 1)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(data).flatten()
    return standardized, scaler


def sliding_window(data, window_len, overlap_ratio):
    """Slice data into overlapping windows with tail-alignment."""
    data_length = len(data)
    step = int(window_len * (1 - overlap_ratio))

    windows, starts = [], []
    start = 0
    while start + window_len <= data_length:
        windows.append(data[start : start + window_len])
        starts.append(start)
        start += step

    # Tail-align: include the last full window if there is a residual
    if start < data_length and len(windows) > 0:
        windows.append(data[-window_len:])
        starts.append(data_length - window_len)

    print(f"  Data length: {data_length}, Window: {window_len}, Step: {step}")
    print(f"  Windows generated: {len(windows)}")
    return np.array(windows), np.array(starts)


def vmd_decompose(windows, K, alpha, tau=0, DC=1, init=1, tol=1e-7):
    """Run VMD on each window and sort IMFs by descending spectral centroid."""
    all_imfs, all_centroids = [], []

    for idx, window in enumerate(windows):
        print(f"  Processing window {idx + 1}/{len(windows)}")
        u, _, omega = VMD(window, alpha, tau, K, DC, init, tol)

        centroids = omega[-1, :]
        sorted_idx = np.argsort(centroids)[::-1]
        all_imfs.append(u[sorted_idx, :])
        all_centroids.append(centroids[sorted_idx])

    return all_imfs, all_centroids


def merge_windows(all_imfs, starts, window_len, overlap_ratio, total_length):
    """Merge overlapping IMF windows via linear cross-fade blending."""
    K = all_imfs[0].shape[0]
    overlap_len = int(window_len * overlap_ratio)

    merged = np.zeros((K, total_length))
    weights = np.zeros(total_length)

    for idx, (imf, s) in enumerate(zip(all_imfs, starts)):
        e = s + window_len
        w = np.ones(window_len)

        if idx > 0:
            w[:overlap_len] = np.linspace(0, 1, overlap_len)
        if idx < len(all_imfs) - 1:
            w[-overlap_len:] = np.linspace(1, 0, overlap_len)

        for k in range(K):
            merged[k, s:e] += imf[k, :] * w
        weights[s:e] += w

    weights[weights == 0] = 1
    for k in range(K):
        merged[k, :] /= weights

    return merged


def reconstruct(merged_imfs):
    """Sum all IMFs to reconstruct the signal."""
    return np.sum(merged_imfs, axis=0)


def verify(original, reconstructed):
    """Compute and print reconstruction accuracy metrics."""
    abs_err = np.abs(original - reconstructed)
    rmse = np.sqrt(np.mean(abs_err**2))
    signal_rms = np.sqrt(np.mean(original**2))
    rel_rmse = (rmse / signal_rms * 100) if signal_rms > 0 else 0.0
    is_reliable = rel_rmse < 1.0

    print("\n" + "=" * 50)
    print("Reconstruction Verification")
    print("=" * 50)
    print(f"  Absolute RMSE:   {rmse:.6f}")
    print(f"  Signal RMS:      {signal_rms:.6f}")
    print(f"  Relative RMSE:   {rel_rmse:.6f}%")
    print(f"  Max Abs Error:   {np.max(abs_err):.6e}")
    print(f"  Mean Abs Error:  {np.mean(abs_err):.6e}")
    print(f"  Reliable (<1%):  {'Yes' if is_reliable else 'No'}")
    print("=" * 50)

    return rel_rmse, is_reliable


# ==============================================================================
# Main Pipeline
# ==============================================================================


def sliding_window_vmd(data, window_len, overlap_ratio, K, alpha):
    """Run the full SMVMD pipeline and return all results."""
    print("\nSMVMD Pipeline")
    print("=" * 50)

    print("\n[1/6] Standardization")
    std_data, scaler = standardize(data)

    print("\n[2/6] Sliding window slicing")
    windows, starts = sliding_window(data, window_len, overlap_ratio)

    print("\n[3/6] VMD decomposition + spectral sorting")
    all_imfs, all_centroids = vmd_decompose(windows, K, alpha)

    print("\n[4/6] Cross-fade merging")
    merged_imfs = merge_windows(all_imfs, starts, window_len, overlap_ratio, len(data))

    print("\n[5/6] Signal reconstruction")
    rec_signal = reconstruct(merged_imfs)

    print("\n[6/6] Reconstruction verification")
    rel_err, is_reliable = verify(data, rec_signal)

    return {
        "original_data": data,
        "standardized_data": std_data,
        "windows": windows,
        "window_starts": starts,
        "all_imfs": all_imfs,
        "all_spectral_centers": all_centroids,
        "merged_imfs": merged_imfs,
        "reconstructed_signal": rec_signal,
        "relative_error": rel_err,
        "is_reliable": is_reliable,
        "scaler": scaler,
    }


# ==============================================================================
# Visualization & Export
# ==============================================================================


def plot_vmd_results(original, reconstructed, merged_imfs, max_imfs=8):
    """Plot original signal, each IMF, and reconstructed signal."""
    K = merged_imfs.shape[0]
    n_plots = min(K, max_imfs) + 2

    plt.figure(figsize=(12, 2 * n_plots))

    plt.subplot(n_plots, 1, 1)
    plt.plot(original, color="black")
    plt.title("Original Signal")

    for i in range(min(K, max_imfs)):
        plt.subplot(n_plots, 1, i + 2)
        plt.plot(merged_imfs[i, :])
        plt.title(f"IMF {i + 1}")

    plt.subplot(n_plots, 1, n_plots)
    plt.plot(reconstructed, color="red")
    plt.title("Reconstructed Signal")

    plt.tight_layout()
    plt.show()


def export_to_csv(merged_imfs, reconstructed, save_path):
    """Export IMFs and reconstructed signal to CSV."""
    K = merged_imfs.shape[0]
    df = pd.DataFrame(merged_imfs.T, columns=[f"IMF_{i + 1}" for i in range(K)])
    df["Reconstructed"] = reconstructed
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"Results exported to: {save_path}")


# ==============================================================================
# Data I/O
# ==============================================================================


def read_csv_column(file_path, column, encoding="utf-16", start=None, end=None):
    """Read a single column from CSV and return as numpy array."""
    df = pd.read_csv(file_path, header=0, index_col=0, encoding=encoding)
    return df[column].values[start:end]


# ==============================================================================
# CLI Entry Point
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Sliding-Window VMD (SMVMD)")

    parser.add_argument("--data_file", type=str, required=True, help="Path to CSV (UTF-16)")
    parser.add_argument("--column", type=str, default="14", help="Target column name")
    parser.add_argument("--start", type=int, default=None, help="Row start index")
    parser.add_argument("--end", type=int, default=None, help="Row end index")

    parser.add_argument("--window_len", type=int, default=20600, help="Sliding window length")
    parser.add_argument("--overlap", type=float, default=0.4, help="Overlap ratio (0-1)")
    parser.add_argument("--K", type=int, default=9, help="Number of VMD modes")
    parser.add_argument("--alpha", type=float, default=2, help="VMD bandwidth constraint")

    parser.add_argument("--save_csv", type=str, default="smvmd_results.csv", help="Export CSV path")
    parser.add_argument("--no_plot", action="store_true", help="Disable visualization")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    signal = read_csv_column(
        args.data_file, column=args.column, start=args.start, end=args.end
    )

    results = sliding_window_vmd(
        data=signal,
        window_len=args.window_len,
        overlap_ratio=args.overlap,
        K=args.K,
        alpha=args.alpha,
    )

    print(f"\nIMF shape: {results['merged_imfs'].shape}")
    print(f"Reconstructed shape: {results['reconstructed_signal'].shape}")
    print(f"Reliable: {results['is_reliable']}")

    if not args.no_plot:
        plot_vmd_results(
            results["original_data"],
            results["reconstructed_signal"],
            results["merged_imfs"],
        )

    export_to_csv(results["merged_imfs"], results["reconstructed_signal"], args.save_csv)
