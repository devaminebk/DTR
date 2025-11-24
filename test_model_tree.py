"""
Test script for Decision Tree / RandomForest models.
- Loads `X_test.npy`, `y_test.npy` (created by `dataset_for_tree.py`) or falls back to `Minput_test.npy` / `Moutput_test.npy`.
- Loads the trained model (`decision_tree_model.joblib` or `random_forest_model.joblib` by default).
- Predicts batch, rounds/clips predictions to integer counts, computes overall metrics.
- Plots radiation diagram for a chosen sample index (default 5) comparing reference vs prediction.

Usage:
    python test_model_tree.py --index 5 --model-file decision_tree_model.joblib

"""
import os
import argparse
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Physical constants (match dataset and test_model_FFNN)
carrierFreq = 2.45e9
c = 3e8
lambda_ = c / carrierFreq
r0 = 0.2 * lambda_
delta_r = 0.5 * lambda_
max_rings = 5


def calcul_AF_performance_metrics(elements_per_ring, theta0=0.0, phi0=0.0, max_rings=max_rings):
    # elements_per_ring: iterable of ints per ring
    k = 2 * np.pi / lambda_
    theta = np.linspace(0, 2 * np.pi, 1000)
    AF_az = np.zeros_like(theta, dtype=complex)
    radii = r0 + delta_r * np.arange(max_rings)
    for ring in range(len(elements_per_ring)):
        a = radii[ring]
        N = int(elements_per_ring[ring])
        if N == 0:
            continue
        phi_n = 2 * np.pi * np.arange(N) / N
        for n in range(N):
            phase = k * a * (np.sin(theta) * np.cos(phi0 - phi_n[n]) -
                             np.sin(theta0) * np.cos(phi0 - phi_n[n]))
            AF_az += np.exp(1j * phase)

    AF_norm_az = np.abs(AF_az) / (np.max(np.abs(AF_az)) + np.finfo(float).eps)
    AF_dB_az = 20 * np.log10(AF_norm_az + np.finfo(float).eps)
    AF_dB_az[AF_dB_az < -40] = -40

    AF_abs_az = np.abs(AF_az)
    maxVal = np.max(AF_abs_az)
    maxVal_non_norm = 20 * np.log10(maxVal + np.finfo(float).eps)

    maxVal_dB = np.max(AF_dB_az)
    maxIdx = np.argmax(AF_dB_az)
    halfPower = maxVal_dB - 3
    AF_dB_ext = np.concatenate((AF_dB_az, AF_dB_az, AF_dB_az))
    theta_deg = np.rad2deg(theta)
    theta_deg_ext = np.concatenate((theta_deg - 360, theta_deg, theta_deg + 360))
    maxIdx_ext = maxIdx + len(theta_deg)

    leftIdx_ext = np.where(AF_dB_ext[:maxIdx_ext] <= halfPower)[0][-1] if np.any(AF_dB_ext[:maxIdx_ext] <= halfPower) else None
    rightIdx_ext = np.where(AF_dB_ext[maxIdx_ext:] <= halfPower)[0]
    rightIdx_ext = rightIdx_ext[0] + maxIdx_ext if len(rightIdx_ext) > 0 else None

    if leftIdx_ext is None or rightIdx_ext is None:
        HPBW = 180
    else:
        HPBW = theta_deg_ext[rightIdx_ext] - theta_deg_ext[leftIdx_ext]

    responseLin = AF_norm_az
    peaks, _ = find_peaks(responseLin, distance=5)
    pk = responseLin[peaks]
    if len(pk) == 0:
        main_lobe_gain = maxVal_non_norm
        true_SSL_gain = 0
    else:
        sorted_idx = np.argsort(pk)[::-1]
        sorted_pk = pk[sorted_idx]
        threshold_dB = 1
        main_lobes_idx = np.where(20 * np.log10(sorted_pk + np.finfo(float).eps) >=
                                  20 * np.log10(sorted_pk[0] + np.finfo(float).eps) - threshold_dB)[0]
        side_lobe_idx = np.setdiff1d(np.arange(len(sorted_pk)), main_lobes_idx)
        main_lobe_gain = maxVal_non_norm
        true_SSL_gain = 20 * np.log10(sorted_pk[side_lobe_idx[0]] + np.finfo(float).eps) if len(side_lobe_idx) > 0 else 0

    return AF_dB_az, HPBW, main_lobe_gain, true_SSL_gain


def load_test_data(script_dir):
    # Prefer X_test/y_test split created by dataset_for_tree.py
    pX = os.path.join(script_dir, 'X_test.npy')
    py = os.path.join(script_dir, 'y_test.npy')
    if os.path.exists(pX) and os.path.exists(py):
        X_test = np.load(pX)
        y_test = np.load(py)
        return X_test, y_test
    # Fallback to original Minput/Moutput naming
    alt_X = os.path.join(script_dir, 'Minput_test.npy')
    alt_y = os.path.join(script_dir, 'Moutput_test.npy')
    if os.path.exists(alt_X) and os.path.exists(alt_y):
        X = np.load(alt_X)
        y = np.load(alt_y)
        # transpose if necessary (project stores features x samples)
        if X.shape[0] < X.shape[1]:
            X = X.T
        if y.shape[0] < y.shape[1]:
            y = y.T
        return X, y
    raise FileNotFoundError('No test files found. Run dataset_for_tree.py first or provide test splits.')


def find_model_file(script_dir, provided=None):
    if provided is not None and os.path.exists(os.path.join(script_dir, provided)):
        return os.path.join(script_dir, provided)
    # common names
    for name in ['decision_tree_model.joblib', 'random_forest_model.joblib', 'decision_tree_model.pkl', 'model.joblib']:
        p = os.path.join(script_dir, name)
        if os.path.exists(p):
            return p
    # fallback: any .joblib in dir
    for f in os.listdir(script_dir):
        if f.endswith('.joblib') or f.endswith('.pkl'):
            return os.path.join(script_dir, f)
    return None


def main(index=5, model_file=None, show_plot=True):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    X_test, y_test = load_test_data(script_dir)
    print('Loaded X_test', X_test.shape, 'y_test', y_test.shape)

    model_path = find_model_file(script_dir, model_file)
    if model_path is None:
        raise FileNotFoundError('No model file found. Train a model first with train_decision_tree.py')
    print('Loading model from', model_path)
    model = joblib.load(model_path)

    # Predict batch
    y_pred = model.predict(X_test)
    # determine clipping max
    max_elements = int(np.max(y_test)) if y_test.size>0 else 10
    y_pred_rounded = np.clip(np.round(y_pred).astype(int), 0, max_elements)

    # Overall numeric metrics
    mae_per_target = np.mean(np.abs(y_pred - y_test), axis=0)
    overall_mae = np.mean(mae_per_target)
    exact_match_rate = np.mean(np.all(y_pred_rounded == y_test.astype(int), axis=1))

    print('\nMAE per target:', mae_per_target)
    print('Overall MAE:', overall_mae)
    print('Exact architecture match rate:', exact_match_rate)

    # Compute antenna metrics across test set (HPBW, gain, SSL)
    HPBW_mae_list = []
    gain_mae_list = []
    SSL_mae_list = []
    for i in range(len(X_test)):
        arch_ref = y_test[i].astype(int)
        arch_pred = y_pred_rounded[i]
        # extract theta0 from feature (if present as 4th feature)
        if X_test.shape[1] >= 4:
            theta0deg = X_test[i, 3]
            theta0 = np.deg2rad(theta0deg)
        else:
            theta0 = 0.0
        _, HPBW_ref, gain_ref, SSL_ref = calcul_AF_performance_metrics(arch_ref, theta0=theta0)
        _, HPBW_pred, gain_pred, SSL_pred = calcul_AF_performance_metrics(arch_pred, theta0=theta0)
        HPBW_mae_list.append(abs(HPBW_ref - HPBW_pred))
        gain_mae_list.append(abs(gain_ref - gain_pred))
        SSL_mae_list.append(abs(SSL_ref - SSL_pred))

    print('\nAntenna metric MAE (over test set):')
    print('HPBW MAE:', np.mean(HPBW_mae_list))
    print('Main-lobe gain MAE:', np.mean(gain_mae_list))
    print('SSL MAE:', np.mean(SSL_mae_list))

    # Pick a sample to plot
    idx = index
    if idx < 0 or idx >= len(X_test):
        idx = 0
    arch_ref = y_test[idx].astype(int)
    arch_pred = y_pred_rounded[idx]
    theta = np.linspace(0, 2 * np.pi, 1000)
    theta_deg = np.rad2deg(theta)
    if X_test.shape[1] >= 4:
        theta0deg = X_test[idx, 3]
        theta0 = np.deg2rad(theta0deg)
    else:
        theta0deg = 0.0
        theta0 = 0.0

    AF_ref_dB, HPBW_ref, gain_main_ref, SSL_ref = calcul_AF_performance_metrics(arch_ref, theta0=theta0)
    AF_pred_dB, HPBW_pred, gain_main_pred, SSL_pred = calcul_AF_performance_metrics(arch_pred, theta0=theta0)

    print('\nSample index', idx)
    print('Reference architecture :', arch_ref)
    print('Predicted architecture :', arch_pred)
    print('\nReference metrics: Gain', gain_main_ref, 'SSL', SSL_ref, 'HPBW', HPBW_ref)
    print('Predicted metrics:  Gain', gain_main_pred, 'SSL', SSL_pred, 'HPBW', HPBW_pred)

    if show_plot:
        # polar
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(111, polar=True)
        ax.plot(theta, AF_ref_dB, label='Référence', linewidth=2)
        ax.plot(theta, AF_pred_dB, label='Prédiction', linewidth=2, linestyle='--')
        ax.set_title(f"Diagramme de rayonnement (sample {idx})", fontsize=13)
        ax.set_rlim([-40, 0])
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

        # cartesian
        plt.figure(figsize=(10, 5))
        plt.plot(theta_deg, AF_ref_dB, label='Référence', linewidth=2)
        plt.plot(theta_deg, AF_pred_dB, label='Prédiction', linewidth=2, linestyle='--')
        plt.xlabel('Azimut (°)')
        plt.ylabel('Gain (dB)')
        plt.title(f'Diagramme de rayonnement - Azimut (sample {idx})')
        plt.xlim([0, 360])
        plt.ylim([-40, 0])
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # save predictions for this run
    np.save(os.path.join(script_dir, 'y_pred_tree.npy'), y_pred)
    np.save(os.path.join(script_dir, 'y_pred_tree_rounded.npy'), y_pred_rounded)
    print('\nSaved `y_pred_tree.npy` and `y_pred_tree_rounded.npy` in script directory.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Decision Tree / RandomForest model and plot example radiation pattern')
    parser.add_argument('--index', type=int, default=5, help='Test sample index to visualize')
    parser.add_argument('--model-file', type=str, default=None, help='Name of model file to load (joblib/pkl)')
    parser.add_argument('--no-plot', dest='plot', action='store_false', help='Do not show plots')
    args = parser.parse_args()
    main(index=args.index, model_file=args.model_file, show_plot=args.plot)
