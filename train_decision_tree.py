"""
Train and evaluate a Decision Tree (or RandomForest) regressor on the prepared dataset.
- Loads `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy` (created by `dataset_for_tree.py`)
- Trains a multi-output DecisionTreeRegressor (or RandomForestRegressor)
- Evaluates numeric metrics (MAE per target, overall MAE), exact-match rates for rounded integer outputs
- Computes antenna metrics (HPBW, main lobe gain, SSL) for predicted architectures and reports MAE
- Saves model and predictions

Usage:
    python train_decision_tree.py --model decision_tree
    python train_decision_tree.py --model random_forest --n-estimators 100

"""
import os
import numpy as np
import argparse
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import joblib
import scipy.io


# Re-implement antenna metric function (copied/adapted from test_model_FFNN.py)
from scipy.signal import find_peaks

def calcul_AF_performance_metrics(elements_per_ring, r0=0.2*3e8/2.45e9, delta_r=0.5*3e8/2.45e9, max_rings=5):
    carrierFreq = 2.45e9
    c = 3e8
    lambda_ = c / carrierFreq
    k = 2 * np.pi / lambda_
    theta0deg = 0  # caller should set correct theta0 if needed; in practice angle enters from input features
    theta0 = 0
    phi0 = 0
    # we will compute AF independent of theta0 here; user can adapt if needed

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


def load_data(script_dir):
    # prefer X_train/X_test if created by dataset_for_tree, otherwise fall back to Minput_training format
    X_train_path = os.path.join(script_dir, 'X_train.npy')
    X_test_path = os.path.join(script_dir, 'X_test.npy')
    y_train_path = os.path.join(script_dir, 'y_train.npy')
    y_test_path = os.path.join(script_dir, 'y_test.npy')

    if os.path.exists(X_train_path) and os.path.exists(y_train_path) and os.path.exists(X_test_path) and os.path.exists(y_test_path):
        X_train = np.load(X_train_path)
        X_test = np.load(X_test_path)
        y_train = np.load(y_train_path)
        y_test = np.load(y_test_path)
    else:
        # try older naming Minput_training.npy (features x n_samples)
        alt_train_in = os.path.join(script_dir, 'Minput_training.npy')
        alt_train_out = os.path.join(script_dir, 'Moutput_training.npy')
        alt_test_in = os.path.join(script_dir, 'Minput_test.npy')
        alt_test_out = os.path.join(script_dir, 'Moutput_test.npy')
        if os.path.exists(alt_train_in) and os.path.exists(alt_train_out) and os.path.exists(alt_test_in) and os.path.exists(alt_test_out):
            X_train = np.load(alt_train_in)
            y_train = np.load(alt_train_out)
            X_test = np.load(alt_test_in)
            y_test = np.load(alt_test_out)
            # transpose to sklearn format if needed
            if X_train.shape[0] > X_train.shape[1]:
                X_train = X_train.T
            if X_test.shape[0] > X_test.shape[1]:
                X_test = X_test.T
            if y_train.shape[0] > y_train.shape[1]:
                y_train = y_train.T
            if y_test.shape[0] > y_test.shape[1]:
                y_test = y_test.T
        else:
            # fallback: try Minput.npy / Moutput.npy full dataset and split locally
            Xp = os.path.join(script_dir, 'X.npy')
            yp = os.path.join(script_dir, 'y.npy')
            if os.path.exists(Xp) and os.path.exists(yp):
                X_full = np.load(Xp)
                y_full = np.load(yp)
                # default split
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.15, random_state=46)
            else:
                raise FileNotFoundError('No suitable input/train/test files found. Run dataset_for_tree.py first or provide training files.')

    return X_train, X_test, y_train, y_test


def main(model_name='decision_tree', random_state=46, gridsearch=False, n_estimators=100):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    X_train, X_test, y_train, y_test = load_data(script_dir)
    print('Loaded data. Shapes:', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # determine max_elements for clipping predicted integers
    max_elements = int(np.max(y_train)) if y_train.size > 0 else 10

    if model_name == 'decision_tree':
        model = DecisionTreeRegressor(random_state=random_state)
        if gridsearch:
            param_grid = {'max_depth': [None, 5, 10, 20], 'min_samples_leaf': [1, 2, 5, 10]}
            grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            print('Best params:', grid.best_params_)
        else:
            model.fit(X_train, y_train)
    elif model_name == 'random_forest':
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        if gridsearch:
            param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20], 'min_samples_leaf': [1, 2, 5]}
            grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            print('Best params:', grid.best_params_)
        else:
            model.fit(X_train, y_train)
    else:
        raise ValueError('Unknown model_name')

    # Predict on test set
    y_pred = model.predict(X_test)

    # Continuous metrics
    mae_per_target = np.mean(np.abs(y_pred - y_test), axis=0)
    overall_mae = np.mean(mae_per_target)
    print('\nMAE per target:', mae_per_target)
    print('Overall MAE:', overall_mae)

    # Round and clip to integer architecture counts
    y_pred_rounded = np.clip(np.round(y_pred).astype(int), 0, max_elements)

    # Exact-match and per-ring accuracy
    exact_matches = np.all(y_pred_rounded == y_test.astype(int), axis=1)
    exact_match_rate = np.mean(exact_matches)
    per_ring_accuracy = np.mean(y_pred_rounded == y_test.astype(int), axis=0)

    print('\nExact architecture match rate:', exact_match_rate)
    print('Per-ring accuracy:', per_ring_accuracy)

    # Compute antenna metrics across test set
    HPBW_mae_list = []
    gain_mae_list = []
    SSL_mae_list = []
    for i in range(len(X_test)):
        arch_ref = y_test[i].astype(int)
        arch_pred = y_pred_rounded[i]
        _, HPBW_ref, gain_ref, SSL_ref = calcul_AF_performance_metrics(arch_ref)
        _, HPBW_pred, gain_pred, SSL_pred = calcul_AF_performance_metrics(arch_pred)
        HPBW_mae_list.append(abs(HPBW_ref - HPBW_pred))
        gain_mae_list.append(abs(gain_ref - gain_pred))
        SSL_mae_list.append(abs(SSL_ref - SSL_pred))

    print('\nAntenna metric MAE (over test set):')
    print('HPBW MAE:', np.mean(HPBW_mae_list))
    print('Main-lobe gain MAE:', np.mean(gain_mae_list))
    print('SSL MAE:', np.mean(SSL_mae_list))

    # Save model and predictions
    model_path = os.path.join(script_dir, f'{model_name}_model.joblib')
    joblib.dump(model, model_path)
    np.save(os.path.join(script_dir, f'y_pred_{model_name}.npy'), y_pred)
    np.save(os.path.join(script_dir, f'y_pred_rounded_{model_name}.npy'), y_pred_rounded)

    # Save metrics to .npz
    np.savez(os.path.join(script_dir, f'{model_name}_metrics.npz'),
             mae_per_target=mae_per_target,
             overall_mae=overall_mae,
             exact_match_rate=exact_match_rate,
             per_ring_accuracy=per_ring_accuracy,
             HPBW_mae=np.mean(HPBW_mae_list),
             gain_mae=np.mean(gain_mae_list),
             SSL_mae=np.mean(SSL_mae_list))

    # also save a .mat for compatibility
    scipy.io.savemat(os.path.join(script_dir, f'{model_name}_results.mat'),
                     {'y_pred': y_pred, 'y_pred_rounded': y_pred_rounded, 'mae_per_target': mae_per_target})

    print('\nSaved model and prediction files in script directory.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DecisionTree/RandomForest on antenna dataset')
    parser.add_argument('--model', choices=['decision_tree', 'random_forest'], default='decision_tree')
    parser.add_argument('--random-state', type=int, default=46)
    parser.add_argument('--gridsearch', action='store_true')
    parser.add_argument('--n-estimators', type=int, default=100)
    args = parser.parse_args()
    main(model_name=args.model, random_state=args.random_state, gridsearch=args.gridsearch, n_estimators=args.n_estimators)
