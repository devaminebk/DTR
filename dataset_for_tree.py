"""
Prepare dataset for scikit-learn Decision Tree / RandomForest regressors.
- Loads `Minput.npy` and `Moutput.npy` from script folder
- Converts to sklearn-friendly shapes: X (n_samples, n_features), y (n_samples, n_targets)
- Optional: replace angle feature (theta0 degrees) with sin/cos
- Optional: stratified split by binned total elements
- Saves: `X.npy`, `y.npy`, `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy` and .mat versions

Usage:
    python dataset_for_tree.py --test-size 0.15 --random-state 46 --transform-angles --stratify

"""
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
import scipy.io


def main(test_size=0.15, random_state=46, transform_angles=False, stratify=False, n_bins=5):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    Minput_path = os.path.join(script_dir, 'Minput.npy')
    Moutput_path = os.path.join(script_dir, 'Moutput.npy')

    if not os.path.exists(Minput_path) or not os.path.exists(Moutput_path):
        raise FileNotFoundError('Minput.npy or Moutput.npy not found in script directory.')

    Minput = np.load(Minput_path)    # shape likely (features, n_samples)
    Moutput = np.load(Moutput_path)  # shape likely (targets, n_samples)

    # Transpose if necessary to get (n_samples, features/targets)
    # Original files are saved as (features, n_samples) in this project, so
    # when rows < cols we transpose to get (n_samples, features).
    if Minput.shape[0] < Minput.shape[1]:
        Minput = Minput.T
    if Moutput.shape[0] < Moutput.shape[1]:
        Moutput = Moutput.T

    X = Minput.copy()
    y = Moutput.copy()

    # Optionally transform angle feature (assume angle is the last column or index 3)
    # Here dataset.py uses Minput rows: [mainLobe, SSL, HPBW, theta0]
    if transform_angles:
        # find theta column. If there are at least 4 features, assume index 3
        if X.shape[1] >= 4:
            theta_deg = X[:, 3]
            theta_rad = np.deg2rad(theta_deg)
            sin_t = np.sin(theta_rad)
            cos_t = np.cos(theta_rad)
            # Replace theta column with sin,cos
            X = np.concatenate([X[:, :3], sin_t.reshape(-1, 1), cos_t.reshape(-1, 1)], axis=1)
        else:
            print('Warning: cannot transform angles — expected at least 4 features.')

    # Save full arrays
    np.save(os.path.join(script_dir, 'X.npy'), X)
    np.save(os.path.join(script_dir, 'y.npy'), y)
    scipy.io.savemat(os.path.join(script_dir, 'X.mat'), {'X': X})
    scipy.io.savemat(os.path.join(script_dir, 'y.mat'), {'y': y})

    # Prepare stratify labels if requested
    stratify_labels = None
    if stratify:
        # compute total elements per sample and bin
        totals = np.sum(y, axis=1)
        bins = np.quantile(totals, np.linspace(0, 1, n_bins + 1))
        # digitize with care (avoids all-zero bins)
        stratify_labels = np.digitize(totals, bins[1:-1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_labels
    )

    # Save splits
    np.save(os.path.join(script_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(script_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(script_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(script_dir, 'y_test.npy'), y_test)

    scipy.io.savemat(os.path.join(script_dir, 'X_train.mat'), {'X_train': X_train})
    scipy.io.savemat(os.path.join(script_dir, 'X_test.mat'), {'X_test': X_test})
    scipy.io.savemat(os.path.join(script_dir, 'y_train.mat'), {'y_train': y_train})
    scipy.io.savemat(os.path.join(script_dir, 'y_test.mat'), {'y_test': y_test})

    print('Saved: X.npy, y.npy and train/test splits.')
    print('Shapes: X:', X.shape, 'y:', y.shape)
    print('Train shapes:', X_train.shape, y_train.shape)
    print('Test shapes:', X_test.shape, y_test.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset for Decision Tree training')
    parser.add_argument('--test-size', type=float, default=0.15)
    parser.add_argument('--random-state', type=int, default=46)
    parser.add_argument('--transform-angles', action='store_true')
    parser.add_argument('--stratify', action='store_true', help='Stratify by binned total elements')
    parser.add_argument('--n-bins', type=int, default=5)
    args = parser.parse_args()
    main(test_size=args.test_size, random_state=args.random_state,
         transform_angles=args.transform_angles, stratify=args.stratify, n_bins=args.n_bins)
