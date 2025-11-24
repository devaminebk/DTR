Project: Predict circular antenna architecture from pattern features

Execution pipeline (recommended):
1. Generate dataset (synthetic architectures + simulations)
   - `python dataset.py`
   - Outputs: `Minput.npy` (features), `Moutput.npy` (architectures)

2. Prepare data for sklearn (reshape, optional angle transform, split)
   - `python dataset_for_tree.py [--test-size FLOAT] [--random-state INT] [--transform-angles] [--stratify] [--n-bins INT]`
   - Examples:
	   - With angle continuity transform and stratified split:
		   `python dataset_for_tree.py --test-size 0.15 --random-state 46 --transform-angles --stratify --n-bins 5`
	   - Without angle transform (raw degrees):
		   `python dataset_for_tree.py --test-size 0.15 --random-state 46 --stratify`
   - Outputs: `X.npy`, `y.npy`, `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy` (+ `.mat` versions)

3. Train model (Decision Tree or Random Forest)
   - `python train_decision_tree.py --model {decision_tree|random_forest} [--gridsearch] [--n-estimators N] [--plot-tree]`
   - Examples:
	   - Decision Tree with tree visualization:
		   `python train_decision_tree.py --model decision_tree --plot-tree`
	   - Random Forest with 100 trees:
		   `python train_decision_tree.py --model random_forest --n-estimators 100`
   - Outputs: model `.joblib`, predictions `y_pred_*.npy`, rounded predictions, metrics `.npz`/`.mat`, and optional `<model>_tree.png` when `--plot-tree` is used.

4. Test / visualize a sample
   - `python test_model_tree.py [--index INT] [--model-file FILE] [--no-plot]`
   - Example: `python test_model_tree.py --index 5`

Notes on options:
- `--transform-angles`: replaces raw `theta0` (degrees) by `sin(theta0)` and `cos(theta0)` to remove circular discontinuity.
- `--stratify`: stratifies the train/test split by binned total element count; use when you want balanced sparse/dense distributions. If you get errors, reduce `--n-bins` or omit `--stratify`.
- `--gridsearch`: runs GridSearchCV to tune hyperparameters (can be slow).
- `--plot-tree`: saves a PNG of the trained decision tree (or the first tree in a RandomForest). Previous PNG is deleted before saving a new one.

Typical quick-run (PowerShell):
```powershell
python dataset.py
python dataset_for_tree.py --test-size 0.15 --random-state 46 --transform-angles --stratify --n-bins 5
python train_decision_tree.py --model decision_tree --plot-tree
python test_model_tree.py --index 5
```

Contact / Notes:
- `dataset.py` creates the synthetic `Minput` (features: main-lobe gain, SSL, HPBW, theta0) and `Moutput` (architecture: elements per ring).
- Use a held-out `X_test/y_test` for final evaluation; do not use the test set for hyperparameter tuning.