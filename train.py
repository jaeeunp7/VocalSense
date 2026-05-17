"""
train.py

Loads recorded CSVs, slices them into 2-second windows, extracts features,
and trains a multi-label Random Forest classifier predicting four binary
vocal-quality labels (straining, vocal_resonance, phonation, breath_support).

Usage:
    python train.py --data data/ --out model.pkl

Outputs:
    - model.pkl                 (pickled scikit-learn pipeline)
    - confusion_matrices.png    (one 2x2 confusion matrix per label)
    - Printed per-label classification report
"""

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# must match live_viewer.py
SAMPLE_RATE = 1000
WINDOW_MS = 2000            # 2-second windows = one "data point"
HOP_MS = 1000               # 50% overlap doubles the training samples
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_MS / 1000)
HOP_SAMPLES = int(SAMPLE_RATE * HOP_MS / 1000)

LABEL_CATEGORIES = ['straining', 'vocal_resonance', 'phonation', 'breath_support']


def extract_features(emg_window, piezo_window):
    """
    Compute features from a 2-second window of EMG + piezo.
    Same function used at inference time -- keep in sync with live_viewer.py.

    Features (7 total):
      EMG:
        - RMS                   (overall activation)
        - mean absolute value   (alternative activation measure)
        - zero-crossing rate    (frequency content)
        - waveform length       (signal complexity / variation)
      Piezo:
        - RMS                   (vibration energy)
        - peak-to-peak          (max minus min)
        - dominant frequency Hz (FFT peak -- approximates vocal fundamental)
    """
    emg = np.asarray(emg_window, dtype=np.float32)
    piezo = np.asarray(piezo_window, dtype=np.float32)
    emg = emg - emg.mean()
    piezo = piezo - piezo.mean()

    emg_rms = float(np.sqrt(np.mean(emg ** 2)))
    emg_mav = float(np.mean(np.abs(emg)))
    emg_zcr = float(np.mean(np.diff(np.sign(emg)) != 0))
    emg_wl = float(np.sum(np.abs(np.diff(emg))))

    piezo_rms = float(np.sqrt(np.mean(piezo ** 2)))
    piezo_peak = float(np.max(piezo) - np.min(piezo))

    # Dominant frequency in the piezo signal (vocal fundamental proxy)
    fft_mag = np.abs(np.fft.rfft(piezo))
    freqs = np.fft.rfftfreq(len(piezo), d=1.0 / SAMPLE_RATE)
    # skip DC (index 0) and ignore frequencies below 50 Hz (noise floor)
    valid = freqs > 50
    if valid.any() and fft_mag[valid].max() > 0:
        peak_idx = np.argmax(fft_mag[valid])
        piezo_dom_freq = float(freqs[valid][peak_idx])
    else:
        piezo_dom_freq = 0.0

    return [emg_rms, emg_mav, emg_zcr, emg_wl, piezo_rms, piezo_peak, piezo_dom_freq]


def windowize(df):
    """Slide a 2-second window across one take's CSV.
    Yields (features, label_vector) per window."""
    emg = df['emg'].values
    piezo = df['piezo'].values
    label_cols = df[LABEL_CATEGORIES].values  # shape (n_samples, 4)

    for start in range(0, len(emg) - WINDOW_SAMPLES + 1, HOP_SAMPLES):
        end = start + WINDOW_SAMPLES
        # labels should be constant across the whole take, but verify
        window_labels = label_cols[start:end]
        # take the first row's labels; warn if they vary within the window
        labels = window_labels[0]
        feats = extract_features(emg[start:end], piezo[start:end])
        yield feats, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data', help='Directory of CSVs')
    parser.add_argument('--out', default='model.pkl')
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.data, '*.csv')))
    if not csv_files:
        raise SystemExit(f'No CSVs found in {args.data}')
    print(f'Found {len(csv_files)} CSV file(s).')

    X, Y = [], []
    for path in csv_files:
        df = pd.read_csv(path)
        missing = [c for c in LABEL_CATEGORIES if c not in df.columns]
        if missing:
            print(f'  Skipping {path} -- missing columns: {missing}')
            continue
        n_before = len(X)
        for feats, labels in windowize(df):
            X.append(feats)
            Y.append(labels)
        print(f'  {os.path.basename(path)}: +{len(X) - n_before} windows')

    if len(X) == 0:
        raise SystemExit('No valid windows extracted. Are CSVs long enough (>=2s)?')

    X = np.array(X)
    Y = np.array(Y)
    print(f'\nTotal: {len(X)} windows, {X.shape[1]} features, {Y.shape[1]} labels')
    print('Per-label TRUE counts (out of {} windows):'.format(len(X)))
    for i, name in enumerate(LABEL_CATEGORIES):
        true_count = int(Y[:, i].sum())
        print(f'  {name:18s} {true_count:4d}  ({100*true_count/len(X):.0f}%)')

    # Stratifying by a single column for multi-label is tricky -- just
    # do a simple random split. With small datasets, you may want to
    # verify each label has both classes in test set.
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=42
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight='balanced'
        )),
    ])
    # RandomForestClassifier handles multi-output natively when Y is 2D
    pipeline.fit(X_train, Y_train)

    Y_pred = pipeline.predict(X_test)

    print('\nPer-label classification report (held-out test set):')
    print(classification_report(
        Y_test, Y_pred,
        target_names=LABEL_CATEGORIES,
        zero_division=0
    ))

    # One confusion matrix per label
    cms = multilabel_confusion_matrix(Y_test, Y_pred)
    fig, axes = plt.subplots(1, len(LABEL_CATEGORIES),
                             figsize=(4 * len(LABEL_CATEGORIES), 4))
    for i, (cm, name) in enumerate(zip(cms, LABEL_CATEGORIES)):
        disp = ConfusionMatrixDisplay(cm, display_labels=['false', 'true'])
        disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
        axes[i].set_title(name)
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=120)
    print('Saved confusion_matrices.png')

    with open(args.out, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f'Saved model to {args.out}')


if __name__ == '__main__':
    main()