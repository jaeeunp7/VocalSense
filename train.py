"""
train.py

Loads recorded CSVs, slices them into windows, extracts features, and
trains a small classifier. Saves the trained model to model.pkl so
live_viewer.py can pick it up.

Usage:
    python train.py --data data/ --out model.pkl

Outputs:
    - model.pkl                 (pickled scikit-learn pipeline)
    - confusion_matrix.png      (held-out test set confusion matrix)
    - Printed classification report
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
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# must match live_viewer.py
SAMPLE_RATE = 1000
WINDOW_MS = 100
HOP_MS = 50
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_MS / 1000)
HOP_SAMPLES = int(SAMPLE_RATE * HOP_MS / 1000)


def extract_features(emg_window, audio_window):
    """Same feature function used at inference time. Keep these in sync."""
    emg = np.asarray(emg_window, dtype=np.float32)
    audio = np.asarray(audio_window, dtype=np.float32)
    emg = emg - emg.mean()
    audio = audio - audio.mean()
    emg_rms = float(np.sqrt(np.mean(emg ** 2)))
    audio_rms = float(np.sqrt(np.mean(audio ** 2)))
    emg_zcr = float(np.mean(np.diff(np.sign(emg)) != 0))
    audio_zcr = float(np.mean(np.diff(np.sign(audio)) != 0))
    return [emg_rms, emg_zcr, audio_rms, audio_zcr]


def windowize(df):
    """Slide a window across one CSV. Yields (features, label) per window."""
    emg = df['emg'].values
    audio = df['audio'].values
    labels = df['label'].values

    for start in range(0, len(emg) - WINDOW_SAMPLES, HOP_SAMPLES):
        end = start + WINDOW_SAMPLES
        # only use the window if the label is consistent across it
        window_labels = labels[start:end]
        if len(set(window_labels)) != 1:
            continue
        feats = extract_features(emg[start:end], audio[start:end])
        yield feats, window_labels[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data', help='Directory of CSVs')
    parser.add_argument('--out', default='model.pkl')
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.data, '*.csv')))
    if not csv_files:
        raise SystemExit(f'No CSVs found in {args.data}')
    print(f'Found {len(csv_files)} CSV file(s):')
    for f in csv_files:
        print(f'  {f}')

    X, y = [], []
    for path in csv_files:
        df = pd.read_csv(path)
        for feats, label in windowize(df):
            X.append(feats)
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    print(f'\nExtracted {len(X)} windows across {len(set(y))} classes')
    print('Class counts:')
    for cls in sorted(set(y)):
        print(f'  {cls:15s} {(y == cls).sum()}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print('\nClassification report (held-out test set):')
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y)))
    disp = ConfusionMatrixDisplay(cm, display_labels=sorted(set(y)))
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=120)
    print('Saved confusion_matrix.png')

    with open(args.out, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f'Saved model to {args.out}')


if __name__ == '__main__':
    main()
