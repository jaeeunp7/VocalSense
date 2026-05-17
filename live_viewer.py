"""
live_viewer.py

Reads EMG + piezo samples from the Arduino over Serial, extracts features
from rolling 2-second windows, and shows a live display with EMG trace,
piezo trace, and live predictions of four binary vocal-quality labels.

Usage:
    python live_viewer.py --port COM3                  (Windows)
    python live_viewer.py --port /dev/cu.usbmodem...   (Mac)

Dependencies: pyserial numpy matplotlib scikit-learn
"""

import argparse
import os
import time
from collections import deque
import threading

import numpy as np
import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---- config (must match train.py) ----------------------------------------

SAMPLE_RATE = 1000
WINDOW_MS = 2000            # 2-second windows
HOP_MS = 500                # predict 2x per second
DISPLAY_SECONDS = 4         # show 4s of history for context around the 2s window
ADC_MAX = 1023
MODEL_PATH = 'model.pkl'

WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_MS / 1000)
HOP_SAMPLES = int(SAMPLE_RATE * HOP_MS / 1000)
DISPLAY_SAMPLES = SAMPLE_RATE * DISPLAY_SECONDS

LABEL_CATEGORIES = ['straining', 'vocal_resonance', 'phonation', 'breath_support']

# ---- shared buffers -------------------------------------------------------

emg_buffer = deque(maxlen=DISPLAY_SAMPLES)
piezo_buffer = deque(maxlen=DISPLAY_SAMPLES)
buffer_lock = threading.Lock()

# predictions: each is {'value': 0/1, 'confidence': 0-1}
latest_predictions = {k: {'value': 0, 'confidence': 0.0} for k in LABEL_CATEGORIES}
status_message = 'starting...'


# ---- feature extraction (MUST match train.py exactly) --------------------

def extract_features(emg_window, piezo_window):
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

    fft_mag = np.abs(np.fft.rfft(piezo))
    freqs = np.fft.rfftfreq(len(piezo), d=1.0 / SAMPLE_RATE)
    valid = freqs > 50
    if valid.any() and fft_mag[valid].max() > 0:
        peak_idx = np.argmax(fft_mag[valid])
        piezo_dom_freq = float(freqs[valid][peak_idx])
    else:
        piezo_dom_freq = 0.0

    return np.array(
        [emg_rms, emg_mav, emg_zcr, emg_wl, piezo_rms, piezo_peak, piezo_dom_freq],
        dtype=np.float32
    )


# ---- serial reader thread -------------------------------------------------

def serial_reader(port, baud=500000):
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    while True:
        try:
            line = ser.readline().decode('ascii', errors='ignore').strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 2:
                continue
            emg_val = int(parts[0])
            piezo_val = int(parts[1])
            with buffer_lock:
                emg_buffer.append(emg_val)
                piezo_buffer.append(piezo_val)
        except (ValueError, UnicodeDecodeError):
            continue


# ---- classifier thread ----------------------------------------------------

def classifier_loop():
    global latest_predictions, status_message

    model = None
    if os.path.exists(MODEL_PATH):
        import pickle
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f'Loaded model from {MODEL_PATH}')
        status_message = 'predicting'
    else:
        status_message = 'no model -- run train.py first'

    while True:
        time.sleep(HOP_MS / 1000)
        if model is None:
            continue
        with buffer_lock:
            if len(emg_buffer) < WINDOW_SAMPLES:
                continue
            emg_win = list(emg_buffer)[-WINDOW_SAMPLES:]
            piezo_win = list(piezo_buffer)[-WINDOW_SAMPLES:]

        feats = extract_features(emg_win, piezo_win).reshape(1, -1)
        try:
            preds = model.predict(feats)[0]  # shape (4,)

            # multi-output classifiers return list of arrays for predict_proba
            confs = [0.5] * len(LABEL_CATEGORIES)
            if hasattr(model, 'predict_proba'):
                try:
                    probas = model.predict_proba(feats)
                    # RandomForestClassifier with multi-output returns a list:
                    # [array for label 0, array for label 1, ...] each (1, 2)
                    for i, p in enumerate(probas):
                        confs[i] = float(p[0].max())
                except Exception:
                    pass

            for i, name in enumerate(LABEL_CATEGORIES):
                latest_predictions[name] = {
                    'value': int(preds[i]),
                    'confidence': confs[i],
                }
        except Exception as e:
            status_message = f'error: {e}'


# ---- live plot ------------------------------------------------------------

def make_plot():
    fig, (ax_emg, ax_piezo, ax_pred) = plt.subplots(
        3, 1, figsize=(10, 7),
        gridspec_kw={'height_ratios': [3, 3, 2]}
    )
    fig.suptitle('EMG + Piezo Vocal Quality Monitor', fontsize=12)

    line_emg, = ax_emg.plot([], [], lw=1, color='tab:blue')
    ax_emg.set_xlim(0, DISPLAY_SECONDS)
    ax_emg.set_ylim(0, ADC_MAX)
    ax_emg.set_ylabel('EMG (raw ADC)')
    ax_emg.grid(alpha=0.3)

    line_piezo, = ax_piezo.plot([], [], lw=1, color='tab:green')
    ax_piezo.set_xlim(0, DISPLAY_SECONDS)
    ax_piezo.set_ylim(0, ADC_MAX)
    ax_piezo.set_ylabel('Piezo (raw ADC)')
    ax_piezo.set_xlabel('time (s)')
    ax_piezo.grid(alpha=0.3)

    # Build one row per label in the prediction panel
    ax_pred.axis('off')
    label_texts = {}
    n = len(LABEL_CATEGORIES)
    for i, name in enumerate(LABEL_CATEGORIES):
        y = 1.0 - (i + 0.5) / n
        # left side: label name
        ax_pred.text(0.05, y, name.replace('_', ' '),
                     ha='left', va='center', fontsize=12,
                     transform=ax_pred.transAxes)
        # right side: prediction + confidence (filled in by update)
        label_texts[name] = ax_pred.text(
            0.95, y, '', ha='right', va='center',
            fontsize=14, fontweight='bold',
            transform=ax_pred.transAxes
        )

    def update(_frame):
        with buffer_lock:
            emg = np.array(emg_buffer)
            piezo = np.array(piezo_buffer)

        if len(emg) > 0:
            t = np.arange(len(emg)) / SAMPLE_RATE
            line_emg.set_data(t, emg)
            ax_emg.set_xlim(max(0, t[-1] - DISPLAY_SECONDS), max(DISPLAY_SECONDS, t[-1]))
        if len(piezo) > 0:
            t = np.arange(len(piezo)) / SAMPLE_RATE
            line_piezo.set_data(t, piezo)
            ax_piezo.set_xlim(max(0, t[-1] - DISPLAY_SECONDS), max(DISPLAY_SECONDS, t[-1]))

        for name in LABEL_CATEGORIES:
            pred = latest_predictions[name]
            mark = 'TRUE' if pred['value'] else 'false'
            color = 'tab:green' if pred['value'] else 'tab:gray'
            label_texts[name].set_text(f"{mark}  ({pred['confidence']*100:.0f}%)")
            label_texts[name].set_color(color)
        return list(label_texts.values()) + [line_emg, line_piezo]

    ani = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', required=True)
    parser.add_argument('--baud', type=int, default=500000)
    args = parser.parse_args()

    t1 = threading.Thread(target=serial_reader, args=(args.port, args.baud), daemon=True)
    t2 = threading.Thread(target=classifier_loop, daemon=True)
    t1.start()
    t2.start()
    make_plot()


if __name__ == '__main__':
    main()