"""
live_viewer.py

Reads EMG + audio samples from the Arduino over Serial, extracts windowed
RMS features in real time, and shows a live matplotlib display with three
panels: raw EMG, raw audio, and the current classifier prediction.

Usage:
    python live_viewer.py --port /dev/cu.usbmodem14101  (Mac)
    python live_viewer.py --port COM3                   (Windows)
    python live_viewer.py --port /dev/ttyUSB0           (Linux)

Dependencies:
    pip install pyserial numpy matplotlib scikit-learn

The classifier is loaded if a model file exists, otherwise the prediction
panel just shows "no model loaded -- run train.py first".
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

# ---- config ---------------------------------------------------------------

SAMPLE_RATE = 1000          # must match the Arduino sketch
WINDOW_MS = 100             # feature window length
HOP_MS = 50                 # how often we compute a new feature vector
DISPLAY_SECONDS = 3         # how much history to show on screen
ADC_MAX = 1023              # 1023 for Uno, 4095 for Teensy 4.0
MODEL_PATH = 'model.pkl'

WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_MS / 1000)
HOP_SAMPLES = int(SAMPLE_RATE * HOP_MS / 1000)
DISPLAY_SAMPLES = SAMPLE_RATE * DISPLAY_SECONDS

# ---- shared buffers (written by serial thread, read by plot) -------------

emg_buffer = deque(maxlen=DISPLAY_SAMPLES)
audio_buffer = deque(maxlen=DISPLAY_SAMPLES)
buffer_lock = threading.Lock()
latest_prediction = {'label': 'starting...', 'confidence': 0.0}

# ---- feature extraction ---------------------------------------------------

def extract_features(emg_window, audio_window):
    """
    Compute a small feature vector from one window of EMG + audio.

    Features (4 total):
      - EMG RMS                : overall muscle activation level
      - EMG zero-crossing rate : rough proxy for signal frequency content
      - Audio RMS              : how loud the mic input is
      - Audio ZCR              : voiced vs. unvoiced / silent
    """
    emg = np.asarray(emg_window, dtype=np.float32)
    audio = np.asarray(audio_window, dtype=np.float32)

    # center each signal so RMS reflects AC content, not DC offset
    emg = emg - emg.mean()
    audio = audio - audio.mean()

    emg_rms = float(np.sqrt(np.mean(emg ** 2)))
    audio_rms = float(np.sqrt(np.mean(audio ** 2)))

    emg_zcr = float(np.mean(np.diff(np.sign(emg)) != 0))
    audio_zcr = float(np.mean(np.diff(np.sign(audio)) != 0))

    return np.array([emg_rms, emg_zcr, audio_rms, audio_zcr], dtype=np.float32)


# ---- serial reader thread -------------------------------------------------

def serial_reader(port, baud=500000):
    """Reads CSV lines from Arduino, pushes into the shared buffers."""
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2)  # let the Arduino reset
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
            audio_val = int(parts[1])
            with buffer_lock:
                emg_buffer.append(emg_val)
                audio_buffer.append(audio_val)
        except (ValueError, UnicodeDecodeError):
            continue


# ---- classifier thread ----------------------------------------------------

def classifier_loop():
    """Runs the trained model on the most recent window every HOP_MS."""
    global latest_prediction

    model = None
    if os.path.exists(MODEL_PATH):
        import pickle
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f'Loaded model from {MODEL_PATH}')
    else:
        latest_prediction = {'label': 'no model -- run train.py', 'confidence': 0.0}

    while True:
        time.sleep(HOP_MS / 1000)
        if model is None:
            continue
        with buffer_lock:
            if len(emg_buffer) < WINDOW_SAMPLES:
                continue
            emg_win = list(emg_buffer)[-WINDOW_SAMPLES:]
            audio_win = list(audio_buffer)[-WINDOW_SAMPLES:]

        feats = extract_features(emg_win, audio_win).reshape(1, -1)
        try:
            label = model.predict(feats)[0]
            if hasattr(model, 'predict_proba'):
                conf = float(np.max(model.predict_proba(feats)))
            else:
                conf = 1.0
            latest_prediction = {'label': str(label), 'confidence': conf}
        except Exception as e:
            latest_prediction = {'label': f'error: {e}', 'confidence': 0.0}


# ---- live plot ------------------------------------------------------------

def make_plot():
    fig, (ax_emg, ax_audio, ax_pred) = plt.subplots(
        3, 1, figsize=(10, 6),
        gridspec_kw={'height_ratios': [3, 3, 1]}
    )
    fig.suptitle('EMG + Audio Live Viewer', fontsize=12)

    line_emg, = ax_emg.plot([], [], lw=1)
    ax_emg.set_xlim(0, DISPLAY_SECONDS)
    ax_emg.set_ylim(0, ADC_MAX)
    ax_emg.set_ylabel('EMG (raw ADC)')
    ax_emg.grid(alpha=0.3)

    line_audio, = ax_audio.plot([], [], lw=1, color='tab:orange')
    ax_audio.set_xlim(0, DISPLAY_SECONDS)
    ax_audio.set_ylim(0, ADC_MAX)
    ax_audio.set_ylabel('Audio (raw ADC)')
    ax_audio.set_xlabel('time (s)')
    ax_audio.grid(alpha=0.3)

    pred_text = ax_pred.text(
        0.5, 0.5, '', ha='center', va='center',
        fontsize=20, transform=ax_pred.transAxes
    )
    ax_pred.axis('off')

    def update(_frame):
        with buffer_lock:
            emg = np.array(emg_buffer)
            audio = np.array(audio_buffer)

        if len(emg) > 0:
            t = np.arange(len(emg)) / SAMPLE_RATE
            line_emg.set_data(t, emg)
            ax_emg.set_xlim(max(0, t[-1] - DISPLAY_SECONDS), max(DISPLAY_SECONDS, t[-1]))
        if len(audio) > 0:
            t = np.arange(len(audio)) / SAMPLE_RATE
            line_audio.set_data(t, audio)
            ax_audio.set_xlim(max(0, t[-1] - DISPLAY_SECONDS), max(DISPLAY_SECONDS, t[-1]))

        pred_text.set_text(
            f"{latest_prediction['label']}   "
            f"({latest_prediction['confidence']*100:.0f}%)"
        )
        return line_emg, line_audio, pred_text

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


# ---- main -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', required=True, help='Serial port (e.g. COM3, /dev/ttyUSB0)')
    parser.add_argument('--baud', type=int, default=500000)
    args = parser.parse_args()

    t1 = threading.Thread(target=serial_reader, args=(args.port, args.baud), daemon=True)
    t2 = threading.Thread(target=classifier_loop, daemon=True)
    t1.start()
    t2.start()

    make_plot()


if __name__ == '__main__':
    main()
