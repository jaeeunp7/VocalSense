# VocalSense

# EMG + Audio Vocal Prototype

Three-week prototype for an EMG-based vocal state classifier.
Captures synchronized EMG (from a single-channel sensor board) and audio
(from a mic on a second analog pin), extracts simple features, and trains
a classifier to distinguish vocal states like silent / speaking / singing.

## Files

| File              | What it does                                              |
| ----------------- | --------------------------------------------------------- |
| emg_streamer.ino  | Arduino sketch -- streams ADC values over Serial          |
| record.py         | Records labeled training sessions to CSV                  |
| train.py          | Trains a classifier on the recorded CSVs                  |
| live_viewer.py    | Real-time display + live classification (the demo)        |

## Setup

```bash
pip install pyserial numpy pandas matplotlib scikit-learn
```

Flash `emg_streamer.ino` to your Arduino/Teensy. Open Serial Monitor at
500000 baud once to confirm you see CSV lines like `512,498`. Close the
Serial Monitor before running any Python script (only one process can
hold the port).

## Workflow

### 1. Hardware sanity check (week 1)

With nothing connected to A0/A1, run live_viewer:
```bash
python live_viewer.py --port COM3
```
You should see two flat-ish lines hovering near the ADC midpoint with
some noise. Now connect the EMG board to A0, place electrodes on your
forearm, and clench your fist -- the EMG trace should jump.

### 2. Record training data (week 2)

Edit the PROTOCOL list at the top of record.py to match the labels you
want to classify. Default is silent / speaking / singing_ah / humming.

```bash
python record.py --port COM3 --subject stephen
```
Run this for 2-3 subjects. Each session takes ~2 minutes.

### 3. Train (week 2-3)

```bash
python train.py --data data/
```
Look at the printed classification report and confusion_matrix.png.
If accuracy is below ~70%, check:
  - Are your classes actually different in the raw data? Plot a CSV.
  - Is electrode contact good? Skin prepped with alcohol?
  - Is the EMG board powered cleanly (battery, not USB)?

### 4. Live demo (week 3)

```bash
python live_viewer.py --port COM3
```
The bottom panel will show the predicted label updating in real time.

## Tuning knobs

- **SAMPLE_RATE** (in emg_streamer.ino, live_viewer.py, train.py): keep
  these in sync. 1 kHz works on Uno; bump to 2-4 kHz on Teensy.
- **WINDOW_MS / HOP_MS**: larger windows = more stable features but
  laggier response. 100 / 50 ms is a reasonable starting point.
- **PROTOCOL** (in record.py): the labels and durations you collect.
- **Features** (in `extract_features`): currently 4 features. Try adding
  variance, mean absolute value, or band-power if accuracy plateaus.

## Common gotchas

- Serial port name differs by OS: COM3 (Win), /dev/cu.usbmodemXXXX (Mac),
  /dev/ttyUSB0 or /dev/ttyACM0 (Linux).
- If readline() returns garbage, baud rates don't match between sketch
  and Python. Both should be 500000.
- If the live plot lags, lower the FuncAnimation interval or downsample
  the displayed buffer.
- ADC_MAX must match your board: 1023 for Uno, 4095 for Teensy 4.0.
