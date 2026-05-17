# EMG + Piezo Vocal Prototype

Three-week prototype for a multi-sensor vocal state classifier.
Captures synchronized EMG (sub-mental muscle activity) and piezo (throat
vibration) at 1 kHz, extracts simple features, and trains a classifier
to distinguish vocal states like silent / speaking / singing.

## Files

| File                    | What it does                                          |
| ----------------------- | ----------------------------------------------------- |
| emg_piezo_streamer.ino  | Arduino sketch -- streams ADC values over Serial      |
| record.py               | Records labeled training sessions to CSV              |
| train.py                | Trains a classifier on the recorded CSVs              |
| live_viewer.py          | Real-time display + live classification (the demo)    |

## Hardware

- Arduino Uno R3 (Elegoo)
- EMG sensor board on A0
- Piezo (LDT0-028K) protection circuit on A1:
  - 1 MΩ bleed resistor across the piezo leads
  - 1 μF AC-coupling capacitor in series with piezo (+)
  - Two 100 kΩ voltage-divider resistors biasing A1 to ~2.5 V

Shared 5V and GND rails on the breadboard power both sensors.

## Setup

```bash
pip install pyserial numpy pandas matplotlib scikit-learn
```

Flash `emg_piezo_streamer.ino` to your Arduino. Open Serial Monitor at
500000 baud once to confirm you see CSV lines like `512,498`. Close
the Serial Monitor before running any Python script (only one process
can hold the port at a time).

## Workflow

### 1. Hardware sanity check (week 1)

With no electrodes on yourself and the piezo loose on the desk, run:
```bash
python live_viewer.py --port COM3
```
Both panels should show flat-ish traces near ADC midpoint (~512).
Tap the piezo with a finger -- the piezo trace should spike clearly.
Place EMG electrodes on your forearm, clench your fist -- the EMG
trace should jump. If both pass, hardware is ready.

### 2. Record training data (week 2)

Move EMG electrodes to sub-mental placement (under the chin) and
tape the piezo to the front of your throat over the larynx. Then:

```bash
python record.py --port COM3 --subject stephen
```

Edit the PROTOCOL list at the top of record.py to change the labels
or durations. Default is silent / speaking / singing_ah / humming.

Run for 2-3 subjects. Each session takes ~2 minutes.

### 3. Train (week 2-3)

```bash
python train.py --data data/
```
Look at the printed classification report and confusion_matrix.png.
If accuracy is below ~70%, check:
  - Are your classes actually different in the raw data? Plot a CSV.
  - Is electrode contact good? Skin prepped with alcohol?
  - Is the piezo taped firmly but not squashed against the throat?
  - Is the EMG board powered cleanly (battery rather than USB)?

### 4. Live demo (week 3)

```bash
python live_viewer.py --port COM3
```
The bottom panel shows the predicted label updating in real time.

## Features extracted

Four features per window (100 ms):
- EMG RMS                 (muscle activation level)
- EMG zero-crossing rate  (rough frequency content)
- Piezo RMS               (vibration energy)
- Piezo peak amplitude    (max minus min, captures spikes)

To extend, edit `extract_features` in BOTH train.py and live_viewer.py
-- they must stay identical or the model breaks at inference time.

Useful additions later:
- Piezo dominant frequency via FFT (captures vocal fundamental)
- EMG mean absolute value
- EMG waveform length (Jou's TD5 feature set)

## Tuning knobs

- **SAMPLE_RATE**: keep in sync across .ino, live_viewer, train.
  1 kHz works on Uno; bump to 2-4 kHz on Teensy.
- **WINDOW_MS / HOP_MS**: larger windows = more stable features but
  laggier response. 100 / 50 ms is a fine starting point.
- **PROTOCOL** (in record.py): the labels and durations you collect.
- **ADC_MAX**: 1023 for Uno, 4095 for Teensy 4.0.

## Common gotchas

- Serial port name differs by OS: COM3 (Win), /dev/cu.usbmodemXXXX
  (Mac), /dev/ttyUSB0 or /dev/ttyACM0 (Linux).
- If readline() returns garbage, baud rates don't match between
  sketch and Python. Both should be 500000.
- If the piezo trace is pegged at 0 or 1023, the voltage divider isn't
  biasing correctly -- check the two 100 kΩ resistors.
- If the EMG trace is flat at 0 or 1023, an electrode isn't making
  contact or the +VS/GND wiring is reversed on the EMG board.
- The piezo CAN spike high enough to damage A1 if you bend the film
  hard with no bleed resistor in place -- always wire the 1 MΩ first.
