# VocalSense – Singing Technique Analyzer

An end-to-end hardware-software prototype built with Arduino and scikit-learn 
that uses sEMG and piezoelectric biosensors to analyze a singer's vocal 
technique in real time. The system classifies technique across 5+ categories 
such as straining, breath support, and phonation, and delivers personalized 
AI-generated feedback through a connected web interface.

---

## How It Works

VocalSense runs as a five-stage pipeline:

1. **Sensor Capture** — An EMG sensor (sub-mental placement) and a piezo 
sensor (taped over the larynx) capture synchronized muscle activation and 
throat vibration data at 1 kHz via an Arduino Uno.

2. **Signal Streaming** — The Arduino streams raw ADC values over USB Serial 
to a connected computer in real time.

3. **Feature Extraction & Classification** — A Python script reads the serial 
stream, segments it into 100 ms windows, and extracts four features per window 
(EMG RMS, EMG zero-crossing rate, Piezo RMS, Piezo peak amplitude). These 
features are passed into a trained scikit-learn classifier that predicts the 
current vocal state (e.g. silent, speaking, singing, straining).

4. **AI Feedback Generation** — The predicted label is sent to the Grok API, 
which generates a personalized, human-readable coaching response based on the 
detected vocal technique.

5. **Web Interface** — The AI-generated feedback is displayed in real time on 
a connected web interface, giving the singer immediate insight into their 
technique without needing a vocal coach present.

---

## Files

| File                   | What it does                                        |
| ---------------------- | --------------------------------------------------- |
| emg_piezo_streamer.ino | Arduino sketch — streams ADC values over Serial     |
| record.py              | Records labeled training sessions to CSV            |
| train.py               | Trains a classifier on the recorded CSVs            |
| live_viewer.py         | Real-time display + live classification (the demo)  |

---

## Hardware

- Arduino Uno R3 (Elegoo)
- EMG sensor board on A0
- Piezo (LDT0-028K) protection circuit on A1:
  - 1 MΩ bleed resistor across the piezo leads
  - 1 μF AC-coupling capacitor in series with piezo (+)
  - Two 100 kΩ voltage-divider resistors biasing A1 to ~2.5 V

Shared 5V and GND rails on the breadboard power both sensors.

---

## Setup

```bash
pip install pyserial numpy pandas matplotlib scikit-learn
```

Flash `emg_piezo_streamer.ino` to your Arduino. Open Serial Monitor at 
500000 baud once to confirm you see CSV lines like `512,498`. Close the 
Serial Monitor before running any Python script (only one process can hold 
the port at a time).

---

## Workflow

### 1. Hardware Sanity Check (Week 1)
With no electrodes on yourself and the piezo loose on the desk, run:
```bash
python live_viewer.py --port COM3
```
Both panels should show flat-ish traces near ADC midpoint (~512).  
Tap the piezo — the piezo trace should spike. Clench your fist with EMG 
electrodes on your forearm — the EMG trace should jump. If both pass, 
hardware is ready.

### 2. Record Training Data (Week 2)
Move EMG electrodes to sub-mental placement (under the chin) and tape the 
piezo to the front of your throat over the larynx. Then:
```bash
python record.py --port COM3 --subject stephen
```
Edit the `PROTOCOL` list in `record.py` to change labels or durations. 
Default is `silent / speaking / singing_ah / humming`. Run for 2–3 subjects 
(~2 minutes per session).

### 3. Train the Classifier (Week 2–3)
```bash
python train.py --data data/
```
Check the printed classification report and `confusion_matrix.png`. If 
accuracy is below ~70%, verify electrode contact, sensor placement, and 
that raw data between classes looks visually different.

### 4. Live Demo (Week 3)
```bash
python live_viewer.py --port COM3
```
The bottom panel shows the predicted vocal state updating in real time. 
The predicted label is simultaneously sent to the Grok API, and the 
AI-generated feedback response appears on the web interface.

---

## Features Extracted

Four features per 100 ms window:

| Feature               | Description                        |
| --------------------- | ---------------------------------- |
| EMG RMS               | Muscle activation level            |
| EMG Zero-Crossing Rate| Rough frequency content            |
| Piezo RMS             | Vibration energy                   |
| Piezo Peak Amplitude  | Max minus min — captures spikes    |

---

## Common Gotchas

- Serial port name differs by OS: `COM3` (Windows), `/dev/cu.usbmodemXXXX` 
(Mac), `/dev/ttyUSB0` (Linux)
- If `readline()` returns garbage, baud rates don't match — both sketch and 
Python should be `500000`
- If the piezo trace is pegged at 0 or 1023, the voltage divider isn't 
biasing correctly — check the two 100 kΩ resistors
- The piezo can spike high enough to damage A1 if bent hard with no bleed 
resistor — always wire the 1 MΩ first
