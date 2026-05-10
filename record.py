"""
record.py

Records labeled EMG + audio sessions to a CSV file for later training.
Prompts you to perform each label for a fixed duration, with a countdown.

Usage:
    python record.py --port COM3 --subject stephen --out data/

Each session writes one CSV: data/<subject>_<timestamp>.csv
Columns: time_s, emg, audio, label
"""

import argparse
import csv
import os
import time
from datetime import datetime

import serial

# ---- config ---------------------------------------------------------------

# Edit these to match your protocol. Each (label, duration_seconds) tuple
# is one block of recording. Add/remove freely.
PROTOCOL = [
    ('silent',   30),
    ('speaking', 30),
    ('singing_ah', 30),
    ('humming',  30),
    ('silent',   15),     # second silent block as a sanity check
]

SAMPLE_RATE = 1000


def countdown(seconds, label):
    print(f'\n>>> Next: "{label}" for {seconds}s')
    for i in range(3, 0, -1):
        print(f'    starting in {i}...')
        time.sleep(1)
    print(f'    GO -- "{label}"')


def record_block(ser, label, duration_s, writer, t0):
    end_time = time.time() + duration_s
    samples_collected = 0
    while time.time() < end_time:
        line = ser.readline().decode('ascii', errors='ignore').strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) != 2:
            continue
        try:
            emg = int(parts[0])
            audio = int(parts[1])
        except ValueError:
            continue
        t = time.time() - t0
        writer.writerow([f'{t:.4f}', emg, audio, label])
        samples_collected += 1
    print(f'    done -- {samples_collected} samples')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', required=True)
    parser.add_argument('--baud', type=int, default=500000)
    parser.add_argument('--subject', required=True, help='Subject ID, e.g. stephen')
    parser.add_argument('--out', default='data', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(args.out, f'{args.subject}_{timestamp}.csv')

    print(f'Opening serial port {args.port} at {args.baud} baud...')
    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    print('Connected.')

    print(f'\nWriting to: {out_path}')
    print(f'Protocol: {len(PROTOCOL)} blocks, '
          f'{sum(d for _, d in PROTOCOL)}s total\n')
    input('Press ENTER when subject is ready and electrodes are placed...')

    t0 = time.time()
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s', 'emg', 'audio', 'label'])
        for label, duration in PROTOCOL:
            countdown(duration, label)
            record_block(ser, label, duration, writer, t0)

    print(f'\nDone. Saved to {out_path}')
    ser.close()


if __name__ == '__main__':
    main()
