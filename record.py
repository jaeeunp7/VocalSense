"""
record.py

Records continuous singing takes with multi-label vocal-quality annotations.
Each take = one singing recording (default 30s) with a fixed set of label
values applied to the whole take. At training time, the raw 1 kHz data is
sliced into 2-second windows; each window inherits its take's labels.

Label categories (all binary, true=1 / false=0):
    - straining
    - vocal_resonance
    - phonation
    - breath_support

You set the label values interactively before each take. Sing for the full
duration trying to consistently demonstrate (or not demonstrate) the
qualities you marked.

Usage:
    python record.py --port COM3 --subject stephen
    python record.py --port COM3 --subject stephen --duration 60

CSV columns: time_s, emg, piezo, straining, vocal_resonance, phonation, breath_support
"""

import argparse
import csv
import os
import time
from datetime import datetime

import serial

LABEL_CATEGORIES = ['straining', 'vocal_resonance', 'phonation', 'breath_support']


def get_labels_interactive():
    """Prompt the user to mark which labels are TRUE for this take."""
    print('\n  Mark which qualities are TRUE for this take:')
    for i, label in enumerate(LABEL_CATEGORIES, 1):
        print(f'    {i}. {label}')
    print('  Enter numbers separated by commas (e.g. "2,3,4"), or "none".')
    while True:
        raw = input('  Selection: ').strip().lower()
        labels = {label: 0 for label in LABEL_CATEGORIES}
        if raw in ('none', ''):
            return labels
        try:
            nums = [int(x.strip()) for x in raw.split(',') if x.strip()]
            if not all(1 <= n <= len(LABEL_CATEGORIES) for n in nums):
                print(f'    Numbers must be 1-{len(LABEL_CATEGORIES)}. Try again.')
                continue
            for n in nums:
                labels[LABEL_CATEGORIES[n-1]] = 1
            return labels
        except ValueError:
            print('    Could not parse. Try again.')


def labels_summary(labels):
    true_labels = [k for k, v in labels.items() if v]
    return ', '.join(true_labels) if true_labels else 'all false'


def countdown(seconds):
    print(f'\n  About to record {seconds}s of singing.')
    for i in range(3, 0, -1):
        print(f'    starting in {i}...')
        time.sleep(1)
    print('    GO -- start singing\n')


def record_take(ser, duration_s, labels, writer):
    end_time = time.time() + duration_s
    t0 = time.time()
    samples_collected = 0
    label_row = [labels[k] for k in LABEL_CATEGORIES]
    while time.time() < end_time:
        line = ser.readline().decode('ascii', errors='ignore').strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) != 2:
            continue
        try:
            emg = int(parts[0])
            piezo = int(parts[1])
        except ValueError:
            continue
        t = time.time() - t0
        writer.writerow([f'{t:.4f}', emg, piezo, *label_row])
        samples_collected += 1
    return samples_collected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', required=True)
    parser.add_argument('--baud', type=int, default=500000)
    parser.add_argument('--subject', required=True, help='Subject ID, e.g. stephen')
    parser.add_argument('--out', default='data', help='Output directory')
    parser.add_argument('--duration', type=int, default=30,
                        help='Length of each take in seconds (default 30)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f'Opening serial port {args.port} at {args.baud} baud...')
    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    print('Connected.\n')

    print(f'Subject:        {args.subject}')
    print(f'Take duration:  {args.duration}s  (--> ~{args.duration // 2} two-second windows per take)')
    print(f'Output folder:  {args.out}/')
    print(f'Categories:     {", ".join(LABEL_CATEGORIES)}')

    take_idx = 1
    while True:
        print(f'\n{"="*54}')
        print(f'TAKE {take_idx}')
        print('='*54)

        labels = get_labels_interactive()
        print(f'  Labels for this take: {labels_summary(labels)}')

        confirm = input('  Record this take? (y/n/q to quit session): ').strip().lower()
        if confirm == 'q':
            break
        if confirm != 'y':
            print('  Skipped -- re-entering labels.')
            continue

        # Filename pattern: subject_timestamp_takeNN_SRPB.csv
        # where SRPB is 4 binary digits: Straining, Resonance, Phonation, Breath
        label_code = ''.join(str(labels[k]) for k in LABEL_CATEGORIES)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(
            args.out,
            f'{args.subject}_{timestamp}_take{take_idx:02d}_{label_code}.csv'
        )

        input(f'  Press ENTER when ready (you will sing for {args.duration}s)...')
        countdown(args.duration)

        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_s', 'emg', 'piezo', *LABEL_CATEGORIES])
            n = record_take(ser, args.duration, labels, writer)

        print(f'  Take {take_idx} done -- {n} samples')
        print(f'  Saved to {out_path}')
        take_idx += 1

        another = input('\n  Record another take? (y/n): ').strip().lower()
        if another != 'y':
            break

    print(f'\nSession complete. {take_idx - 1} take(s) saved to {args.out}/')
    ser.close()


if __name__ == '__main__':
    main()