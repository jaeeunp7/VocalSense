"""
app.py - VocalSense Analysis Dashboard

Upload pre-recorded CSV files, analyze them with your trained model,
and get AI coaching feedback from Groq (free tier, very fast).

No Arduino or live connection needed -- this is a post-recording
analysis tool.

Usage:
    python app.py
    python app.py --api-key gsk_...

Then open http://127.0.0.1:5000 in your browser.

Getting a free Groq API key:
    1. Go to console.groq.com
    2. Sign in with a Google account
    3. Click "API Keys" -> "Create API key"
    4. Copy the key (starts with gsk_...)
    Free tier: 14,400 requests/day, no credit card needed.

Dependencies:
    pip install flask numpy pandas scikit-learn groq
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# ---- config ---------------------------------------------------------------

SAMPLE_RATE = 1000
WINDOW_MS = 2000
HOP_MS = 1000
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_MS / 1000)
HOP_SAMPLES = int(SAMPLE_RATE * HOP_MS / 1000)
LABEL_CATEGORIES = ['straining', 'vocal_resonance', 'phonation', 'breath_support']
MODEL_PATH = 'model.pkl'
DATA_DIR = 'data'
DISPLAY_DOWNSAMPLE = 20     # show every Nth sample in browser charts

# ---- load model -----------------------------------------------------------

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f'Loaded model from {MODEL_PATH}')
else:
    print(f'WARNING: No model found at {MODEL_PATH}')
    print('  Run train.py first, then restart app.py.\n')

# ---- feature extraction (MUST match train.py) -----------------------------

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

    return [emg_rms, emg_mav, emg_zcr, emg_wl, piezo_rms, piezo_peak, piezo_dom_freq]


# ---- CSV analysis ---------------------------------------------------------

def analyze_csv(df):
    """Process a CSV through the model and return full analysis."""
    if model is None:
        return {'error': 'No model loaded. Run train.py first.'}

    emg = df['emg'].values
    piezo = df['piezo'].values
    time_s = df['time_s'].values if 'time_s' in df.columns else np.arange(len(emg)) / SAMPLE_RATE

    # ground truth labels if present in the CSV
    has_ground_truth = all(c in df.columns for c in LABEL_CATEGORIES)

    # downsample signals for display
    step = max(1, DISPLAY_DOWNSAMPLE)
    signals = {
        'time': time_s[::step].tolist(),
        'emg': emg[::step].tolist(),
        'piezo': piezo[::step].tolist(),
    }

    # slide 2-second windows and predict each one
    windows = []
    for start in range(0, len(emg) - WINDOW_SAMPLES + 1, HOP_SAMPLES):
        end = start + WINDOW_SAMPLES
        feats = extract_features(emg[start:end], piezo[start:end])
        feats_arr = np.array(feats).reshape(1, -1)

        preds = model.predict(feats_arr)[0]
        confs = [0.5] * len(LABEL_CATEGORIES)
        if hasattr(model, 'predict_proba'):
            try:
                probas = model.predict_proba(feats_arr)
                for i, p in enumerate(probas):
                    confs[i] = float(p[0].max())
            except Exception:
                pass

        window_data = {
            'start_s': round(float(time_s[start]), 2),
            'end_s': round(float(time_s[min(end, len(time_s)-1)]), 2),
            'predictions': {},
            'features': {
                'emg_rms': round(feats[0], 2),
                'emg_mav': round(feats[1], 2),
                'emg_zcr': round(feats[2], 4),
                'emg_wl': round(feats[3], 2),
                'piezo_rms': round(feats[4], 2),
                'piezo_peak': round(feats[5], 2),
                'piezo_dom_freq': round(feats[6], 1),
            }
        }
        for i, label in enumerate(LABEL_CATEGORIES):
            window_data['predictions'][label] = {
                'value': int(preds[i]),
                'confidence': round(confs[i], 3),
            }

        # include ground truth if available
        if has_ground_truth:
            window_data['ground_truth'] = {}
            for label in LABEL_CATEGORIES:
                gt_val = int(df[label].iloc[start])
                window_data['ground_truth'][label] = gt_val

        windows.append(window_data)

    # aggregate stats
    n = len(windows)
    aggregate = {
        'total_windows': n,
        'duration_s': round(float(time_s[-1]) if len(time_s) > 0 else 0, 1),
    }
    for label in LABEL_CATEGORIES:
        true_count = sum(1 for w in windows if w['predictions'][label]['value'] == 1)
        aggregate[f'{label}_pct'] = round(true_count / n * 100, 1) if n > 0 else 0
        avg_conf = np.mean([w['predictions'][label]['confidence'] for w in windows])
        aggregate[f'{label}_avg_conf'] = round(float(avg_conf) * 100, 1) if n > 0 else 0

    # feature importances from the model
    feature_importances = []
    try:
        clf = model.named_steps.get('clf')
        if clf and hasattr(clf, 'feature_importances_'):
            names = ['emg_rms', 'emg_mav', 'emg_zcr', 'emg_wl',
                     'piezo_rms', 'piezo_peak', 'piezo_dom_freq']
            for name, imp in sorted(zip(names, clf.feature_importances_),
                                     key=lambda x: -x[1]):
                feature_importances.append({
                    'name': name, 'importance': round(float(imp), 3)
                })
    except Exception:
        pass

    return {
        'signals': signals,
        'windows': windows,
        'aggregate': aggregate,
        'feature_importances': feature_importances,
        'has_ground_truth': has_ground_truth,
        'filename': '',
    }


# ---- Groq coaching --------------------------------------------------------

def get_coaching(api_key, analysis):
    """Send analysis results to Groq for coaching feedback."""
    try:
        from groq import Groq
    except ImportError:
        return {'message': 'Install the groq package:\n  pip install groq',
                'error': True}

    if not api_key:
        return {
            'message': (
                'Enter your Groq API key to enable coaching.\n'
                'Get a free key at console.groq.com → "API Keys".\n'
                'Free tier: 14,400 requests/day, no credit card needed.'    
            ),
            'error': True
        }

    agg = analysis.get('aggregate', {})
    windows = analysis.get('windows', [])
    n = agg.get('total_windows', 0)
    dur = agg.get('duration_s', 0)

    # Build a timeline summary for Claude
    # Group into thirds: beginning, middle, end
    timeline_summary = ''
    if n >= 6:
        third = n // 3
        sections = [
            ('Beginning', windows[:third]),
            ('Middle', windows[third:2*third]),
            ('End', windows[2*third:]),
        ]
        parts = []
        for section_name, section_windows in sections:
            stats = {}
            for label in LABEL_CATEGORIES:
                pct = sum(1 for w in section_windows
                          if w['predictions'][label]['value'] == 1) / len(section_windows) * 100
                stats[label] = f'{pct:.0f}%'
            parts.append(f"  {section_name}: " + ', '.join(
                f'{k.replace("_"," ")}={v}' for k, v in stats.items()))
        timeline_summary = '\n'.join(parts)

    # Feature importance context
    feat_summary = ''
    fi = analysis.get('feature_importances', [])
    if fi:
        top3 = fi[:3]
        feat_summary = 'Top features the model relies on: ' + ', '.join(
            f'{f["name"]} ({f["importance"]*100:.0f}%)' for f in top3)

    prompt = f"""You are VocalSense AI, a vocal coach analyzing biosensor data from a
singing session recorded with an EMG sensor (under the chin, measuring muscle
activation) and a piezo vibration sensor (on the throat, measuring vocal fold
vibration). A trained machine learning classifier has analyzed {n} two-second
windows across {dur} seconds of singing.

Overall results:
- Straining detected: {agg.get('straining_pct', 0)}% of the time (avg confidence: {agg.get('straining_avg_conf', 0)}%)
- Vocal resonance detected: {agg.get('vocal_resonance_pct', 0)}% of the time (avg confidence: {agg.get('vocal_resonance_avg_conf', 0)}%)
- Good phonation detected: {agg.get('phonation_pct', 0)}% of the time (avg confidence: {agg.get('phonation_avg_conf', 0)}%)
- Breath support detected: {agg.get('breath_support_pct', 0)}% of the time (avg confidence: {agg.get('breath_support_avg_conf', 0)}%)

{f'Timeline breakdown:{chr(10)}{timeline_summary}' if timeline_summary else ''}

{feat_summary}

Based on this analysis, provide:
1. A brief overall assessment (2-3 sentences) of the singer's technique in this recording.
2. The most important area for improvement, with 1-2 specific physical cues the singer can try (e.g. "drop your jaw", "breathe from your belly", "lift your soft palate").
3. Something the singer is doing well that they should continue.

Be specific, encouraging, and actionable. Use personal pronouns like you and your. Do not repeat the raw numbers back --
interpret them as a knowledgeable vocal coach would."""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=500,
        )
        return {'message': response.choices[0].message.content, 'error': False}
    except Exception as e:
        return {'message': f'Groq API error: {str(e)}', 'error': True}


# ---- Flask app ------------------------------------------------------------

app = Flask(__name__)
api_key_store = {'key': None}  # set via --api-key flag, env var, or browser input
last_analysis = {}


@app.route('/')
def dashboard():
    return render_template('dashboard.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Upload and analyze a CSV file."""
    global last_analysis

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a .csv file'}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Could not parse CSV: {str(e)}'}), 400

    required = ['emg', 'piezo']
    missing = [c for c in required if c not in df.columns]
    if missing:
        return jsonify({'error': f'CSV missing required columns: {missing}'}), 400

    if len(df) < WINDOW_SAMPLES:
        return jsonify({
            'error': f'Recording too short. Need at least {WINDOW_MS/1000}s '
                     f'({WINDOW_SAMPLES} samples). Got {len(df)} samples.'
        }), 400

    result = analyze_csv(df)
    result['filename'] = file.filename
    last_analysis = result
    return jsonify(result)


@app.route('/coaching', methods=['POST'])
def coaching():
    """Get Claude coaching feedback on the last analyzed recording."""
    key = api_key_store.get('key') or os.environ.get('GROQ_API_KEY', '')

    if not last_analysis or 'aggregate' not in last_analysis:
        return jsonify({
            'message': 'Upload and analyze a CSV file first.',
            'error': True
        })

    result = get_coaching(key, last_analysis)
    return jsonify(result)


@app.route('/set-api-key', methods=['POST'])
def set_api_key():
    data = request.get_json()
    api_key_store['key'] = data.get('key', '')
    return jsonify({'ok': True})


@app.route('/list-csvs')
def list_csvs():
    """List available CSVs in the data/ folder."""
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    files = []
    for path in csv_files:
        try:
            size = os.path.getsize(path)
            files.append({
                'name': os.path.basename(path),
                'path': path,
                'size_kb': round(size / 1024, 1),
            })
        except Exception:
            continue
    return jsonify({'files': files})


@app.route('/analyze-existing', methods=['POST'])
def analyze_existing():
    """Analyze a CSV from the data/ folder by name."""
    global last_analysis
    data = request.get_json()
    filename = data.get('filename', '')
    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        return jsonify({'error': f'File not found: {filename}'}), 404

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return jsonify({'error': f'Could not parse CSV: {str(e)}'}), 400

    required = ['emg', 'piezo']
    missing = [c for c in required if c not in df.columns]
    if missing:
        return jsonify({'error': f'CSV missing required columns: {missing}'}), 400

    if len(df) < WINDOW_SAMPLES:
        return jsonify({
            'error': f'Recording too short ({len(df)} samples).'
        }), 400

    result = analyze_csv(df)
    result['filename'] = filename
    last_analysis = result
    return jsonify(result)


# ---- main -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='VocalSense Analysis Dashboard')
    parser.add_argument('--api-key', default=None,
                        help='Groq API key (or set GROQ_API_KEY env var)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    if args.api_key:
        api_key_store['key'] = args.api_key

     # Render sets PORT automatically — read it, fall back to 5000 locally
    port = int(os.environ.get('PORT', 5000))
    
    print(f'\n  VocalSense Dashboard: http://{args.host}:{args.port}\n')
    if not model:
        print('  WARNING: No model.pkl found. Run train.py first.\n')
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
