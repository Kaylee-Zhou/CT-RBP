import os
import sys
import logging
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, send_file, flash
import pandas as pd
import uuid
import traceback

from model_utils import load_model_and_vocab, predict_dataframe
from rag_utils import rag_explain

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limitation: 16MB

# Set up logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
MODEL_PATH = 'HNRNPC/Output/model.pth'
VOCAB_PATH = 'HNRNPC/Output/vocab.pkl'

model, kmer_vocab = load_model_and_vocab(MODEL_PATH, VOCAB_PATH)

# Temporary Directory
TEMP_DIR = '/tmp/rna_predictor'
os.makedirs(TEMP_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/download_sample')
def download_sample():
    sample_path = os.path.join('static', 'dataset_sample', 'test.tsv')
    if os.path.exists(sample_path):
        return send_file(sample_path, as_attachment=True, download_name='test.tsv')
    else:
        flash('The sample file does not exist. Please contact the administrator.', 'danger')
        return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            flash('No file selected', 'danger')
            return redirect(url_for('index'))
        file = request.files['file']
        if file.filename == '':
            flash('The file name is empty.', 'danger')
            return redirect(url_for('index'))

        df = pd.read_csv(file, sep='\t')
        required_cols = ['sequence', 'structure', 'pairing_probabilities']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            flash(f'The uploaded file is missing the necessary columns: {missing}', 'danger')
            return redirect(url_for('index'))

        probs = predict_dataframe(df, model, kmer_vocab)
        pred_labels = (probs > 0.5).astype(int)

        df['predicted_probability'] = probs
        df['predicted_label'] = pred_labels

        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        temp_path = os.path.join(TEMP_DIR, f'{session_id}.csv')
        df.to_csv(temp_path, sep='\t', index=False)

        return redirect(url_for('result', session_id=session_id))

    except Exception as e:
        logger.error(f"Prediction failure: {traceback.format_exc()}")
        flash(f'Errors occurred during the prediction process: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/result/<session_id>')
def result(session_id):
    temp_path = os.path.join(TEMP_DIR, f'{session_id}.csv')
    if not os.path.exists(temp_path):
        flash('The conversation has expired or is invalid.', 'warning')
        return redirect(url_for('index'))
    df = pd.read_csv(temp_path, sep='\t')
    # Add an index column (starting from 1)
    df.insert(0, 'index', range(1, len(df)+1))
    display_cols = ['index'] + ['sequence', 'structure', 'pairing_probabilities', 'label', 'predicted_probability', 'predicted_label']
    display_cols = [c for c in display_cols if c in df.columns]
    df_display = df[display_cols].copy()
    if 'sequence' in df_display.columns:
        df_display['sequence'] = df_display['sequence'].apply(
            lambda x: (x[:30] + '...') if isinstance(x, str) and len(x) > 30 else x
        )
    table_html = df_display.to_html(classes='table table-striped table-hover table-sm', index=False, escape=False)
    return render_template('result.html', table=table_html, session_id=session_id, num_rows=len(df))

@app.route('/download_result/<session_id>')
def download_result(session_id):
    temp_path = os.path.join(TEMP_DIR, f'{session_id}.csv')
    if not os.path.exists(temp_path):
        flash('The file does not exist or has expired.', 'warning')
        return redirect(url_for('index'))
    return send_file(temp_path, as_attachment=True, download_name='predictions.tsv')

@app.route('/explain_predictions', methods=['POST'])
def explain_predictions():
    try:
        if 'file' not in request.files:
            flash('No file selected', 'danger')
            return redirect(url_for('index'))
        file = request.files['file']
        if file.filename == '':
            flash('The file name is empty.', 'danger')
            return redirect(url_for('index'))

        df = pd.read_csv(file, sep='\t')
        required_cols = ['sequence', 'predicted_probability', 'predicted_label']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            flash(f'The explanation file is missing the necessary columns: {missing}', 'danger')
            return redirect(url_for('index'))

        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        temp_path = os.path.join(TEMP_DIR, f'{session_id}.csv')
        df.to_csv(temp_path, sep='\t', index=False)

        return redirect(url_for('result', session_id=session_id))

    except Exception as e:
        logger.error(f"Explanation: File processing failed: {traceback.format_exc()}")
        flash(f'Error occurred while processing the explanatory file: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/explain/<session_id>/<int:row_idx>')
def explain_page(session_id, row_idx):
    return render_template('explain_loading.html', session_id=session_id, row_idx=row_idx)

@app.route('/api/explain/<session_id>/<int:row_idx>')
def api_explain(session_id, row_idx):
    try:
        temp_path = os.path.join(TEMP_DIR, f'{session_id}.csv')
        if not os.path.exists(temp_path):
            return jsonify({'status': 'failed', 'error': 'The conversation has expired.'}), 404
        df = pd.read_csv(temp_path, sep='\t')
        if row_idx < 0 or row_idx >= len(df):
            return jsonify({'status': 'failed', 'error': 'Invalid index operation'}), 404

        row = df.iloc[row_idx]
        sequence = row['sequence']
        true_label = row.get('label', -1)
        predicted_label = int(row['predicted_label'])
        predicted_prob = float(row['predicted_probability'])

        result = rag_explain(sequence, predicted_label, predicted_prob)
        return jsonify(result)

    except Exception as e:
        logger.error(f"API explanation failed: {traceback.format_exc()}")
        return jsonify({'status': 'failed', 'error': str(e)}), 500

@app.route('/explain/result/<session_id>/<int:row_idx>')
def explain_result(session_id, row_idx):
    return render_template('explain_result.html', session_id=session_id, row_idx=row_idx)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)