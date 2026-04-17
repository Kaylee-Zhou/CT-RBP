import os
import sys
import logging
import io
import base64

from flask import Flask, render_template, request, session, redirect, url_for, jsonify, send_file, flash
import pandas as pd
import uuid
import traceback

from model_utils import load_model_and_vocab, predict_dataframe
from rag_utils import rag_explain

app = Flask(__name__)
app.secret_key = '202218020308'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = 'HNRNPC/Output/model.pth'
VOCAB_PATH = 'HNRNPC/Output/vocab.pkl'

model, kmer_vocab = load_model_and_vocab(MODEL_PATH, VOCAB_PATH)

TEMP_DIR = '/tmp/rna_predictor'
os.makedirs(TEMP_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download_sample')
def download_sample():
    sample_path = os.path.join('static', 'dataset_sample', 'test.tsv')
    if os.path.exists(sample_path):
        return send_file(sample_path, as_attachment=True, download_name='test.tsv')
    else:
        return jsonify({'status': 'failed', 'error': 'Sample file not found'}), 404

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'failed', 'error': 'No file selected'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'failed', 'error': 'File name is empty'}), 400

        df = pd.read_csv(file, sep='\t')
        required_cols = ['sequence', 'structure', 'pairing_probabilities']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return jsonify({'status': 'failed', 'error': f'Missing columns: {missing}'}), 400

        probs = predict_dataframe(df, model, kmer_vocab)
        pred_labels = (probs > 0.5).astype(int)

        df['predicted_probability'] = probs
        df['predicted_label'] = pred_labels

        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        temp_path = os.path.join(TEMP_DIR, f'{session_id}.csv')
        df.to_csv(temp_path, sep='\t', index=False)

        # Generate table HTML
        df_display = df.copy()
        df_display.insert(0, 'index', range(1, len(df_display)+1))
        display_cols = ['index'] + ['sequence', 'structure', 'pairing_probabilities', 'label', 'predicted_probability', 'predicted_label']
        display_cols = [c for c in display_cols if c in df_display.columns]
        df_display = df_display[display_cols].copy()
        if 'sequence' in df_display.columns:
            df_display['sequence'] = df_display['sequence'].apply(
                lambda x: (x[:30] + '...') if isinstance(x, str) and len(x) > 30 else x
            )
        table_html = df_display.to_html(classes='table table-striped table-hover table-sm', index=False, escape=False)

        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'table_html': table_html,
            'num_rows': len(df)
        })
    except Exception as e:
        logger.error(f"Prediction error: {traceback.format_exc()}")
        return jsonify({'status': 'failed', 'error': str(e)}), 500

@app.route('/download_result/<session_id>')
def download_result(session_id):
    temp_path = os.path.join(TEMP_DIR, f'{session_id}.csv')
    if not os.path.exists(temp_path):
        return jsonify({'status': 'failed', 'error': 'File not found'}), 404
    return send_file(temp_path, as_attachment=True, download_name='predictions.tsv')

@app.route('/generate_shap/<session_id>', methods=['POST'])
def generate_shap(session_id):
    temp_path = os.path.join(TEMP_DIR, f'{session_id}.csv')
    if not os.path.exists(temp_path):
        return jsonify({'status': 'failed', 'error': 'Session expired'}), 404
    df = pd.read_csv(temp_path, sep='\t')
    if 'predicted_probability' not in df.columns:
        return jsonify({'status': 'failed', 'error': 'No predictions found, please run prediction first'}), 400
    predictions = df['predicted_probability'].values
    try:
        from shap_utils import generate_shap_images_base64
        images = generate_shap_images_base64(df, kmer_vocab, predictions)
        return jsonify({'status': 'success', 'images': images})
    except Exception as e:
        logger.error(f"SHAP generation failed: {traceback.format_exc()}")
        return jsonify({'status': 'failed', 'error': str(e)}), 500
    
@app.route('/download_shap/<session_id>', methods=['POST'])
def download_shap(session_id):
    temp_path = os.path.join(TEMP_DIR, f'{session_id}.csv')
    if not os.path.exists(temp_path):
        return jsonify({'status': 'failed', 'error': 'Session expired'}), 404
    df = pd.read_csv(temp_path, sep='\t')
    if 'predicted_probability' not in df.columns:
        return jsonify({'status': 'failed', 'error': 'No predictions found'}), 400
    predictions = df['predicted_probability'].values
    try:
        from shap_utils import generate_shap_zip
        zip_data = generate_shap_zip(df, kmer_vocab, predictions)
        return send_file(
            io.BytesIO(zip_data),
            mimetype='application/zip',
            as_attachment=True,
            download_name='shap_analysis.zip'
        )
    except Exception as e:
        logger.error(f"SHAP Generation failed: {traceback.format_exc()}")
        return jsonify({'status': 'failed', 'error': str(e)}), 500

@app.route('/api/explain/<session_id>/<int:row_idx>')
def api_explain(session_id, row_idx):
    try:
        temp_path = os.path.join(TEMP_DIR, f'{session_id}.csv')
        if not os.path.exists(temp_path):
            return jsonify({'status': 'failed', 'error': 'Session expired'}), 404
        df = pd.read_csv(temp_path, sep='\t')
        if row_idx < 0 or row_idx >= len(df):
            return jsonify({'status': 'failed', 'error': 'Invalid index'}), 404

        row = df.iloc[row_idx]
        sequence = row['sequence']
        predicted_label = int(row['predicted_label'])
        predicted_prob = float(row['predicted_probability'])

        result = rag_explain(sequence, predicted_label, predicted_prob)
        return jsonify(result)

    except Exception as e:
        logger.error(f"API explanation failed: {traceback.format_exc()}")
        return jsonify({'status': 'failed', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6006)
