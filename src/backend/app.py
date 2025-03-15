from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import load_model, evaluate_text

app = Flask(__name__)
CORS(app)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    model_type = data.get('model', 'logistic')
        
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    model, vectorizer, word_to_idx, embedding_matrix = load_model(model_type)
    
    if model_type == "rnn" and (word_to_idx is None or embedding_matrix is None):
        return jsonify({'error': 'RNN model components not found'}), 500
    elif model_type != "rnn" and vectorizer is None:
        return jsonify({'error': 'Model or vectorizer not found'}), 500

    prediction, confidence = evaluate_text(model, vectorizer, word_to_idx, embedding_matrix, text, model_type)
    
    return jsonify({
        'prediction': prediction,
        'confidence': float(confidence),
        'model': model_type
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({
        'models': [
            {'id': 'logistic', 'name': 'Logistic Regression'},
            {'id': 'dnn', 'name': 'Deep Neural Network'},
            {'id': 'rnn', 'name': 'Recurrent Neural Network'}
        ]
    })

if __name__ == '__main__':
    app.run(debug=True)