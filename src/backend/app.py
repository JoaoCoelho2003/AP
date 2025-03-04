from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import evaluate_text

app = Flask(__name__)
CORS(app)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    model = data.get('model', 'logistic')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    prediction, confidence = evaluate_text(text)
    
    return jsonify({
        'prediction': prediction,
        'confidence': float(confidence),
        'model': model
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({
        'models': [
            {'id': 'logistic', 'name': 'Logistic Regression'}
        ]
    })

if __name__ == '__main__':
    app.run(debug=True)