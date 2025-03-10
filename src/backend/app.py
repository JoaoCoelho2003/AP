from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import load_model, evaluate_text

app = Flask(__name__)
CORS(app)

model_cache = {}

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    model_type = data.get('model', 'logistic')
    
    valid_models = ["rnn", "dnn", "logistic"]
    if model_type not in valid_models:
        return jsonify({'error': f'Invalid model type. Choose from: {", ".join(valid_models)}'}), 400
        
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        if model_type not in model_cache:
            print(f"Loading {model_type} model...")
            model, word_to_idx, embedding_model = load_model(model_type)
            model_cache[model_type] = (model, word_to_idx, embedding_model)
        else:
            print(f"Using cached {model_type} model")
            model, word_to_idx, embedding_model = model_cache[model_type]

        prediction, confidence = evaluate_text(model, word_to_idx, embedding_model, text, model_type)
        
        return jsonify({
            'prediction': prediction,
            'confidence': float(confidence),
            'model': model_type
        })
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({
        'models': [
            {'id': 'logistic', 'name': 'Logistic Regression'},
            {'id': 'dnn', 'name': 'Deep Neural Network'},
            {'id': 'rnn', 'name': 'Recurrent Neural Network'}
        ]
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True)

