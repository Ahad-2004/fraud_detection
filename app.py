from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for the HTML frontend

# Load the trained model, scaler, and label encoders
try:
    # Use relative paths to look in the same directory as app.py
    model_path = os.path.join(os.path.dirname(__file__), 'fraud_detection_model.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
    encoders_path = os.path.join(os.path.dirname(__file__), 'label_encoders.pkl')
    
    print(f"Looking for model files in: {os.path.dirname(__file__)}")
    print(f"Model path: {model_path}")
    print(f"Scaler path: {scaler_path}")
    print(f"Encoders path: {encoders_path}")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(encoders_path)
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = None
    scaler = None
    label_encoders = None

@app.route('/')
def home():
    return jsonify({"message": "Insurance Fraud Detection API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler, label_encoders
    
    if model is None or scaler is None or label_encoders is None:
        return jsonify({'error': 'Models not loaded properly'}), 500
    
    try:
        # Get input data from request
        data = request.json
        
        # Create DataFrame from input
        input_df = pd.DataFrame([data])
        
        # Apply label encoding to categorical columns
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                # Handle unseen labels by using the first label
                input_df[col] = input_df[col].apply(
                    lambda x: x if str(x) in encoder.classes_ else encoder.classes_[0]
                )
                input_df[col] = encoder.transform(input_df[col].astype(str))
        
        # Ensure all features are present and in the correct order
        expected_features = scaler.feature_names_in_
        missing_features = set(expected_features) - set(input_df.columns)
        
        # Add missing features with default values (mean from training)
        for feature in missing_features:
            input_df[feature] = 0  # Default value for missing features
        
        # Reorder columns to match training order
        input_df = input_df.reindex(columns=expected_features, fill_value=0)
        
        # Scale the features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]  # Probability of fraud
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'is_fraud': bool(prediction),
            'fraud_probability_percentage': float(probability * 100),
            'status': 'fraudulent' if prediction == 1 else 'genuine'
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None,
        'expected_features': list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else []
    })

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)