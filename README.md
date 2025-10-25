# Insurance Fraud Detection System

A machine learning-based web application for detecting fraudulent insurance claims using Flask and scikit-learn.

## Features

- **Real-time Fraud Detection**: Upload claim data and get instant fraud predictions
- **Interactive Web Interface**: User-friendly HTML interface with modern design
- **Probability Scoring**: Get fraud probability percentages for better decision making
- **RESTful API**: Clean API endpoints for integration with other systems
- **Health Monitoring**: Built-in health check endpoints

## Project Structure

```
fraud_detection/
├── app.py                      # Flask backend application
├── UI.html                     # Frontend web interface
├── fraud_detection_model.pkl   # Trained machine learning model
├── scaler.pkl                  # Feature scaler for data preprocessing
├── label_encoders.pkl          # Label encoders for categorical features
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fraud_detection.git
   cd fraud_detection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   - Open your browser and go to `http://127.0.0.1:5000`
   - Or open the `UI.html` file directly in your browser

### API Endpoints

#### Health Check
```http
GET /health
```
Returns the status of the application and model loading.

#### Fraud Prediction
```http
POST /predict
Content-Type: application/json

{
  "feature1": "value1",
  "feature2": "value2",
  ...
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.85,
  "is_fraud": true,
  "fraud_probability_percentage": 85.0,
  "status": "fraudulent"
}
```

## Model Information

The fraud detection model uses:
- **Algorithm**: Trained scikit-learn model (specific algorithm depends on training)
- **Features**: Preprocessed insurance claim data
- **Output**: Binary classification (0 = genuine, 1 = fraudulent)
- **Confidence**: Probability score for fraud likelihood

## Dependencies

- **Flask**: Web framework for the API
- **Flask-CORS**: Cross-origin resource sharing support
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library
- **joblib**: Model serialization

## Development

### Project Setup
1. Ensure all model files (`*.pkl`) are present in the project directory
2. The application automatically loads models on startup
3. Check the console output for model loading status

### Adding New Features
1. Modify `app.py` for backend changes
2. Update `UI.html` for frontend modifications
3. Test with the health check endpoint: `GET /health`

## API Documentation

### Request Format
All prediction requests should be JSON objects with the required features for the trained model.

### Response Format
```json
{
  "prediction": 0|1,
  "probability": 0.0-1.0,
  "is_fraud": true|false,
  "fraud_probability_percentage": 0.0-100.0,
  "status": "genuine"|"fraudulent"
}
```

## Troubleshooting

### Common Issues

1. **Models not loading**
   - Ensure all `.pkl` files are in the same directory as `app.py`
   - Check file permissions
   - Verify model files are not corrupted

2. **CORS errors**
   - The application includes Flask-CORS for cross-origin requests
   - If issues persist, check browser console for specific errors

3. **Prediction errors**
   - Verify input data format matches expected features
   - Check the health endpoint for model status
   - Review server logs for detailed error messages

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please open an issue in the GitHub repository.
