# Predictive Maintenance Project

## Project Overview

This project is a comprehensive predictive maintenance system that uses machine learning to predict machine failures before they occur. The system processes real-time sensor data to identify potential failures and provide early warnings to maintenance teams.

## Project Structure

```
Folder/
|
|-- data/                    # Dataset and data files
|   |-- ai4i2020.csv        # Sensor dataset with failure labels
|
|-- models/                  # Trained machine learning models
|   |-- catboost_model.cbm  # CatBoost model for failure prediction
|
|-- api/                     # FastAPI web service
|   |-- main.py             # REST API endpoints
|
|-- streaming/               # Real-time data simulation
|   |-- simulator.py        # Sensor data simulator
|
|-- model_training.py        # Model training script
|-- requirements.txt         # Python dependencies
|-- README.md               # This file
```

## Features

### Machine Learning Model
- **Algorithm**: CatBoost Classifier with class weighting
- **Performance**: 99.90% accuracy, 98.51% F1-score
- **Features**: Temperature, torque, rotational speed, tool wear, failure types
- **Handling**: Class imbalance with weighted training

### API Service
- **Framework**: FastAPI with Uvicorn server
- **Endpoints**:
  - `GET /`: Root endpoint with service info
  - `GET /health`: Health check endpoint
  - `POST /predict`: Single prediction endpoint
  - `POST /predict/batch`: Batch prediction endpoint
- **Features**: Real-time failure probability calculation

### Real-time Simulator
- **Data Source**: Random sampling from AI4I2020 dataset
- **Frequency**: 1 request per second
- **Output**: Color-coded terminal output with machine-specific warnings
- **Alerts**: Critical warnings for >80% failure probability

## Technology Stack

### Core Technologies
- **CatBoost**: Gradient boosting for classification
- **FastAPI**: Modern web framework for APIs
- **Uvicorn**: ASGI server for FastAPI
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning utilities
- **Kafka-python**: Real-time data streaming (ready for integration)

### Additional Libraries
- **Requests**: HTTP client for API calls
- **NumPy**: Numerical computing
- **Pydantic**: Data validation and settings management

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python model_training.py
```

## Usage

### 1. Start the API Service
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Run the Simulator
```bash
python streaming/simulator.py
```

### 3. API Usage Examples

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "Machine_ID": 1,
    "Air_temperature": 298.1,
    "Process_temperature": 308.6,
    "Rotational_speed": 1551,
    "Torque": 42.8,
    "Tool_wear": 0,
    "TWF": 0,
    "HDF": 0,
    "PWF": 0,
    "OSF": 0,
    "RNF": 0
}'
```

#### Health Check
```bash
curl http://localhost:8000/health
```

## Model Performance

The trained CatBoost model achieves:
- **Accuracy**: 99.90%
- **F1-Score**: 98.51%
- **Precision**: 100%
- **Recall**: 97%

### Feature Importance
1. PWF (Power Failure) - 34.6%
2. HDF (Heat Dissipation Failure) - 23.5%
3. OSF (Overstrain Failure) - 20.1%
4. TWF (Tool Wear Failure) - 13.5%
5. Rotational Speed - 3.6%

## Data Characteristics

- **Dataset**: AI4I2020 Predictive Maintenance Dataset
- **Samples**: 10,000 records
- **Features**: 14 columns including sensor readings and failure types
- **Class Distribution**: 96.61% normal, 3.39% failure
- **Failure Types**: TWF, HDF, PWF, OSF, RNF

## Alert System

The simulator provides color-coded alerts:
- **RED**: >80% failure probability - Critical warning
- **YELLOW**: 50-80% - High risk
- **BLUE**: 20-50% - Medium risk  
- **GREEN**: <20% - Normal operation

## Future Enhancements

- [ ] Kafka integration for real-time streaming
- [ ] Web dashboard for monitoring
- [ ] Historical data analysis
- [ ] Multiple machine support
- [ ] SMS/email alert notifications
- [ ] Model retraining pipeline

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Contact

For questions or suggestions, please open an issue in the repository.
