# Financial Health Score Prediction Model

A machine learning-powered web application that analyzes personal financial data to predict financial health scores and provide actionable insights for financial improvement.

## 🌟 Features

- **Financial Health Analysis**: Predicts financial health scores based on multiple financial indicators
- **Risk Assessment**: Categorizes financial risk levels (Low, Medium, High)
- **Personalized Recommendations**: Provides tailored advice for improving financial health
- **RESTful API**: Easy-to-use Flask backend with CORS support
- **Responsive Frontend**: Clean HTML interface for data input and results display

## 🏗️ Project Structure

```
ScorePredictionModel/
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── financial_health_model.py      # ML model training script
├── financial_health_model.pkl     # Trained model file
├── flask_backend.py              # Flask API server

```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Local Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/financial-health-prediction.git
   cd financial-health-prediction
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (if model file doesn't exist)

   ```bash
   python financial_health_model.py
   ```

4. **Start the Flask server**

   ```bash
   python flask_backend.py
   ```

5. **Open the web interface**
   - Open `index.html` in your browser, or
   - Navigate to `http://localhost:5000` (if serving static files through Flask)

## 🔧 API Documentation

### Base URL

- Local: `http://localhost:5000`
- Production: `https://your-app-name.onrender.com`

### Endpoints

#### Health Check

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### Financial Analysis

```http
POST /analyze
Content-Type: application/json
```

**Request Body:**

```json
{
  "monthly_income": 5000,
  "monthly_expenses": 3500,
  "savings_amount": 1000,
  "debt_amount": 15000,
  "credit_score": 720,
  "age": 30,
  "emergency_fund_months": 4,
  "investment_amount": 500,
  "employment_stability_years": 3,
  "number_of_dependents": 1,
  "credit_utilization": 0.25
}
```

**Response:**

```json
{
  "financial_health_score": 75.32,
  "risk_level": "Medium",
  "recommendations": [
    "Consider increasing your emergency fund to 6 months of expenses",
    "Your credit utilization is good, keep it below 30%",
    "Continue building your investment portfolio"
  ],
  "key_metrics": {
    "savings_rate": 20.0,
    "debt_to_income_ratio": 3.0,
    "emergency_fund_adequacy": "Good"
  }
}
```

## 📊 Model Information

### Algorithm

- **Type**: Random Forest Regressor
- **Features**: 11 financial indicators
- **Target**: Financial Health Score (0-100)

### Input Features

| Feature                      | Description                | Type    | Range   |
| ---------------------------- | -------------------------- | ------- | ------- |
| `monthly_income`             | Monthly gross income       | Float   | > 0     |
| `monthly_expenses`           | Monthly total expenses     | Float   | > 0     |
| `savings_amount`             | Current savings balance    | Float   | ≥ 0     |
| `debt_amount`                | Total debt amount          | Float   | ≥ 0     |
| `credit_score`               | Credit score               | Integer | 300-850 |
| `age`                        | Age in years               | Integer | 18-100  |
| `emergency_fund_months`      | Emergency fund in months   | Float   | ≥ 0     |
| `investment_amount`          | Investment portfolio value | Float   | ≥ 0     |
| `employment_stability_years` | Years at current job       | Float   | ≥ 0     |
| `number_of_dependents`       | Number of dependents       | Integer | ≥ 0     |
| `credit_utilization`         | Credit utilization ratio   | Float   | 0-1     |

### Risk Categories

- **Low Risk**: Score ≥ 70
- **Medium Risk**: Score 40-69
- **High Risk**: Score < 40

## 🚀 Deployment

### Deploy on Render

1. **Push to GitHub**

   ```bash
   git add .
   git commit -m "Deploy to Render"
   git push origin main
   ```

2. **Deploy on Render**
   - Go to [render.com](https://render.com)
   - Create new Web Service
   - Connect your GitHub repository
   - Use these settings:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn flask_backend:app`
     - **Environment**: Python 3

### Other Deployment Options

- **Heroku**: Use `Procfile` with `web: gunicorn flask_backend:app`
- **Railway**: Automatic deployment from GitHub
- **AWS EC2**: Manual server setup with nginx/gunicorn
- **Google Cloud Run**: Containerized deployment

## 🧪 Testing

### Test the API with curl

```bash
# Health check
curl -X GET http://localhost:5000/health

# Financial analysis
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_income": 5000,
    "monthly_expenses": 3500,
    "savings_amount": 1000,
    "debt_amount": 15000,
    "credit_score": 720,
    "age": 30,
    "emergency_fund_months": 4,
    "investment_amount": 500,
    "employment_stability_years": 3,
    "number_of_dependents": 1,
    "credit_utilization": 0.25
  }'
```

## 🛠️ Development

### Adding New Features

1. **Model improvements**: Modify `financial_health_model.py`
2. **API endpoints**: Add routes in `flask_backend.py`
3. **Frontend features**: Update `index.html`, `style.css`, and `script.js`

### Model Retraining

To retrain the model with new data:

```bash
python financial_health_model.py
```

This will generate a new `financial_health_model.pkl` file.

## 📋 Dependencies

```txt
Flask==2.3.3
Flask-CORS==4.0.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
gunicorn==21.2.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For support, email your-email@example.com or create an issue on GitHub.

## 🚧 Roadmap

- [ ] Add user authentication
- [ ] Implement data persistence with database
- [ ] Add more sophisticated ML models
- [ ] Create mobile app version
- [ ] Add historical tracking and trends
- [ ] Implement A/B testing for recommendations

## ⚠️ Disclaimer

This application provides general financial insights and should not be considered as professional financial advice. Always consult with qualified financial advisors for important financial decisions.

---

**Built with ❤️ using Flask, scikit-learn, and modern web technologies**
