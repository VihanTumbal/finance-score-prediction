from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialHealthAPI:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_model()

    def load_model(self):
        """Load the trained model"""
        try:
            model_path = 'financial_health_model.pkl'
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                logger.info("Model loaded successfully")
            else:
                logger.error("Model file not found. Please train the model first.")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")

    def validate_input(self, data):
        """Validate input data"""
        required_fields = [
            'monthly_income', 'monthly_expenses', 'savings_amount',
            'debt_amount', 'credit_score', 'age'
        ]

        errors = []

        # Check required fields
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        if errors:
            return False, errors

        # Validate data types and ranges
        try:
            # Income validation
            if float(data['monthly_income']) <= 0:
                errors.append("Monthly income must be positive")

            # Expenses validation
            if float(data['monthly_expenses']) < 0:
                errors.append("Monthly expenses cannot be negative")

            # Savings validation
            if float(data['savings_amount']) < 0:
                errors.append("Savings amount cannot be negative")

            # Debt validation
            if float(data['debt_amount']) < 0:
                errors.append("Debt amount cannot be negative")

            # Credit score validation
            credit_score = float(data['credit_score'])
            if credit_score < 300 or credit_score > 850:
                errors.append("Credit score must be between 300 and 850")

            # Age validation
            age = int(data['age'])
            if age < 18 or age > 100:
                errors.append("Age must be between 18 and 100")

            # Credit utilization validation
            if 'credit_utilization' in data:
                credit_util = float(data['credit_utilization'])
                if credit_util < 0 or credit_util > 1:
                    errors.append("Credit utilization must be between 0 and 1")

        except ValueError:
            errors.append("Invalid data types provided")

        return len(errors) == 0, errors

    def process_input(self, data):
        """Process and prepare input data"""
        # Set default values for optional fields
        processed_data = {
            'monthly_income': float(data['monthly_income']),
            'monthly_expenses': float(data['monthly_expenses']),
            'savings_amount': float(data['savings_amount']),
            'debt_amount': float(data['debt_amount']),
            'credit_score': float(data['credit_score']),
            'age': int(data['age']),
            'emergency_fund_months': float(data.get('emergency_fund_months', 0)),
            'investment_amount': float(data.get('investment_amount', 0)),
            'employment_stability_years': float(data.get('employment_stability_years', 1)),
            'number_of_dependents': int(data.get('number_of_dependents', 0)),
            'credit_utilization': float(data.get('credit_utilization', 0.3))
        }

        # Calculate derived features
        processed_data['debt_to_income_ratio'] = processed_data['debt_amount'] / processed_data['monthly_income']
        processed_data['savings_rate'] = processed_data['savings_amount'] / processed_data['monthly_income']
        processed_data['expense_ratio'] = processed_data['monthly_expenses'] / processed_data['monthly_income']

        return processed_data

    def predict_financial_health(self, user_data):
        """Predict financial health score"""
        if self.model is None:
            raise ValueError("Model not loaded")

        # Prepare input data
        input_df = pd.DataFrame([user_data])
        X = input_df[self.feature_names]

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        score = self.model.predict(X_scaled)[0]
        score = max(0, min(100, score))  # Ensure score is between 0-100

        # Generate detailed analysis
        analysis = self.generate_detailed_analysis(user_data, score)

        return {
            'financial_health_score': round(score, 2),
            'score_category': self.get_score_category(score),
            'analysis': analysis,
            'recommendations': self.generate_recommendations(user_data, score),
            'key_metrics': self.calculate_key_metrics(user_data)
        }

    def get_score_category(self, score):
        """Categorize financial health score"""
        if score >= 80:
            return {"category": "Excellent", "color": "#22c55e", "description": "Outstanding financial health"}
        elif score >= 60:
            return {"category": "Good", "color": "#3b82f6", "description": "Good financial position"}
        elif score >= 40:
            return {"category": "Fair", "color": "#f59e0b", "description": "Room for improvement"}
        elif score >= 20:
            return {"category": "Poor", "color": "#ef4444", "description": "Needs attention"}
        else:
            return {"category": "Critical", "color": "#dc2626", "description": "Urgent action required"}

    def generate_detailed_analysis(self, user_data, score):
        """Generate detailed financial analysis"""
        analysis = {
            'strengths': [],
            'weaknesses': [],
            'risk_factors': []
        }

        # Analyze debt-to-income ratio
        debt_ratio = user_data['debt_to_income_ratio']
        if debt_ratio <= 0.2:
            analysis['strengths'].append("Excellent debt management")
        elif debt_ratio <= 0.4:
            analysis['strengths'].append("Manageable debt levels")
        else:
            analysis['weaknesses'].append("High debt-to-income ratio")
            analysis['risk_factors'].append("Debt burden may affect financial stability")

        # Analyze savings rate
        savings_rate = user_data['savings_rate']
        if savings_rate >= 0.2:
            analysis['strengths'].append("Strong savings habit")
        elif savings_rate >= 0.1:
            analysis['strengths'].append("Decent savings rate")
        else:
            analysis['weaknesses'].append("Low savings rate")

        # Analyze emergency fund
        emergency_months = user_data['emergency_fund_months']
        if emergency_months >= 6:
            analysis['strengths'].append("Excellent emergency fund")
        elif emergency_months >= 3:
            analysis['strengths'].append("Adequate emergency fund")
        else:
            analysis['weaknesses'].append("Insufficient emergency fund")
            analysis['risk_factors'].append("Vulnerable to unexpected expenses")

        # Analyze credit score
        credit_score = user_data['credit_score']
        if credit_score >= 750:
            analysis['strengths'].append("Excellent credit score")
        elif credit_score >= 650:
            analysis['strengths'].append("Good credit score")
        else:
            analysis['weaknesses'].append("Credit score needs improvement")

        return analysis

    def generate_recommendations(self, user_data, score):
        """Generate personalized recommendations"""
        recommendations = []

        # Priority recommendations based on score
        if score < 40:
            recommendations.append({
                'priority': 'high',
                'category': 'Emergency',
                'title': 'Focus on Financial Stability',
                'description': 'Your financial situation needs immediate attention. Consider financial counseling.'
            })

        # Debt management
        debt_ratio = user_data['debt_to_income_ratio']
        if debt_ratio > 0.4:
            recommendations.append({
                'priority': 'high',
                'category': 'Debt Management',
                'title': 'Reduce Debt Burden',
                'description': 'Your debt-to-income ratio is high. Consider debt consolidation or snowball method.'
            })

        # Emergency fund
        if user_data['emergency_fund_months'] < 3:
            recommendations.append({
                'priority': 'high',
                'category': 'Emergency Fund',
                'title': 'Build Emergency Fund',
                'description': 'Aim to save 3-6 months of expenses for unexpected situations.'
            })

        # Savings improvement
        savings_rate = user_data['savings_rate']
        if savings_rate < 0.2:
            recommendations.append({
                'priority': 'medium',
                'category': 'Savings',
                'title': 'Increase Savings Rate',
                'description': 'Try to save at least 20% of your income for long-term financial security.'
            })

        # Credit improvement
        if user_data['credit_score'] < 650:
            recommendations.append({
                'priority': 'medium',
                'category': 'Credit',
                'title': 'Improve Credit Score',
                'description': 'Focus on timely payments and keeping credit utilization low.'
            })

        # Investment advice
        if user_data['investment_amount'] < user_data['savings_amount'] * 0.3 and score > 60:
            recommendations.append({
                'priority': 'low',
                'category': 'Investment',
                'title': 'Consider Investing',
                'description': 'Start investing a portion of your savings for long-term growth.'
            })

        return recommendations

    def calculate_key_metrics(self, user_data):
        """Calculate key financial metrics"""
        return {
            'debt_to_income_ratio': {
                'value': round(user_data['debt_to_income_ratio'] * 100, 2),
                'label': 'Debt-to-Income Ratio (%)',
                'benchmark': 'Below 40% is ideal'
            },
            'savings_rate': {
                'value': round(user_data['savings_rate'] * 100, 2),
                'label': 'Savings Rate (%)',
                'benchmark': '20% or higher is recommended'
            },
            'expense_ratio': {
                'value': round(user_data['expense_ratio'] * 100, 2),
                'label': 'Expense Ratio (%)',
                'benchmark': 'Below 80% is good'
            },
            'emergency_fund_months': {
                'value': round(user_data['emergency_fund_months'], 1),
                'label': 'Emergency Fund (months)',
                'benchmark': '3-6 months is recommended'
            }
        }


# Initialize the API
fh_api = FinancialHealthAPI()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': fh_api.model is not None
    })


@app.route('/analyze', methods=['POST'])
def analyze_financial_health():
    """Main endpoint for financial health analysis"""
    try:
        # Get JSON data
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'No data provided',
                'status': 'error'
            }), 400

        # Validate input
        is_valid, errors = fh_api.validate_input(data)
        if not is_valid:
            return jsonify({
                'error': 'Invalid input data',
                'details': errors,
                'status': 'error'
            }), 400

        # Process input
        processed_data = fh_api.process_input(data)

        # Get prediction
        result = fh_api.predict_financial_health(processed_data)

        return jsonify({
            'status': 'success',
            'data': result,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'status': 'error'
        }), 500


@app.route('/metrics', methods=['POST'])
def get_financial_metrics():
    """Endpoint to get financial metrics only"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        processed_data = fh_api.process_input(data)
        metrics = fh_api.calculate_key_metrics(processed_data)

        return jsonify({
            'status': 'success',
            'metrics': metrics
        })

    except Exception as e:
        return jsonify({
            'error': 'Error calculating metrics',
            'message': str(e)
        }), 500


@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    """Endpoint to get recommendations only"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        processed_data = fh_api.process_input(data)

        # Get basic score for recommendations
        if fh_api.model is not None:
            input_df = pd.DataFrame([processed_data])
            X = input_df[fh_api.feature_names]
            X_scaled = fh_api.scaler.transform(X)
            score = fh_api.model.predict(X_scaled)[0]
        else:
            score = 50  # Default score if model not loaded

        recommendations = fh_api.generate_recommendations(processed_data, score)

        return jsonify({
            'status': 'success',
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({
            'error': 'Error generating recommendations',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)