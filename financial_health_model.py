import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings

warnings.filterwarnings('ignore')


class FinancialHealthModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'monthly_income', 'monthly_expenses', 'savings_amount',
            'debt_amount', 'credit_score', 'emergency_fund_months',
            'investment_amount', 'age', 'employment_stability_years',
            'number_of_dependents', 'debt_to_income_ratio', 'savings_rate',
            'expense_ratio', 'credit_utilization'
        ]

    def generate_synthetic_data(self, n_samples=10000):
        """Generate synthetic financial data for training"""
        np.random.seed(42)

        # Generate base features
        data = {
            'monthly_income': np.random.normal(5000, 2000, n_samples).clip(1000, 20000),
            'credit_score': np.random.normal(650, 100, n_samples).clip(300, 850),
            'age': np.random.randint(18, 70, n_samples),
            'employment_stability_years': np.random.exponential(3, n_samples).clip(0, 40),
            'number_of_dependents': np.random.poisson(1.5, n_samples).clip(0, 8),
            'emergency_fund_months': np.random.exponential(2, n_samples).clip(0, 24),
        }

        # Generate correlated features
        data['monthly_expenses'] = data['monthly_income'] * np.random.uniform(0.3, 0.9, n_samples)
        data['savings_amount'] = (data['monthly_income'] - data['monthly_expenses']) * np.random.uniform(0.1, 0.8,
                                                                                                         n_samples)
        data['debt_amount'] = data['monthly_income'] * np.random.uniform(0.1, 3.0, n_samples)
        data['investment_amount'] = data['savings_amount'] * np.random.uniform(0.0, 0.5, n_samples)

        # Calculate derived features
        data['debt_to_income_ratio'] = data['debt_amount'] / data['monthly_income']
        data['savings_rate'] = data['savings_amount'] / data['monthly_income']
        data['expense_ratio'] = data['monthly_expenses'] / data['monthly_income']
        data['credit_utilization'] = np.random.uniform(0.0, 1.0, n_samples)

        df = pd.DataFrame(data)

        # Create financial health score (0-100)
        score = (
                (df['credit_score'] - 300) / 550 * 25 +  # Credit score component (25%)
                (1 - df['debt_to_income_ratio'].clip(0, 2) / 2) * 20 +  # Debt ratio (20%)
                df['savings_rate'].clip(0, 0.5) / 0.5 * 20 +  # Savings rate (20%)
                (df['emergency_fund_months'].clip(0, 6) / 6) * 15 +  # Emergency fund (15%)
                (1 - df['expense_ratio'].clip(0.5, 1.0) / 0.5) * 10 +  # Expense control (10%)
                (df['employment_stability_years'].clip(0, 10) / 10) * 10  # Employment stability (10%)
        )

        # Add some noise and ensure score is between 0-100
        score += np.random.normal(0, 5, n_samples)
        df['financial_health_score'] = score.clip(0, 100)

        return df

    def prepare_features(self, df):
        """Prepare features for training"""
        X = df[self.feature_names].copy()
        y = df['financial_health_score']
        return X, y

    def train_model(self, df):
        """Train the financial health model"""
        X, y = self.prepare_features(df)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Try different models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        best_model = None
        best_score = -np.inf

        print("Training and evaluating models...")

        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')

            # Train model
            model.fit(X_train_scaled, y_train)

            # Test predictions
            y_pred = model.predict(X_test_scaled)

            # Metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            print(f"\n{name} Results:")
            print(f"Cross-validation R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"Test R²: {r2:.4f}")
            print(f"Test MSE: {mse:.4f}")
            print(f"Test MAE: {mae:.4f}")

            if r2 > best_score:
                best_score = r2
                best_model = model
                self.model = model

        print(f"\nBest model selected with R² score: {best_score:.4f}")

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\nFeature Importance:")
            print(feature_importance)

        return self.model

    def predict_financial_health(self, user_data):
        """Predict financial health score for user data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Prepare input data
        input_df = pd.DataFrame([user_data])

        # Calculate derived features if not provided
        if 'debt_to_income_ratio' not in input_df.columns:
            input_df['debt_to_income_ratio'] = input_df['debt_amount'] / input_df['monthly_income']
        if 'savings_rate' not in input_df.columns:
            input_df['savings_rate'] = input_df['savings_amount'] / input_df['monthly_income']
        if 'expense_ratio' not in input_df.columns:
            input_df['expense_ratio'] = input_df['monthly_expenses'] / input_df['monthly_income']

        # Select features
        X = input_df[self.feature_names]

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        score = self.model.predict(X_scaled)[0]

        # Generate recommendations
        recommendations = self.generate_recommendations(user_data, score)

        return {
            'financial_health_score': round(score, 2),
            'score_category': self.get_score_category(score),
            'recommendations': recommendations
        }

    def get_score_category(self, score):
        """Categorize financial health score"""
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        elif score >= 20:
            return "Poor"
        else:
            return "Critical"

    def generate_recommendations(self, user_data, score):
        """Generate personalized recommendations"""
        recommendations = []

        # Debt-to-income ratio check
        debt_ratio = user_data['debt_amount'] / user_data['monthly_income']
        if debt_ratio > 0.4:
            recommendations.append(
                "Your debt-to-income ratio is high. Consider debt consolidation or payment strategies.")

        # Emergency fund check
        if user_data.get('emergency_fund_months', 0) < 3:
            recommendations.append("Build an emergency fund covering 3-6 months of expenses.")

        # Savings rate check
        savings_rate = user_data['savings_amount'] / user_data['monthly_income']
        if savings_rate < 0.2:
            recommendations.append("Try to save at least 20% of your monthly income.")

        # Credit score check
        if user_data['credit_score'] < 650:
            recommendations.append(
                "Focus on improving your credit score through timely payments and credit utilization management.")

        # Credit utilization check
        if user_data.get('credit_utilization', 0) > 0.3:
            recommendations.append("Keep credit utilization below 30% to improve your credit score.")

        # Investment recommendation
        if user_data.get('investment_amount', 0) < user_data['savings_amount'] * 0.3:
            recommendations.append("Consider investing a portion of your savings for long-term growth.")

        return recommendations

    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filepath}")


# Example usage and training
if __name__ == "__main__":
    # Create and train the model
    fh_model = FinancialHealthModel()

    print("Generating synthetic training data...")
    training_data = fh_model.generate_synthetic_data(10000)

    print("Training the model...")
    fh_model.train_model(training_data)

    # Save the model
    fh_model.save_model('financial_health_model.pkl')

    # Example prediction
    sample_user = {
        'monthly_income': 5000,
        'monthly_expenses': 3500,
        'savings_amount': 1000,
        'debt_amount': 15000,
        'credit_score': 720,
        'emergency_fund_months': 4,
        'investment_amount': 500,
        'age': 30,
        'employment_stability_years': 3,
        'number_of_dependents': 1,
        'credit_utilization': 0.25
    }

    result = fh_model.predict_financial_health(sample_user)
    print(f"\nSample Prediction:")
    print(f"Financial Health Score: {result['financial_health_score']}")
    print(f"Category: {result['score_category']}")
    print(f"Recommendations: {result['recommendations']}")