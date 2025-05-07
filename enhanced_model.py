# enhanced_model.py
import pandas as pd
import numpy as np
import pickle
import os

class EnhancedFraudModel:
    """A more sophisticated mock model that simulates patterns"""
    
    def __init__(self):
        print("Loading enhanced fraud detection model...")
        self.patterns = {
            'high_value': {'threshold': 1000, 'weight': 0.4},
            'suspicious_accounts': {
                'senders': ['user_7', 'user_13', 'user_42', 'user_87', 'suspicious_wallet'],
                'receivers': ['user_23', 'user_56', 'user_91', 'suspicious_entity'],
                'weight': 0.5
            },
            'unusual_timing': {'weight': 0.2},
            'network_centrality': {'weight': 0.3}
        }
        
        # Load "feature importance" data
        self.feature_importance = {
            'Transaction_Amount': 0.35,
            'Sender_Profile': 0.25,
            'Receiver_Profile': 0.20,
            'Timing_Pattern': 0.15,
            'Network_Position': 0.05
        }
        
        # "Model" weights
        self.weights = {
            "rf": 0.25,
            "xgb": 0.30,
            "isolation_forest": 0.20,
            "autoencoder": 0.25
        }
    
    def _calculate_base_score(self, transaction):
        """Calculate base fraud score using rule-based simulation"""
        score = 0.0
        
        # High value transaction check
        amount = transaction.get('Transaction_Amount', 0)
        if amount > self.patterns['high_value']['threshold']:
            score += min(0.5, (amount / 5000)) * self.patterns['high_value']['weight']
        
        # Suspicious account check
        sender = transaction.get('Sender_ID', '')
        receiver = transaction.get('Receiver_ID', '')
        
        if sender in self.patterns['suspicious_accounts']['senders']:
            score += 0.6 * self.patterns['suspicious_accounts']['weight']
        
        if receiver in self.patterns['suspicious_accounts']['receivers']:
            score += 0.7 * self.patterns['suspicious_accounts']['weight']
        
        # Keyword checks - looking for suspicious terms
        for field in [sender, receiver]:
            if 'suspicious' in str(field).lower() or 'anon' in str(field).lower():
                score += 0.8 * self.patterns['suspicious_accounts']['weight']
        
        # Add a touch of randomness (model uncertainty)
        score += np.random.normal(0, 0.05)  # Small random factor
        
        # Ensure score is between 0 and 1
        return max(0.01, min(0.99, score))
    
    def predict(self, transaction_df):
        """Generate prediction scores based on transaction features"""
        num_transactions = len(transaction_df)
        
        # Base scores
        base_scores = []
        for _, transaction in transaction_df.iterrows():
            base_scores.append(self._calculate_base_score(transaction))
        
        # Create variations for different models
        # Each model has slightly different predictions (like real ML models would)
        rf_variation = np.random.normal(0, 0.03, num_transactions)
        xgb_variation = np.random.normal(0, 0.02, num_transactions)
        if_variation = np.random.normal(0, 0.05, num_transactions)
        ae_variation = np.random.normal(0, 0.04, num_transactions)
        
        # Combine into final scores
        base_scores = np.array(base_scores)
        
        return {
            'ensemble_score': base_scores,
            'rf_score': np.clip(base_scores + rf_variation, 0, 1),
            'xgb_score': np.clip(base_scores + xgb_variation, 0, 1),
            'isolation_forest_score_norm': np.clip(base_scores + if_variation, 0, 1),
            'autoencoder_score_norm': np.clip(base_scores + ae_variation, 0, 1)
        }
    
    def explain_prediction(self, transaction_df):
        """Provide detailed explanation for a prediction"""
        if len(transaction_df) != 1:
            raise ValueError("Explanation only works for single transactions")
            
        transaction = transaction_df.iloc[0]
        fraud_score = self._calculate_base_score(transaction)
        is_fraud = fraud_score > 0.7
        
        # Create detailed explanation
        amount = transaction.get('Transaction_Amount', 0)
        sender = transaction.get('Sender_ID', '')
        receiver = transaction.get('Receiver_ID', '')
        
        # Construct explanation based on transaction properties
        factors = []
        if amount > 1000:
            factors.append(f"High transaction amount (${amount:.2f})")
        
        if sender in self.patterns['suspicious_accounts']['senders']:
            factors.append(f"Sender ({sender}) has unusual transaction history")
        
        if receiver in self.patterns['suspicious_accounts']['receivers']:
            factors.append(f"Receiver ({receiver}) associated with suspicious activity")
            
        if 'suspicious' in str(sender).lower() or 'suspicious' in str(receiver).lower():
            factors.append("Transaction involves flagged keywords")
            
        # Default factors if none found
        if not factors:
            if is_fraud:
                factors.append("Combination of unusual patterns detected")
            else:
                factors.append("Transaction appears to follow normal patterns")
        
        explanation_text = f"Analysis of transaction from {sender} to {receiver} for ${amount:.2f}:\n"
        explanation_text += " • " + "\n • ".join(factors)
        
        # Create model scores with realistic variations
        model_scores = {
            'Random Forest': min(0.95, max(0.05, fraud_score + np.random.normal(0, 0.03))),
            'XGBoost': min(0.95, max(0.05, fraud_score + np.random.normal(0, 0.02))),
            'Isolation Forest': min(0.95, max(0.05, fraud_score + np.random.normal(0, 0.05))),
            'Autoencoder': min(0.95, max(0.05, fraud_score + np.random.normal(0, 0.04)))
        }
        
        return {
            'is_fraud': is_fraud,
            'score': float(fraud_score),
            'explanation': explanation_text,
            'model_scores': model_scores,
            'contributing_factors': factors
        }

def create_dummy_model_file(file_path="models/ensemble/fraud_detection_model.pkl"):
    """Create a dummy model file for demonstration purposes"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Create and save a dummy model
    model = EnhancedFraudModel()
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Dummy model created at {file_path}")
    return file_path

def load_ensemble_model(model_path=None):
    """Load the enhanced model"""
    # If model doesn't exist, create it
    if model_path and not os.path.exists(model_path):
        print(f"Model not found at {model_path}, creating dummy model...")
        model_path = create_dummy_model_file(model_path)
    
    # Create and return the model
    return EnhancedFraudModel()

# Create the dummy model file when this script is run directly
if __name__ == "__main__":
    model_path = "models/ensemble/fraud_detection_model.pkl"
    create_dummy_model_file(model_path)
    print(f"Enhanced fraud detection model created at {model_path}")