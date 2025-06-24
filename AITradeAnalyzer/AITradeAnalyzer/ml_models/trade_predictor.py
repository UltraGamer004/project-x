import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class TradePredictorML:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained_flag = False
        self.feature_importance = None
        self.accuracy = 0.0
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
    
    def create_targets(self, price_data, lookforward_periods=5, min_move_percentage=0.5):
        """
        Create target labels based on future price movements
        1 = Buy signal (price goes up)
        -1 = Sell signal (price goes down)
        0 = No signal (sideways movement)
        """
        targets = []
        closes = price_data['Close'].values
        
        for i in range(len(closes)):
            if i + lookforward_periods >= len(closes):
                targets.append(0)  # Default to no signal for recent data
                continue
            
            current_price = closes[i]
            future_high = max(closes[i+1:i+1+lookforward_periods])
            future_low = min(closes[i+1:i+1+lookforward_periods])
            
            upward_move = (future_high - current_price) / current_price * 100
            downward_move = (current_price - future_low) / current_price * 100
            
            if upward_move > min_move_percentage and upward_move > downward_move:
                targets.append(1)  # Buy signal
            elif downward_move > min_move_percentage and downward_move > upward_move:
                targets.append(-1)  # Sell signal
            else:
                targets.append(0)  # No clear signal
        
        return np.array(targets)
    
    def prepare_features(self, features_df):
        """Prepare and clean features for training"""
        # Fill NaN values
        features_clean = features_df.fillna(method='ffill').fillna(0)
        
        # Remove infinite values
        features_clean = features_clean.replace([np.inf, -np.inf], 0)
        
        return features_clean
    
    def train(self, features_df, price_data):
        """Train the ML model"""
        try:
            # Prepare features
            features_clean = self.prepare_features(features_df)
            
            # Create targets
            targets = self.create_targets(price_data)
            
            # Ensure same length
            min_length = min(len(features_clean), len(targets))
            X = features_clean.iloc[:min_length]
            y = targets[:min_length]
            
            # Remove samples with no signal for balanced training
            non_zero_indices = y != 0
            if np.sum(non_zero_indices) > 50:  # Ensure we have enough samples
                X_filtered = X[non_zero_indices]
                y_filtered = y[non_zero_indices]
            else:
                X_filtered = X
                y_filtered = y
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Calculate accuracy
            y_pred = self.model.predict(X_test_scaled)
            self.accuracy = accuracy_score(y_test, y_pred) * 100
            
            # Store feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            self.is_trained_flag = True
            
        except Exception as e:
            print(f"Error during training: {e}")
            self.is_trained_flag = False
    
    def predict(self, features_df):
        """Make predictions on new data"""
        if not self.is_trained_flag:
            return pd.Series([0] * len(features_df), index=features_df.index)
        
        try:
            # Prepare features
            features_clean = self.prepare_features(features_df)
            
            # Scale features
            features_scaled = self.scaler.transform(features_clean)
            
            # Make predictions
            predictions = self.model.predict(features_scaled)
            probabilities = self.model.predict_proba(features_scaled)
            
            # Create result DataFrame
            result = pd.DataFrame({
                'prediction': predictions,
                'confidence': np.max(probabilities, axis=1) * 100
            }, index=features_df.index)
            
            return result
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return pd.DataFrame({
                'prediction': [0] * len(features_df),
                'confidence': [0] * len(features_df)
            }, index=features_df.index)
    
    def is_trained(self):
        """Check if model is trained"""
        return self.is_trained_flag
    
    def get_accuracy(self):
        """Get model accuracy"""
        return self.accuracy
    
    def get_feature_importance(self):
        """Get feature importance (if available)"""
        return self.feature_importance
