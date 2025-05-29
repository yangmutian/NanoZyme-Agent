import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils.logger import setup_logger
from config.settings import TRAINING_CONFIG

logger = setup_logger("material_analysis.models.trainer")

class ModelTrainer:
    def __init__(self, 
                 iterations: int = TRAINING_CONFIG['MODEL_PARAMS']['iterations'],
                 learning_rate: float = TRAINING_CONFIG['MODEL_PARAMS']['learning_rate'],
                 depth: int = TRAINING_CONFIG['MODEL_PARAMS']['depth'],
                 loss_function: str = TRAINING_CONFIG['MODEL_PARAMS']['loss_function'],
                 random_state: int = TRAINING_CONFIG['RANDOM_STATE']):
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function=loss_function,
            random_state=random_state,
            verbose=TRAINING_CONFIG['VERBOSE']
        )
        logger.info("ModelTrainer initialized with CatBoostClassifier")
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        logger.info(f"Training model on {X_train.shape[0]} samples with {X_train.shape[1]} features")
        self.model.fit(X_train, y_train)
        logger.info("Model training completed")
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, log_metrics: bool = True) -> Optional[Tuple[float, float, float, float]]:
        logger.info(f"Evaluating model on {X_test.shape[0]} test samples")
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        if log_metrics:
            logger.info(f"Model evaluation results:")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            return accuracy, precision, recall, f1
        
        return None
    
    def save_model(self, path: str) -> None:
        logger.info(f"Saving model to {path}")
        self.model.save_model(path)
        logger.info("Model saved successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X) 