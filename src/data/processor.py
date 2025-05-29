import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import Composition

from src.utils.helpers import safe_composition_conversion
from src.utils.logger import setup_logger

logger = setup_logger("material_analysis.data.processor")

class DataProcessor:
    def __init__(self):
        self.ep_featurizer = ElementProperty.from_preset('magpie')
        logger.info("DataProcessor initialized with magpie ElementProperty featurizer")
    
    def load_and_split_data(self, 
                            file_path: str, 
                            target_column: str = 'label',
                            substance_column: str = 'Substance',
                            test_size: float = 0.2, 
                            random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logger.info(f"Loading data from {file_path}")
        
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Loaded {len(data)} samples from {file_path}")
            
            if target_column not in data.columns:
                available_columns = ', '.join(data.columns)
                error_msg = f"Target column '{target_column}' not found in data. Available columns: {available_columns}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            data['composition'] = data[substance_column].apply(safe_composition_conversion)
            
            logger.info("Generating magpie features")
            df_features = self.ep_featurizer.featurize_dataframe(data, col_id='composition')
            
            X = df_features.drop([substance_column, 'composition', target_column], axis=1)
            y = df_features[target_column]
            
            logger.info(f"Feature matrix shape: {X.shape}, Label vector shape: {y.shape}")
            
            logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            logger.info(f"Data split complete: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
            
            train_pos = sum(y_train == 1)
            train_neg = sum(y_train == 0)
            test_pos = sum(y_test == 1)
            test_neg = sum(y_test == 0)
            
            logger.info(f"Training set distribution: {train_pos} positive, {train_neg} negative")
            logger.info(f"Test set distribution: {test_pos} positive, {test_neg} negative")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
    
    def process_new_data(self, file_path: str, substance_column: str = 'Substance') -> pd.DataFrame:
        logger.info(f"Processing new data from {file_path}")
        
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Loaded {len(data)} samples for prediction")
            
            data['composition'] = data[substance_column].apply(safe_composition_conversion)
            
            df_features = self.ep_featurizer.featurize_dataframe(data, col_id='composition')
            
            X = df_features.drop([substance_column, 'composition'], axis=1)
            
            return X
        except Exception as e:
            logger.error(f"Error processing new data: {str(e)}")
            raise 