import os
import logging
from datetime import datetime
from src.preprocess import DataPreprocessor
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator

# Configure logging
def setup_logging():
    """Set up logging configuration"""
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_directory}/gstin_classifier_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

def main():
    """Main function to run the GSTIN classification pipeline"""
    try:
        # Set up logging
        setup_logging()
        logging.info("Starting GSTIN Classification Pipeline")

        # Define data paths
        data_paths = {
            'train_features': "GSTIN dataset/Train_60/Train_60/Train_60/X_Train_Data_Input.csv",
            'train_labels': "GSTIN dataset/Train_60/Train_60/Train_60/Y_Train_Data_Target.csv",
            'test_features': "GSTIN dataset/Test_20/Test_20/Test_20/X_Test_Data_Input.csv",
            'test_labels': "GSTIN dataset/Test_20/Test_20/Test_20/Y_Test_Data_Target.csv"
        }

        # Initialize components
        logging.info("Initializing pipeline components")
        preprocessor = DataPreprocessor()
        trainer = ModelTrainer()
        evaluator = ModelEvaluator()

        # Preprocess data
        logging.info("Starting data preprocessing")
        X_train_scaled, X_test_scaled, y_train, y_test = preprocessor.preprocess_data(
            train_features_path=data_paths['train_features'],
            train_labels_path=data_paths['train_labels'],
            test_features_path=data_paths['test_features'],
            test_labels_path=data_paths['test_labels']
        )
        logging.info("Data preprocessing completed")

        # Optimize hyperparameters
        logging.info("Starting hyperparameter optimization")
        best_params = trainer.optimize_hyperparameters(
            X_train_scaled, 
            y_train, 
            X_test_scaled, 
            y_test, 
            max_evals=50
        )
        logging.info(f"Best hyperparameters found: {best_params}")

        # Train model
        logging.info("Training model with optimized hyperparameters")
        model = trainer.train_model(X_train_scaled, y_train)
        logging.info("Model training completed")

        # Evaluate model
        logging.info("Starting model evaluation")
        evaluator.set_model(model)
        
        # Generate comprehensive evaluation report
        logging.info("Generating evaluation report")
        evaluator.generate_evaluation_report(
            X_train_scaled, 
            y_train, 
            X_test_scaled, 
            y_test
        )

        # Save results
        save_results(model, best_params)
        logging.info("Pipeline completed successfully")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

def save_results(model, best_params):
    """Save model and results"""
    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # Save model
        import joblib
        model_path = f"{results_dir}/model_{timestamp}.joblib"
        joblib.dump(model, model_path)
        logging.info(f"Model saved to {model_path}")

        # Save hyperparameters
        import json
        params_path = f"{results_dir}/hyperparameters_{timestamp}.json"
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        logging.info(f"Hyperparameters saved to {params_path}")

    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")
        raise

if __name__ == "__main__":
    main()