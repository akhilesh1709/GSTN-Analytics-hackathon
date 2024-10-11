from lightgbm import LGBMClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np

class ModelTrainer:
    def __init__(self):
        self.space = {
            'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'max_depth': hp.quniform('max_depth', 3, 10, 1),
            'num_leaves': hp.quniform('num_leaves', 20, 100, 1),
            'min_child_samples': hp.quniform('min_child_samples', 10, 100, 1),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-8), np.log(1.0)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-8), np.log(1.0))
        }
        self.best_params = None
        self.model = None

    def objective(self, params, X_train, y_train, X_test, y_test):
        """
        Objective function for hyperparameter optimization
        """
        from sklearn.metrics import accuracy_score
        
        params = {
            'n_estimators': int(params['n_estimators']),
            'max_depth': int(params['max_depth']),
            'num_leaves': int(params['num_leaves']),
            'min_child_samples': int(params['min_child_samples']),
            'learning_rate': params['learning_rate'],
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'reg_alpha': params['reg_alpha'],
            'reg_lambda': params['reg_lambda']
        }
        
        model = LGBMClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {'loss': -accuracy, 'status': STATUS_OK}

    def optimize_hyperparameters(self, X_train, y_train, X_test, y_test, max_evals=50):
        """
        Optimize hyperparameters using Hyperopt
        """
        trials = Trials()
        best = fmin(
            fn=lambda params: self.objective(params, X_train, y_train, X_test, y_test),
            space=self.space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )
        
        self.best_params = {
            'n_estimators': int(best['n_estimators']),
            'learning_rate': best['learning_rate'],
            'max_depth': int(best['max_depth']),
            'num_leaves': int(best['num_leaves']),
            'min_child_samples': int(best['min_child_samples']),
            'subsample': best['subsample'],
            'colsample_bytree': best['colsample_bytree'],
            'reg_alpha': best['reg_alpha'],
            'reg_lambda': best['reg_lambda']
        }
        
        return self.best_params

    def train_model(self, X_train, y_train, params=None):
        """
        Train the model with given or best parameters
        """
        if params is None:
            params = self.best_params
        if params is None:
            raise ValueError("No parameters provided. Run optimize_hyperparameters first.")
            
        self.model = LGBMClassifier(**params, random_state=42)
        self.model.fit(X_train, y_train)
        return self.model

    def get_model(self):
        """
        Return the trained model
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Run train_model first.")
        return self.model