import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from scipy import stats

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputers = {
            'Column0': SimpleImputer(strategy='mean'),
            'Column3': SimpleImputer(strategy='median'),
            'Column4': SimpleImputer(strategy='median'),
            'Column5': IterativeImputer(),
            'Column6': SimpleImputer(strategy='median'),
            'Column8': SimpleImputer(strategy='median'),
            'Column14': IterativeImputer(),
            'Column15': SimpleImputer(strategy='median')
        }

    def load_data(self, train_features_path, train_labels_path, 
                 test_features_path, test_labels_path):
        """
        Load data from CSV files
        """
        features_train = pd.read_csv(train_features_path)
        labels_train = pd.read_csv(train_labels_path)
        features_test = pd.read_csv(test_features_path)
        labels_test = pd.read_csv(test_labels_path)
        
        return features_train, labels_train, features_test, labels_test

    def combine_features_labels(self, features, labels):
        """
        Combine features and labels into a single DataFrame
        """
        return pd.concat([features, labels], axis=1)

    def drop_columns(self, df, columns_to_drop=None):
        """
        Drop unnecessary columns from DataFrame
        """
        if columns_to_drop is None:
            columns_to_drop = ["ID", "Column9"]
        return df.drop(columns=columns_to_drop, axis=1)

    def handle_missing_values(self, df):
        """
        Handle missing values using different imputation strategies
        """
        for column, imputer in self.imputers.items():
            if column in df.columns:
                df[column] = imputer.fit_transform(df[[column]])
        return df

    def remove_outliers(self, df, threshold=3):
        """
        Remove outliers using z-score method
        """
        z_scores = np.abs(stats.zscore(df.drop(columns=['target'])))
        return df[(z_scores < threshold).all(axis=1)]

    def scale_features(self, X_train, X_test):
        """
        Scale features using StandardScaler
        """
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train), 
            columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test), 
            columns=X_test.columns
        )
        return X_train_scaled, X_test_scaled

    def preprocess_data(self, train_features_path, train_labels_path, 
                       test_features_path, test_labels_path):
        """
        Complete preprocessing pipeline
        """
        # Load data
        features_train, labels_train, features_test, labels_test = self.load_data(
            train_features_path, train_labels_path, 
            test_features_path, test_labels_path
        )

        # Combine features and labels
        df_train = self.combine_features_labels(features_train, labels_train)
        df_test = self.combine_features_labels(features_test, labels_test)

        # Drop unnecessary columns
        df_train = self.drop_columns(df_train)
        df_test = self.drop_columns(df_test)

        # Handle missing values
        df_train = self.handle_missing_values(df_train)
        df_test = self.handle_missing_values(df_test)

        # Remove outliers (only from training data)
        df_train_clean = self.remove_outliers(df_train)

        # Separate features and target
        X_train = df_train_clean.drop(columns=['target'])
        y_train = df_train_clean['target']
        X_test = df_test.drop(columns=['target'])
        y_test = df_test['target']

        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test