# GSTIN Classification Model

## Overview
This project implements a classification model to predict GSTIN status using a dataset that contains various features. The model is built using the LightGBM algorithm, and hyperparameter optimization is performed with Hyperopt to achieve optimal performance.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Training](#model-training)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Visualizations](#visualizations)
7. [Contributing](#contributing)
8. [License](#license)

## Installation

To run this project, you need to have Python installed (preferably Python 3.6 or higher). The following packages are required:

- pandas
- numpy
- scikit-learn
- lightgbm
- hyperopt
- matplotlib
- seaborn
- scipy

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn lightgbm hyperopt matplotlib seaborn scipy
```

## Usage

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Prepare your dataset**: Place your training and test data CSV files in the correct paths as defined in the code. Update the paths in the following lines if necessary:
   ```python
   features_train = pd.read_csv("../GSTIN dataset/Train_60/Train_60/Train_60/X_Train_Data_Input.csv")
   labels_train = pd.read_csv("../GSTIN dataset/Train_60/Train_60/Train_60/Y_Train_Data_Target.csv")
   features_test = pd.read_csv("../GSTIN dataset/Test_20/Test_20/Test_20/X_Test_Data_Input.csv")
   labels_test = pd.read_csv("../GSTIN dataset/Test_20/Test_20/Test_20/Y_Test_Data_Target.csv")
   ```

3. **Run the script**:
   Execute the Python script in your terminal:
   ```bash
   python <script_name>.py
   ```

## Data Preprocessing

The data preprocessing steps include:
- **Dropping Unnecessary Columns**: Certain columns that do not contribute to the prediction are dropped from the dataset.
- **Handling Missing Values**: Missing values are imputed using strategies like mean, median, and iterative imputation.
- **Feature Reduction**: Features that do not provide significant information are removed.
- **Outlier Detection and Removal**: Outliers are identified and removed using Z-score method.
- **Feature Scaling**: Features are standardized using `StandardScaler` to improve model performance.

## Model Training

The model is trained using the LightGBM classifier with hyperparameter optimization performed through Hyperopt. The search space for hyperparameters includes:
- `n_estimators`
- `learning_rate`
- `max_depth`
- `num_leaves`
- `min_child_samples`
- `subsample`
- `colsample_bytree`
- `reg_alpha`
- `reg_lambda`

The optimization process aims to minimize the negative accuracy on the validation set.

## Evaluation Metrics

The following metrics are calculated to evaluate the model's performance:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC
- Balanced Accuracy
- Log Loss

## Visualizations

The script generates various visualizations to analyze model performance:
- **Confusion Matrix**: Displays the classification performance in terms of true positives, true negatives, false positives, and false negatives.
- **ROC Curve**: Illustrates the trade-off between sensitivity and specificity.
- **Precision-Recall Curve**: Shows the relationship between precision and recall for different probability thresholds.
- **Train vs Test Accuracy Plot**: Compares the accuracy of the model on training and test datasets.
