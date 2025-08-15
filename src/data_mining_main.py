# Import all the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier

# Function to read the datasets into dataframes using pandas
def load_train_data(dataset_type):
    df = None
    if dataset_type == "TRAIN":
        train_df = pd.read_csv('train.csv')
        add_train_df = pd.read_csv('add_train.csv')
        df = pd.concat([train_df, add_train_df], ignore_index=True)
    elif dataset_type == "TEST":
        df = pd.read_csv('test.csv')

    return df

# Function to pre-process the data
def data_preprocessing(df, dataset_type):
    # Fetch numerical and categorical columns from the dataset
    numerical_cols = df.iloc[:, :100]
    categorical_cols = df.iloc[:, 100:128]

    # Handle missing/null values for numerical columns with mean imputation by class
    numerical_imputer = SimpleImputer(strategy='mean')
    numerical_cols = numerical_imputer.fit_transform(numerical_cols)

    # Handle missing/null values for categorical columns with mode imputation by class
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    categorical_cols = categorical_imputer.fit_transform(categorical_cols)

    # Update the DataFrame with imputed values
    df.iloc[:, :100] = numerical_cols
    df.iloc[:, 100:128] = categorical_cols

    if dataset_type == "TRAIN":
        # Replace outliers with median
        for col in df.select_dtypes(include=[np.number]).columns:
            outliers = detect_outliers_iqr(df[col])
            median_value = df[col].median()
            df[col] = np.where(outliers, median_value, df[col])

    # Normalize the dataset
    scaler = StandardScaler()
    numerical_cols = scaler.fit_transform(numerical_cols)
    df.iloc[:, :100] = numerical_cols

    return df

# Function to detect outliers using IQR method
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((data < lower_bound) | (data > upper_bound))
    return outliers

# Function to define and train the best model along with hyperparameters
def train_best_model(df):
    # Best hyperparameters for each model
    best_dt_params = {'criterion': 'gini', 'max_depth': 20}
    best_rf_params = {'criterion': 'gini', 'n_estimators': 100}
    best_knn_params = {'n_neighbors': 7, 'weights': 'distance'}

    # Initialize individual models
    decision_tree = DecisionTreeClassifier(**best_dt_params)
    random_forest = RandomForestClassifier(**best_rf_params)
    knn_classifier = KNeighborsClassifier(**best_knn_params)

    # Create a VotingClassifier with majority voting
    ensemble_model = VotingClassifier(estimators=[
        ('Decision Tree', decision_tree),
        ('Random Forest', random_forest),
        ('KNN', knn_classifier)
    ], voting='hard')

    # Initialize lists to store results
    accuracy_scores = []
    f1_scores = []

    # Initialize K-Fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Iterate through the K-Fold cross-validation
    for train_index, test_index in kf.split(df.iloc[:, :-1]):
        X_train, X_test = df.iloc[train_index, :-1], df.iloc[test_index, :-1]
        y_train, y_test = df.iloc[train_index, -1], df.iloc[test_index, -1]

        # Fit the ensemble model
        ensemble_model.fit(X_train, y_train)

        # Make predictions
        y_pred = ensemble_model.predict(X_test)

        # Calculate accuracy and F1-score
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        accuracy_scores.append(accuracy)
        f1_scores.append(f1)

    # Calculate the mean accuracy and mean F1 score
    mean_accuracy = np.mean(accuracy_scores)
    mean_f1 = np.mean(f1_scores)

    return ensemble_model, mean_accuracy, mean_f1

# Function to generate the output file for predictions and evaluation scores in desired format
def outputFormatter(pred, acc, f1, filename):
    # round acc and f1 to 3rd decimal place
    acc = "{:.3f}".format(acc, 3)
    f1 = "{:.3f}".format(f1, 3)
    if isinstance(pred, pd.DataFrame):
        pred = pred.values.tolist()
    if isinstance(pred, np.ndarray):
        pred = pred.tolist()
    assert isinstance(pred, list), "Unsupported type for pred. It should be either a list, numpy array or pandas dataframe"
    assert len(pred) == 300, "pred should be a list of 300 elements"
    pred_int = [int(x) for x in pred]
    csv_string = ',\n'.join(map(str, pred_int))
    csv_string += ',\n' + acc + ',' + f1
    filename = filename if filename.endswith('.csv') else filename + '.csv'
    with open(filename, 'w') as f:
        f.write(csv_string)
    return csv_string

# Function to call all the above functions to execute the logic
def main():
    # Load the training datasets
    train_df = load_train_data("TRAIN")

    # Data pre-processing on the training dataset
    train_df = data_preprocessing(train_df, "TRAIN")

    # Train the best model on training dataset
    ensemble_model, accuracy, f1 = train_best_model(train_df)

    # Load the testing dataset
    test_df = load_train_data("TEST")

    # Data pre-processing on the testing dataset
    test_df = data_preprocessing(test_df, "TEST")

    # Make predictions on the test data
    test_predictions = ensemble_model.predict(test_df)

    # FORMAT PREDICTIONS INTO OUTPUT FILE
    outputFormatter(test_predictions, accuracy, f1, "s4768708")

if __name__ == "__main__":
    main()
