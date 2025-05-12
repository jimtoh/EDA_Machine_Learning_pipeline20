# Requires Python 3.11.0
# Standard library imports
import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
# Define the mapping for classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# error: remove import that is not used

#########################################################################################
# print_separator() - prints a separator to the console for ease of reading
#########################################################################################
def print_separator():  # This function is used within the same scope
    """
    This function performs the following steps:
        1. Print a separator line to the console.
    """
    print("-"*30)

#########################################################################################
# Define or import the classification_setup function
#########################################################################################
def classification_setup(config: Dict):
    """
    This function performs the following steps:
        1. Load the dataset from a CSV file.
        2. Print the shape of the dataframe.
        3. Call the classification_preprocessor function to preprocess the data.
    """
    print_separator
    print("<<< Classification Setup Results >>>")
    print_separator

    # Map model names to their corresponding classes
    classification_model_classes = {
        'LogisticRegression': LogisticRegression,
        'KNeighborsClassifier': KNeighborsClassifier,
        'RandomForestClassifier': RandomForestClassifier,
        'SVC': SVC,
        'GradientBoostingClassifier': GradientBoostingClassifier,
        'AdaBoostClassifier': AdaBoostClassifier,
        'DecisionTreeClassifier': DecisionTreeClassifier
    }

    def get_classification_models():
        """
        This function returns a dictionary of classification models.
        """
        models = {}
        for model_name in classification_models:
            # Dynamically create the model object and add it to the models dictionary
            if model_name in classification_model_classes:  # classification_model_classes is defined in CELL INDEX: 3
                models[model_name] = classification_model_classes[model_name]()
                # List the model name and its object
                # print(f"Model mapping: {model_name}, Model object: {models[model_name]}")
 
        return models
    
    # print_separator()
    # Initialize classification_models from the config dictionary
    classification_models = config.get('classification_models', []) if config else []

    # Print model list
    if classification_models:
        classification_models = get_classification_models()
        # Print the classification models
        # print('The classification models are:', classification_models)
    else:
        print("No classification models to process.")

    # print_separator()
    # Get "classification_target" from config.yaml
    if 'classification_target' in config:
        target_column_discrete = config['classification_target'] 
        print('Target column for Classification:', target_column_discrete)
    else:
        print("Error: 'classification_target' key is missing in the configuration.")
    
    # print_separator()

    # Print type of hyperparameters performance:
    # print("Classification: Optimum or fast tuning performance set to:", config['classification_tuning_performance_set_optimum'])

    # Get the hyperparameters from the config.yaml file
    if config.get('classification_tuning_performance_set_optimum') == 1:
        # If the key exists, set hyperparameters_classification to classification_hyperparameters_fast
        hyperparameters_classification = config['classification_hyperparameters_fast']
    elif config.get('classification_tuning_performance_set_optimum') == 2:
        # If the key exists, set hyperparameters_classification to classification_hyperparameters_slow
        hyperparameters_classification = config['classification_hyperparameters_slow']
    else:
        print("Error: 'classification_tuning_performance_set_optimum' key is missing or invalid in the configuration.")

    # Print hyperparameters values
    # print('Hyperparameters are:')
    # print('Models:', hyperparameters_classification)

    # print_separator()
    # From config.yaml, the values in "classification_metrics" are the metrics to be used for classification
    # Get the classification metrics from the config.yaml file
    if 'classification_metrics' in config:
        classification_metrics = config['classification_metrics']
        # print(f"Metrics used: {classification_metrics}")
    else:
        print("Error: 'classification_metrics' key is missing in the configuration.")
        return

    # print_separator()

    return (target_column_discrete, 
            classification_models, 
            classification_model_classes, 
            hyperparameters_classification, 
            classification_metrics)

# def classification_setup(config)

#########################################################################################
# Define or import the classification_setup_results function
#########################################################################################
def classification_setup_results (target_column_discrete, classification_models, hyperparameters_classification, classification_metrics, config):
    """
    This function returns the results of the classification setup.
    """
    
    print_separator()
    # Print the classification target, models, hyperparameters_classification and metrics
    print(f"Target column: {target_column_discrete}")
    print_separator()
    print(f"Classification models: {classification_models}")
    print_separator()
    print(f"Classification tuning optimum is: {config['classification_tuning_performance_set_optimum']}")
    print(f"Hyperparameters classification: {hyperparameters_classification}")
    print_separator()
    print(f"Classification metrics: {classification_metrics}")

    return
# def classification_setup_results (target_column_discrete, classification_models, hyperparameters_classification, classification_metrics)

#########################################################################################
# The function classification_preprocessor is defined to preprocess the data for classification tasks.
#########################################################################################
def classification_preprocessor(df, config):
    """
    This function performs the following steps:
        1. Identifies the target column for classification
        2. Identifies the numerical columns
        3. Identifies the categorical columns
        4. Creates a column transformer with OneHotEncoder for categorical columns and StandardScaler for numerical columns
        5. Splits the data into training and testing sets
        6. Returns the training and testing sets and the column transformer
    """
    
    logging.info("--- Identifying the target column for classification")
    target_column_discrete = config.get('classification_target', 'Subscription Status')
    # print("Target column for classification: ", target_column_discrete)

    logging.info("Data cleaning - showing initial data")
    # print_separator()

    logging.info("--- Identifying the numerical columns")
    # Identify numerical columns
    numerical_columns = df.select_dtypes(exclude=['object']).columns
    # print("Numerical columns: ", numerical_columns)
    
    # If target column is in numerical-columns, remove it
    numerical_columns = numerical_columns.difference([target_column_discrete])

    # print the numerical columns for verification
    print('Verifying Numerical columns: ', numerical_columns)
    # print_separator()

    logging.info("--- Identifying the categorical columns")
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # If target column is in categorical, remove it
    categorical_columns = categorical_columns.difference([target_column_discrete])
    # print the categorical columns for verification
    print('Verifying Categorical_columns : ', categorical_columns)
    # print_separator()

    logging.info("--- Creating a column transformer with OneHotEncoder for categorical columns and StandardScaler for numerical columns")
    # Create a column transformer with OneHotEncoder for categorical columns and StandardScaler for numerical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)  # Added handle_unknown='ignore'
    ])

    logging.info("--- Splitting the data into training and testing sets")
    # Split the data into training and testing sets
    X = df.drop(target_column_discrete, axis=1)
    y = df[target_column_discrete]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logging.info("--- Classification Preprocessor - Completed")

    return (target_column_discrete, 
            numerical_columns, 
            categorical_columns, 
            X_train, 
            X_test, 
            y_train, 
            y_test, 
            preprocessor)

# def classification_preprocessor(df, config)

def classification_preprocessor_results(target_column_discrete, df, numerical_columns, categorical_columns, X_train, X_test, y_train, y_test, preprocessor):
    """
    This function displays the preprocessor results.
    """
    print_separator
    print("<<< Classification Preprocessor Setup Results >>>")
    print_separator
    # Check the returned values
    print('The target column is:', target_column_discrete)
    print_separator()
    
    # Print the shape of the dataframe
    print('The number of rows in the dataset is:', df.shape[0])
    
    # Print the numerical columns for verification
    print('Verifying Numerical columns:', numerical_columns)
    
    # Print the categorical columns for verification
    print('Verifying Categorical columns:', categorical_columns)
    print_separator()
    
    # Print the shape of the training and testing sets
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)
    print_separator()
    
    # Print preprocessor details
    print('Preprocessor:', preprocessor)
    print_separator()

    return (target_column_discrete, numerical_columns, categorical_columns, X_train, X_test, y_train, y_test, preprocessor)
# def classification_preprocessor_results(target_column_discrete, df, numerical_columns, categorical_columns, X_train, X_test, y_train, y_test, preprocessor)

# ###################################################################
# Perform the Classification modeling - getting the baesline
# ###################################################################
def classification_baseline(df: pd.DataFrame, classification_models, preprocessor, X_train, X_test, y_train, y_test):
    """
    This function performs the following steps:
        1. Imports the required libraries
        2. Creates a dictionary of classification models
        3. Preprocesses the data for classification
        4. Creates a pipeline for each model
        5. Fits the pipeline to the training data
        6. Predicts the target on the testing data
        7. Calculates the metrics
        8. Returns the metrics dataframe
    """
    # Ensure preprocessor and train-test split variables are passed correctly
    if preprocessor is None or X_train is None or X_test is None or y_train is None or y_test is None:
        raise ValueError("Preprocessor and train-test split variables must be provided.")
    # Use the passed classification_models directly
    # Use the passed classification_models directly
    models = classification_models
    # print the dictionary of models
    print ('The classification models are:')
    print(models)
    print_separator()

    print("Part 1 - Baseline of Classification Models")
    print_separator()

    # Create an empty dictionary to store the metrics
    metrics = {}

    # For each model, calculate the metrics and store in the dictionary
    for name, model in models.items():
        # print the name of the model
        print("--- Model running: ", name)
        logging.info("--- 1. Creating a pipeline for the model")
        # Create a pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', model)])
        logging.info("--- 2. Fitting the pipeline to the training data")
        # Fit the pipeline to the training data
        pipeline.fit(X_train, y_train)
        logging.info("--- 3. Predicting the target on the testing data")
        # Predict the target on the testing data
        logging.info("--- 4. Calculating the metrics")
        y_pred = pipeline.predict(X_test)
        # Calculate the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # Calculate the precision
        precision = precision_score(y_test, y_pred, average='weighted')
        # Calculate the recall
        recall = recall_score(y_test, y_pred, average='weighted')
        # Calculate the F1 score
        f1 = f1_score(y_test, y_pred, average='weighted')
        # Store the metrics in the dictionary
        metrics[name] = [accuracy, precision, recall, f1]

        # Create a dataframe from the metrics dictionary
        logging.info("--- Creating a dataframe from the metrics dictionary")
        metrics_df = pd.DataFrame(metrics, index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])

    # Print the metrics_tuned_df dataframe
    print(metrics_df)

    # Create a dataframe from the metrics dictionary
    logging.info("--- Creating a dataframe from the metrics dictionary")
    print_separator()
    metrics_df = pd.DataFrame(metrics, index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])

    # Print the metrics dataframe
    print(metrics_df)

    print_separator()
    # Print the bext model based on the metrics
    # Print the best model based on F1 Score with 3 decimal places
    print("Best model based on F1 Score:", metrics_df.loc['F1 Score'].idxmax(), ", with a F1 Score of: ", round(metrics_df.loc['F1 Score'].max(), 3))

    # Print the hyperparameters of the best model
    print("Hyperparameters of the best model:")
    print(models[metrics_df.loc['F1 Score'].idxmax()])
    logging.info("--- Classification Baseline Models - Completed")

    # With the best model, print the confusion matrix
    # Get the best model
    best_model = metrics_df.loc['F1 Score'].idxmax()
    # Get the best model object
    best_model_object = models[best_model]
    # Create a pipeline with the best model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', best_model_object)])
    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)
    # Predict the target on the testing data
    y_pred = pipeline.predict(X_test)
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Create a confusion matrix display
    cmd = ConfusionMatrixDisplay(cm, display_labels=pipeline.classes_)
    # Plot the confusion matrix
    cmd.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {best_model}')
    #plt.show()
    plt.savefig(f'./data/Baseline_confusion_matrix_{best_model}.png', dpi=300, bbox_inches='tight')  # Save the plot as a PNG file
    plt.close()  # Close the plot to free up memory
    
    return metrics_df
# def classification_baseline(df, models):

# ###################################################################
# Perform the Classification modeling - with hyperparameters tuning
# ###################################################################
def classification_models_hyperparameters(df: pd.DataFrame, classification_models, hyperparameters_classification, preprocessor, X_train, X_test, y_train, y_test, config):

    """
    This function performs the following steps:
        1. Imports the required libraries
        2. Creates a dictionary of classification models
        3. Preprocesses the data for classification
        4. Creates a pipeline for each model
        5. Defines a dictionary of hyperparameters for each model
        6. Fits the GridSearchCV object to the training data
        7. Predicts the target on the testing data
        8. Calculates the metrics
        9. Prints the best model based on the F1 score
        10. Returns the metrics dataframe

    Args:
        df: The input dataframe containing the data.
        models: A list of classification models to be used.
        hyperparameters: A dictionary containing the hyperparameters for each model.

    Returns:
        metrics_tuned_df: A dataframe containing the metrics for each model.
    """
    # performance is a string that can take one of three values: 'low', 'mid', 'high'

    """
    This function performs the following steps:
        1. Imports the required libraries
        2. Creates a dictionary of classification models
        3. Preprocesses the data for classification
        4. Creates a pipeline for each model
        5. Defines a dictionary of hyperparameters for each model
        6. Fits the GridSearchCV object to the training data
        7. Predicts the target on the testing data
        8. Calculates the metrics
        9. Prints the best model based on the R2 score
        10. Returns the metrics dataframe
    """
    # Use the classification_models dictionary directly
    models = classification_models
    # print the dictionary of models
    print ('The classification models are:')
    print(models)
    print_separator()

    print("Part 2 - Classification Models with Hyperparameter tuning")
    print_separator()

    # print the Classification tuning optimum setting
    print(f"Classification tuning optimum is: {config['classification_tuning_performance_set_optimum']}") 
    # Print the hyperparameters dictionary
    print(hyperparameters_classification)
    print_separator()

    # Create an empty dictionary to store the metrics
    metrics_tuned = {}

    # For each model, calculate the metrics and store in the dictionary
    for name, model in models.items():
                # print the name of the model
        print("--- Model running (with hyperparameters tuning ): ", name)
        logging.info("--- 1. Creating a pipeline for the model")
        # Create a pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', model)])
        logging.info("--- 2. Creating a GridSearchCV object")
        # Create a GridSearchCV object
        grid_search = GridSearchCV(pipeline, hyperparameters_classification[name], cv=5, n_jobs=-1)
        logging.info("--- 3. Fitting the GridSearchCV object to the training data")
        # Fit the GridSearchCV object to the training data
        grid_search.fit(X_train, y_train)
        logging.info("--- 4. Predicting the target on the testing data")
        # Predict the target on the testing data
        logging.info("--- 5. Calculating the metrics")
        y_pred = grid_search.predict(X_test)
        # Calculate the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # Calculate the precision
        precision = precision_score(y_test, y_pred, average='weighted')
        # Calculate the recall
        recall = recall_score(y_test, y_pred, average='weighted')
        # Calculate the F1 score
        f1 = f1_score(y_test, y_pred, average='weighted')
        # Store the metrics in the dictionary
        metrics_tuned[name] = [accuracy, precision, recall, f1]

        # Create a dataframe from the metrics dictionary
        logging.info("--- Creating a dataframe from the metrics dictionary")
        metrics_tuned_df = pd.DataFrame(metrics_tuned, index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])

    # Print the metrics_tuned_df dataframe
    print(metrics_tuned_df)

    print_separator()
    # Print the bext model based on the metrics
    # Print the best model based on F1 Score with 3 decimal places
    print("Best model based on F1 Score:", metrics_tuned_df.loc['F1 Score'].idxmax(), ", with a F1 Score of: ", round(metrics_tuned_df.loc['F1 Score'].max(), 3))

    # Assign the best model based on the F1 Score
    best_model = metrics_tuned_df.loc['F1 Score'].idxmax()

    # Print the best model and its name
    print(f"Best model: ", best_model)
    
    # From hyperparameters_classification, print the hyperparameters for the best_model
    print(f"Hyperparameters for the best model ({best_model}): {hyperparameters_classification[best_model]}")

    logging.info("--- Classification Baseline Models - Completed")
    
    # With the best model, print the confusion matrix
    # Get the best model
    best_model = metrics_tuned_df.loc['F1 Score'].idxmax()
    # Get the best model object
    best_model_object = models[best_model]
    # Create a pipeline with the best model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', best_model_object)])
    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)
    # Predict the target on the testing data
    y_pred = pipeline.predict(X_test)
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Create a confusion matrix display
    cmd = ConfusionMatrixDisplay(cm, display_labels=pipeline.classes_)
    # Plot the confusion matrix
    cmd.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {best_model}')
    #plt.show()
    plt.savefig(f'./data/Tuned_confusion_matrix_{best_model}.png', dpi=300, bbox_inches='tight')  # Save the plot as a PNG file
    plt.close()  # Close the plot to free up memory
    
    return metrics_tuned_df
# def classification_models_hyperparameters(df, models, hyperparameters_classification):


