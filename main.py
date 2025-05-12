# Requires Python 3.11.0

# Standard library imports
import logging
import os

# Third-party imports
import pandas as pd
import yaml
from sklearn.utils._testing import ignore_warnings
import sqlite3 # for working with SQLite database

# Local application/library specific imports
from src.data_preparation import DataCleaning

# Import libraries required for classification models
from src.model_training import classification_setup
from src.model_training import classification_setup_results
from src.model_training import classification_preprocessor
from src.model_training import classification_preprocessor_results
from src.model_training import classification_baseline
from src.model_training import classification_models_hyperparameters



logging.basicConfig(level=logging.INFO)

# To assist in EDA process, the following functions are defined:
# Prints separator to console for better readability
def print_separator():
    """
    This function performs print a separator line to the console.
    """
    print("."*30)

def access_db(config, database_path):
    # Open database and create a pandas dataframe
    # Connect to the SQLite database using the path from config.yaml
    conn = sqlite3.connect(database_path)

    # Check if the connection was successful
    if conn is None:
        print("Error: Unable to connect to the database.")
        print_separator()
    else:
        print("Database connection successful.")
        print_separator()

        # Read Table Name from database
        table_name_df = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)

        # Check if table_name_df is empty
        if table_name_df.empty:
            print("Error: No tables found in the database.")
        else:
            # Extract Table Name from database
            table_name = table_name_df['name'].values[0]
            # Print Table Name of database
            print('Database table name: ', table_name)
            # Create a pandas dataframe from database
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn) # Load the dataset

    # Close database connection
    conn.close()

    # Save the dataframe to a CSV file
    df.to_csv('data/bmarket.csv', index=False) # Save the dataframe to a CSV file

def read_config(config_path):
    """
    This function reads the configuration file and returns the database path.
    """
    # Check if the file exists
    if os.path.exists(config_path):
        # Load configuration from a YAML file as encoding='utf-8'
        with open(config_path, 'r', encoding='utf-8') as file: # open the config.yaml file
            config = yaml.safe_load(file) # load the configuration

        # Get "database_path" from config.yaml
        if 'database_path' in config:
            database_path = config['database_path'] # get the database path
            print('The database path is:', database_path,"n")
        else:
            print("Error: 'database_path' key is missing in the configuration.")        
        print_separator()
        
        # Get "classification_target" from config.yaml
        if 'classification_target' in config:
            target_column_discrete = config['classification_target'] 
            print('The target column (Classification) is:', target_column_discrete)
        else:
            print("Error: 'classification_target' key is missing in the configuration.")
    else:
        print(f"Error: The file '{config_path}' does not exist. Please check the file path.")
        
    return config, database_path

@ignore_warnings(category=Warning)
def main():
    print_separator()
    print("Welcome to the AIAP program!")
    print_separator()

    # Define the path to the config.yaml file
    config_path = './src/config.yaml'

    # Read the configuration file - config.yaml
    config, database_path = read_config(config_path)

    # Load the data from the database
    access_db(config, database_path)
    df = pd.read_csv('data/bmarket.csv')

    # Data Cleaning
    data_cleaning = DataCleaning(config)
    f = data_cleaning.clean_data(df)
    print_separator()
    logging.info("--- Cleaned data - Display dataframe")
    print_separator()
    print(df.info())
     
    ##### Classification_setup #####
    # Step 1 - Call Classification_setup
    target_column_discrete, classification_models, classification_model_classes, hyperparameters_classification, classification_metrics = classification_setup(config)
    # Display Classification_setup_results
    classification_setup_results(target_column_discrete, classification_models, hyperparameters_classification, classification_metrics, config)

   ##### Classification_preprocessor #####
    # Step 2- Call Classification_preprocessor
    target_column_discrete, numerical_columns, categorical_columns, X_train, X_test, y_train, y_test, preprocessor = classification_preprocessor(df, config)
    # Call Classification_preprocessor_result
    classification_preprocessor_results(target_column_discrete, df, numerical_columns, categorical_columns, X_train, X_test, y_train, y_test, preprocessor)
    
    ##### Part 1 - Classification modeling - Baseline #####
    # Step 3 - Perform classification modeling - Baseline
    classification_baseline(df, classification_models, preprocessor, X_train, X_test, y_train, y_test)
    
    ##### Part 2 - Classification modeling -  with hyperparameters #####
    # Step 4 - Perform classification modeling - with hyperparameters tuning
    classification_models_hyperparameters(df, classification_models, hyperparameters_classification, preprocessor, X_train, X_test, y_train, y_test, config)
    
    logging.info("--- End of the program")
    print_separator()
 
if __name__ == "__main__":
    main()