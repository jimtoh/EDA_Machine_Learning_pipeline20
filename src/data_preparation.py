# Requires Python 3.11.0
# Standard library imports
import logging
import os
from typing import Any, Dict
# For logging purposes
logging.basicConfig(level=logging.INFO)
# For data manipulation
import pandas as pd
# For regular expression operations
import re
# For SQLite database operations
import sqlite3
# For YAML file operations
import yaml
# For numerical operations
import numpy as np
# For printing formatted text
from prompt_toolkit import print_formatted_text
# For ignoring warnings
from sklearn.utils._testing import ignore_warnings
# For data visualization
import matplotlib.pyplot as plt
import seaborn as sns
# For ignoring warnings
@ignore_warnings(category=Warning)


# Prints separator to console for better readability
def print_separator():
    """
    This function performs print a separator line to the console.
    """
    print("."*30)

# Put the following as a routine to check for outliers
def outliers_for_removal(df, column, threshold,lower_bound=False, upper_bound=True):
    """
    This function checks for outliers in a given column of the dataframe using the IQR method.
    """
    Q1 = df[column].quantile(0.25) # 25th percentile
    Q3 = df[column].quantile(0.75) # 75th percentile
    IQR = Q3 - Q1 # Interquartile range

    if lower_bound == True:
        # Calculate lower bound for outliers
        lower_bound = Q1 - threshold * IQR
    else:
        # Calculate lower bound for outliers
        lower_bound = min(df[column])

    # Print the lower bound for outliers
    print(f"Lower bound for outliers in {column}: {lower_bound}")

    if upper_bound == True:
        # Calculate upper bound for outliers
        upper_bound = Q3 + threshold * IQR
    else:
        # Calculate upper bound for outliers
        upper_bound = max(df[column])

    # Print the upper bound for outliers
    print(f"Upper bound for outliers in {column}: {upper_bound}")

    # Remove outliers from the dataframe using lower and upper bounds
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print(f"Removed outliers from the column: {column}, with threshold: {threshold}")
    print_separator()
    return df

# Feature Engineering for Age by creating a new feature 'Age Group'
def age_group(age):
    """
    This function categorizes age into groups.
    """
    if age < 30:
        return 'Twenties'
    elif age < 40:
        return 'Thirties'
    elif age < 50:
        return 'Forties'
    elif age < 60:
        return 'Fifties'
    else:
        return 'Above Sixties'

# Feature Engineering for Education are mapped into 3 groups, as specified by Education_Group in the config.yaml file:

def education_level(education, config):
    """
    This function categorizes the education level into groups,
    as specified by Education_Group in the config.yaml file.
    """
    if config and 'Education_Group' in config and education in config['Education_Group']['Primary']:
        return 'primary'
    elif education in config['Education_Group']['Secondary']:
        return 'secondary'
    elif education in config['Education_Group']['Tertiary']:
        return 'tertiary'
    else:
        return 'others'

# Feature Engineering for Occupation are mapped into 3 groups, as specified by Occupation_Group in the config.yaml file:

def Occupation_level(occupation, config):
    """
    This function categorizes the occupation level into groups, 
    as specified by Occupation_Group in the config.yaml file.
    """
    if occupation in config['Occupation_Group']['Blue_Collar']:
        return 'blue collar'
    elif occupation in config['Occupation_Group']['White_Collar']:
        return 'white collar'
    elif occupation in config['Occupation_Group']['Grey_Collar']:
        return 'grey collar'
    else:
        return 'other'

# Create a class DataCleaning
class DataCleaning:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Show info()
        logging.info("Data cleaning - showing initial data")
        print_separator() 

        # Define the path to the config.yaml file
        config_path = './src/config.yaml'

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
            
            # Get "regression_target" from config.yaml
            if 'regression_target' in config:
                target_column_continuous = config['regression_target']
                print('The target column (Regression) is:', target_column_continuous)
            else:
                print("Error: 'regression_target' key is missing in the configuration.")
                
            print_separator()
            # Get "classification_target" from config.yaml
            if 'classification_target' in config:
                target_column_discrete = config['classification_target'] 
                print('The target column (Classification) is:', target_column_discrete)
            else:
                print("Error: 'classification_target' key is missing in the configuration.")
        else:
            print(f"Error: The file '{config_path}' does not exist. Please check the file path.")

        print_separator()
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
        
        # The database is loaded properly and the first 5 rows of the dataframe are printed to check 
        # if the data is read correctly. Also it gives a sense of the data's structure and content
        df.head()
        print_separator()
        # Provides a concise summary and display the number of non-null entries, data type, the overall memory usage.
        # Understanding the shape, size, datatypes of the dataset.
        df.info()
        print_separator()
        # Calculate initial number of rows and columns in dataframe
        initial_row_count = len(df)
        # Print Number of Rows in dataframe
        print(f"Initial number of rows: {initial_row_count}")
        initial_column_count = len(df.columns)
        # Print Number of Columns in dataframe
        print(f"Initial number of columns: {initial_column_count}")
        print_separator()
        # Reload config.yaml file
        with open(config_path, 'r', encoding='utf-8') as file: # open the config.yaml file
            config = yaml.safe_load(file) # load the configuration

        # If "ID_column" from config.yaml is not empty, then get the ID column name and 
        # remove it from the dataframe
        if 'ID_column' in config:
            ID_column = config['ID_column'] # get the ID column name
            if ID_column in df.columns:
                # Since there is an ID column, we can remove this column from the dataframe
                df.drop(columns=[ID_column], inplace=True)
                print(f"Removed '{ID_column}' column from the dataframe.")
            else:
                print(f"'{ID_column}' column not found in the dataframe. No column removed.")

        # Remove duplicate rows from the dataframe
        df.drop_duplicates(inplace=True)
        # Print the number of duplicate rows after removal
        print("Number of duplicates after removal: ", df.duplicated().sum())
        print_separator()
        # To correct the data format, for all categorical features, convert the values to lowercase
        # for all categorical features, convert the values to lowercase
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.lower()
        print ("Converted categorical values to lowercase.")
        print_separator()
        # Remove duplicate rows from the dataframe
        df.drop_duplicates(inplace=True)
        # Print the number of duplicate rows after removal
        print("Number of duplicates after removal: ", df.duplicated().sum())
        print_separator()
        # In col Contact Method, the value 'cell' is rename as 'cellular'
        df['Contact Method'] = df['Contact Method'].replace({'cell': 'cellular'})
        print("Renamed 'cell' to 'cellular' in Contact Method column.")
        print_separator()
        # Get the "data_type_correction" from config.yaml
        if 'data_type_correction' in config:
            # For each data_type_correction, if it is an object, convert it to a int by removing the string portion
            df[config['data_type_correction']] = df[config['data_type_correction']].str.extract('(\d+)').astype(int)
            print("Converted data types as per the configuration.")
        else:
            print("Error: 'data_type_correction' key is missing in the configuration.")
        print_separator()
        # Replace values greater than 100 in the Age column with mean of the column taken from only when Age is less than 100
        if 'Age' not in df.columns:
            print("Error: 'Age' column not found in the dataframe.")
        else:
            mean_age = df[df['Age'] < 100]['Age'].mean()
            df.loc[df['Age'] > 100, 'Age'] = mean_age.astype(df['Age'].dtype)
            print("Impute Age values that are > 100, with mean from the Age range < 100.")
     
        print_separator()
        # For each column, if the percentage of 'unknown' values is less than 5%, remove the rows from the dataframe
        for col in df.columns:
            if df[col].dtype == 'object':
                unknown_count = df[col].str.contains('unknown', case=False, na=False).sum()
                if unknown_count > 0:
                    percentage_unknown = round(unknown_count/df.shape[0]*100, 2)
                    if percentage_unknown < 5:
                        df = df[~df[col].str.contains('unknown', case=False, na=False)]
                        print(f"Removed rows with 'unknown' values from column: {col}")
                    else:
                        print(f"Percentage of 'unknown' values in column {col} is greater than 5%. No rows removed.")
                        print_separator()
        print_separator()
        # Since the value 'yes' in the Credit Default column is not useful, we can remove it from the dataframe
        df = df[df['Credit Default'] != 'yes']
        print("Removed 'yes' values from the Credit Default column.")
        print_separator()
        # Rename the 'unknown' values to 'undisclosed' in the Credit Default column, for better understanding of the data
        df['Credit Default'] = df['Credit Default'].replace('unknown', 'undisclosed')
        print("Renamed 'unknown' values to 'undisclosed' in the Credit Default column.")
        print_separator()
        # For Campaign Calls, convert the negative values to positive values
        df['Campaign Calls'] = df['Campaign Calls'].abs()
        print("Converted negative values to positive values in Campaign Calls column.")
        print_separator()

        # Copy 'Previous Contact Days' as 'Previous Contact'
        df['Previous Contact'] = df['Previous Contact Days']
        # Print the copy is done
        print("Copied 'Previous Contact Days' as 'Previous Contact'")
        # For 'Previous Contact', replace row value '999' with '0'
        df['Previous Contact'] = df['Previous Contact'].replace(999, 0)
        # Change the data type of 'Previous Contact' to an object
        df['Previous Contact'] = df['Previous Contact'].astype('object')
        # For each non-zero value in the 'Previous Contact' column, change the value to '1'
        df.loc[df['Previous Contact'] != 0, 'Previous Contact'] = 1
        # Change the data type of 'Previous Contact' to an object
        df['Previous Contact'] = df['Previous Contact'].astype('object')
        # Rename each value in the 'Previous Contact' column from 0 to 'no prior contact'
        df['Previous Contact'] = df['Previous Contact'].replace({0: 'no prior contact'})
        # Rename each value in the 'Previous Contact' column from 1 to 'prior contact'
        df['Previous Contact'] = df['Previous Contact'].replace({1: 'prior contact'})

        # Drop the column Previous Contact Days, if it exists
        if 'Previous Contact Days' in df.columns:
            df.drop(columns=['Previous Contact Days'], inplace=True)
            print("Dropped Previous Contact Days column.")
        else:
            print("Previous Contact Days column not found. No column dropped.")

        print_separator()
        # Dropping column Contact Method that are not useful for the analysis
        if 'Contact Method' in df.columns:
            df.drop(columns=['Contact Method'], inplace=True)
            print("Removed 'Contact Method' column from the dataframe.")
        else:
            print("Column 'Contact Method' not found in the dataframe. No column removed.")

        print_separator()
        # Use IQR method to remove outliers from the Campaign Calls column, using threshold_for_outlier_removal from config.yaml
        if 'threshold_for_outlier_removal' in config:
            threshold = config['threshold_for_outlier_removal']['Campaign Calls']  # Extract the threshold for 'Campaign Calls'
            df = outliers_for_removal(df, 'Campaign Calls', threshold=threshold, lower_bound=False, upper_bound=True)
            print("Removed outliers from the Campaign Calls column using IQR method.")
        else:
            print("Error: 'threshold_for_outlier_removal' key is missing in the configuration.")

        print_separator()
        # For Marital Status, rename 'divorced' and 'single' as 'not married'
        df['Marital Status'] = df['Marital Status'].replace({'divorced': 'not married', 'single': 'not married'})
        print("Renamed 'divorced' and 'single' as 'not married' in Marital Status column.")

        print_separator()
        # Check if the 'Age' column exists in the dataframe before creating the 'Age Group' feature
        if 'Age' in df.columns:
            # Call the function to create a new feature 'Age Group'
            df['Age Group'] = df['Age'].apply(age_group)
            print("Added 'Age Group' column to the dataframe.")

            # Drop the 'Age' column from the dataframe
            df.drop(columns=['Age'], inplace=True)

        else:
            print("The 'Age' column does not exist in the dataframe.")

        # Check the datatype of the 'Age Group' column
        print("Data type of 'Age Group' column: ", df['Age Group'].dtype)

        print_separator()
        if 'On Loan' not in df.columns:
            if 'Housing Loan' in df.columns and 'Personal Loan' in df.columns:
                # Create the 'On Loan' column
                # New feature 'On Loan' will have a 'yes' if Housing Loan OR Personal Loan has a 'yes'
                # New feature 'On Loan' will have a 'no' if Housing Loan AND Personal Loan has a 'no'
                # New feature 'On Loan' will have a 'unknown' if Housing Loan AND Personal Loan has a 'unknown'
                df['On Loan'] = np.where((df['Housing Loan'] == 'yes') | (df['Personal Loan'] == 'yes'), 'yes', 
                    np.where((df['Housing Loan'] == 'no') & (df['Personal Loan'] == 'no'), 'no', 'unknown'))
                
                # Print the status of the 'On Loan' column
                print("Created the 'On Loan' column.")

                # Drop the 'Housing Loan' and 'Personal Loan' columns from the dataframe
                df.drop(columns=['Housing Loan', 'Personal Loan'], inplace=True)
                print("Removed 'Housing Loan' and 'Personal Loan' columns from the dataframe.")
        else:
            print("The 'On Loan' column already exists in the dataframe. No column created.")

        # Rename the 'unknown' value on 'On Loan' column to 'undisclosed'
        df['On Loan'] = df['On Loan'].replace('unknown', 'undisclosed')
        print("Renamed 'unknown' value to 'undisclosed' in 'On Loan' column.")

        print_separator()
        # Check if the 'Education Level' column exists in the dataframe before creating the 'Education Group' feature as in config.yaml
        if 'Education Level' in df.columns:
            # Call the function to create a new feature 'Education Group'
            df['Education Group'] = df['Education Level'].apply(lambda education: education_level(education, config))
            print("Added 'Education Group' column to the dataframe.")

            # Drop the 'Education Level' column from the dataframe
            df.drop(columns=['Education Level'], inplace=True)
            print("Removed 'Education Level' column from the dataframe.")
        else:
            print("The 'Education Level' column does not exist in the dataframe.")

        print_separator()
        # Check if the 'Occupation' column exists in the dataframe before creating the 'Occupation Group' feature
        if 'Occupation' in df.columns:
            # Call the function to create a new feature 'Occupation Group'
            df['Occupation Group'] = df['Occupation'].apply(lambda occupation: Occupation_level(occupation, config))
            print("Added 'Occupation Group' column to the dataframe.")

            # Drop the 'Occupation' column from the dataframe
            df.drop(columns=['Occupation'], inplace=True)
            print("Removed 'Occupation' column from the dataframe.")
        else:
            print("The 'Occupation' column does not exist in the dataframe.")

        print_separator()
        # Remove duplicate rows from the dataframe
        df.drop_duplicates(inplace=True)
        # Print the number of duplicate rows after removal
        print("Number of duplicates after removal: ", df.duplicated().sum())

        # Calculate final number of rows and columns in the dataframe
        final_row_count_at_eda = len(df)
        final_column_count_at_eda = len(df.columns)

        print_separator()
        print(f"Initial number of columns in the dataframe: {initial_column_count}\n"
            f"After EDA, final number of columns in the dataframe: {final_column_count_at_eda}\n"
            f"Initial Number of rows: {initial_row_count}\n"           
            f"After EDA, final Number of rows: {final_row_count_at_eda}\n"           
            f"Percentage data loss : {((initial_row_count - final_row_count_at_eda) / initial_row_count) * 100:.1f}%")

        # The data is now ready for further analysis and modeling.
        # Save the cleaned data to a new csv - agri_cleaned_final.csv
        df.to_csv('./data/eda_completed.csv', index=False)

        # EDA completed
        print_separator()
        print("Exploratory Data Analysis (EDA) completed")

        return df
