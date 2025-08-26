import pandas as pd
import numpy as np
import os

def load_and_clean_data(file_path):
    """
    Loads the Telecom customer churn dataset, cleans the 'TotalCharges' column,
    and returns a cleaned DataFrame.

    Args:
        file_path (str): The path to the raw CSV file.

    Returns:
        pd.DataFrame: A DataFrame with the 'TotalCharges' column cleaned and
                      rows with missing values removed.
    """
    # Load the dataset
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    
    # Handle the 'TotalCharges' column
    # It contains empty strings, which prevents it from being a numeric type.
    # We replace these with NaN and then convert the column to float.
    df['TotalCharges'] = df['TotalCharges'].replace(' ', 0)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

    return df

if __name__ == "__main__":
    # Define the file path for the raw data
    raw_data_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    # Define the file path for the cleaned data
    cleaned_data_path = os.path.join("data","raw")

    
    # Load and clean the data
    cleaned_df = load_and_clean_data(raw_data_path)
    
    if cleaned_df is not None:
        try:
            # Save the cleaned DataFrame to a new CSV file
            os.makedirs(cleaned_data_path)
            cleaned_df.to_csv(os.path.join(cleaned_data_path,"cleaned_telco_churn.csv"), index=False)
            
            print(f"Data has been successfully loaded, cleaned, and saved to '{cleaned_data_path}'")
            print("\nCleaned Data Info:")
            cleaned_df.info()
        except Exception as e:
            raise Exception(f'An error occured : {e}')
