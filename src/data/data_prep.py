import pandas as pd
import os

def preprocess_data(file_path):
    """
    Loads the cleaned Telco customer churn dataset, performs one-hot encoding
    on categorical variables, and prepares the data for machine learning.

    Args:
        file_path (str): The path to the cleaned CSV file.

    Returns:
        tuple: A tuple containing:
               - pd.DataFrame X: The features DataFrame.
               - pd.Series y: The target variable Series.
    """
    # Load the cleaned dataset
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None

    # Drop customerID since it is an identifier and not a feature
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Convert the 'Churn' column to a binary format (1 for 'Yes', 0 for 'No')
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Identify categorical columns to be one-hot encoded
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Perform one-hot encoding using pd.get_dummies()
    # The 'drop_first=True' argument is used to avoid multicollinearity.
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Separate features (X) and target variable (y)
    if 'Churn' in df_encoded.columns:
        X = df_encoded.drop('Churn', axis=1)
        y = df_encoded['Churn']
    else:
        print("The 'Churn' column was not found in the DataFrame after encoding.")
        return None, None
    
    return X, y

if __name__ == "__main__":
    # Define the file path for the cleaned data
    cleaned_data_path = 'data/raw/cleaned_telco_churn.csv'
    
    # Define the output file paths for the prepared data
    processed_data_path = 'data/processed'

    # Preprocess the data
    X_features, y_target = preprocess_data(cleaned_data_path)
    
    if X_features is not None and y_target is not None:
        try:
            # Save the processed data to CSV files
            if not os.path.isdir(processed_data_path):
                os.makedirs(processed_data_path)
            X_features.to_csv(os.path.join(processed_data_path, 'x_features.csv'), index=False)
            y_target.to_csv(os.path.join(processed_data_path, 'y_target.csv'), index=False)
            
            print("Data pre-processing successful.")
            print(f"\nFeatures (X) saved to '{os.path.join(processed_data_path, 'x_features.csv')}'")
            print(f"Target (y) saved to '{os.path.join(processed_data_path, 'x_target.csv')}'")
            
            print("\nShape of features (X):", X_features.shape)
            print("Shape of target (y):", y_target.shape)
        except Exception as e:
            raise Exception(f'An error occured : {e}')