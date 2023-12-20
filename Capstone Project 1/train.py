import pandas as pd
import numpy as np
import pickle
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Ustaw katalog roboczy na folder, gdzie znajduje się skrypt
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def preparing_data():
    # Load data from the URL
    df = pd.read_csv("https://raw.githubusercontent.com/adamfdnb/course-mlzoomcamp2023/main/Capstone%20Project%201/data/water_potability.csv")

    # Clean column names (lowercase and replace spaces with underscores)
    df.columns = df.columns.str.replace(' ', '_').str.lower()

    # Fill missing data in the 'ph' column
    df = fill_missing_data_ph(df)

    # Fill missing data in the 'sulfate' column
    df = fill_missing_sulfate_with_mean(df)

    # Fill missing data in the 'trihalomethanes' column
    df = fill_missing_tmh_with_mean(df)

    return df

def fill_missing_data_ph(df):
    
    df['hardness_threshold'] = pd.cut(df['hardness'], bins=[-float('inf'), 89, 179, 269, float('inf')],
                                           labels=['Below 89', '90-179', '180-269', 'Above 268'])

    # Define conditions for data imputation
    condition_1 = (df['hardness_threshold'] == 'Below 89') & (df['potability'] == 0)
    condition_2 = (df['hardness_threshold'] == 'Below 89') & (df['potability'] == 1)
    condition_3 = (df['hardness_threshold'] == '90-179') & (df['potability'] == 0)
    condition_4 = (df['hardness_threshold'] == '90-179') & (df['potability'] == 1)
    condition_5 = (df['hardness_threshold'] == '180-269') & (df['potability'] == 0)
    condition_6 = (df['hardness_threshold'] == '180-269') & (df['potability'] == 1)
    condition_7 = (df['hardness_threshold'] == 'Above 268') & (df['potability'] == 0)
    condition_8 = (df['hardness_threshold'] == 'Above 268') & (df['potability'] == 1)

    # Impute missing data in the 'ph' column based on conditions
    df.loc[condition_1, 'ph'] = df.loc[condition_1, 'ph'].fillna(df.loc[condition_1, 'ph'].mean())
    df.loc[condition_2, 'ph'] = df.loc[condition_2, 'ph'].fillna(df.loc[condition_2, 'ph'].mean())
    df.loc[condition_3, 'ph'] = df.loc[condition_3, 'ph'].fillna(df.loc[condition_3, 'ph'].mean())
    df.loc[condition_4, 'ph'] = df.loc[condition_4, 'ph'].fillna(df.loc[condition_4, 'ph'].mean())
    df.loc[condition_5, 'ph'] = df.loc[condition_5, 'ph'].fillna(df.loc[condition_5, 'ph'].mean())
    df.loc[condition_6, 'ph'] = df.loc[condition_6, 'ph'].fillna(df.loc[condition_6, 'ph'].mean())
    df.loc[condition_7, 'ph'] = df.loc[condition_7, 'ph'].fillna(df.loc[condition_7, 'ph'].mean())
    df.loc[condition_8, 'ph'] = df.loc[condition_8, 'ph'].fillna(df.loc[condition_8, 'ph'].mean())

    return df

def fill_missing_sulfate_with_mean(df):
    # Add 'ph_category' based on 'ph'
    df['ph_category'] = np.where(df['ph'] < 7, 'Below 7', '7 and Above')

    # Fill missing 'sulfate' values based on mean values for categories
    df['sulfate'] = df.groupby(['potability', 'ph_category'])['sulfate'].transform(lambda x: x.fillna(x.mean()))

    return df

def fill_missing_tmh_with_mean(df):
    # Add 'ph_category' based on 'ph'
    df['ph_category'] = np.where(df['ph'] < 7, 'Below 7', '7 and Above')

    # Fill missing 'trihalomethanes' values based on mean values for categories
    df['trihalomethanes'] = df.groupby(['potability', 'ph_category'])['trihalomethanes'].transform(lambda x: x.fillna(x.mean()))

    return df

def train(df):
    # Split the DataFrame into train, validation, and test sets
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

    columns_to_drop = ['hardness_threshold', 'ph_category']

    # Separate features and target variable
    X_train = df_train.drop(['potability'] + columns_to_drop, axis=1)
    X_val = df_val.drop(['potability'] + columns_to_drop, axis=1)
    
    y_train = df_train['potability']
    y_val = df_val['potability']
  
    # Best hyperparameters obtained from cross-validation
    best_hyperparameters = {'eta': 0.1, 'max_depth': 6, 'min_child_weight': 3, 'n_estimators': 25, 'subsample': 0.7}

    # Create XGBoost model with the best hyperparameters
    xgb_model = XGBClassifier(
        learning_rate=best_hyperparameters['eta'],
        max_depth=best_hyperparameters['max_depth'],
        min_child_weight=best_hyperparameters['min_child_weight'],
        n_estimators=best_hyperparameters['n_estimators'],
        subsample=best_hyperparameters['subsample'],
        use_label_encoder=False

    )

    # Train the model on the training data
    xgb_model.fit(X_train, y_train)

    # Predict on the validation set
    y_pred = xgb_model.predict(X_val)

    # Calculate accuracy on the validation dataset
    val_accuracy = accuracy_score(y_val, y_pred)
    print(f"Accuracy on the validation dataset: {val_accuracy:.5f}")
    

    # Download Booster
    booster = xgb_model.get_booster()

    # Save mode
    booster.save_model('model_wpp.model')
    # Get the current working directory
    current_directory = os.getcwd()

    # Define the model filename
    model_filename = 'model_wpp.model'

    # Get the full path to the saved file
    output_filepath = os.path.abspath(model_filename)

    # Display the saved model's name and its full path
    model_path = os.path.join(current_directory, model_filename)
    print(f"Model saved as: {model_filename}")
    print(f"Full path to the model: {model_path}")
    print("\n")

if __name__ == "__main__":
    # Load and preprocess the data
    df = preparing_data()
    # Train the XGBoost model
    train(df)
