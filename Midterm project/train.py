import pickle
import numpy as np
import pandas as pd
import os

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def preparing_data():
     # Importing a dataset and creating a DataFrame
    df_init = pd.read_csv("https://raw.githubusercontent.com/adamfdnb/course-mlzoomcamp2023/main/Midterm%20project/dataset/milknew.csv")
    
    # Converting string columns to lowercase and replacing spaces with underscores
    df_init.columns = df_init.columns.str.replace(" ", "_").str.lower()
    string_columns = list(df_init.dtypes[df_init.dtypes == "object"].index)

    for col in string_columns:
        df_init[col] = df_init[col].str.lower().str.replace(" ", "_")

    # Rename the "temprature" column to "temperature"
    df_init.rename(columns={"temprature": "temperature"}, inplace=True)

    # Replace values in the "grade" column of DataFrame 'df_init' based on certain conditions.
    df_init.loc[df_init["grade"] == "high", "grade"] = 2
    df_init.loc[df_init["grade"] == "medium", "grade"] = 1
    df_init.loc[df_init["grade"] == "low", "grade"] = 0

    # Convert the "grade" column to integer
    df_init["grade"] = df_init["grade"].astype(int)

    return df_init

def train(df):
    # Splitting the DataFrame 'df' into three subsets: training, validation, and testing.
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

    y_train = df_train["grade"]
    y_test = df_test["grade"]

    del df_train["grade"]
    del df_test["grade"]

    # preparation of features_matrices:
    X_train = df_train
    X_test = df_test

    # Train the RandomForestClassifier model with chosen hyperparameters
    model_rfc = RandomForestClassifier(n_estimators=5, max_depth=5)
    model_rfc.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model_rfc.predict(X_test)

    # Calculate accuracy on the test dataset
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on the test dataset: {test_accuracy:.5f}")

    # Save the trained model to a file
    output_file = "model_mqp.pkl"
    with open(output_file, "wb") as f_out:
        pickle.dump(model_rfc, f_out)

    # Get the full path to the saved model file
    output_filepath = os.path.abspath(output_file)
    print(f"Saved the model as: {output_file}")
    print(f"Full path to the saved model: {output_filepath}")

if __name__ == "__main__":
    df = preparing_data()
    train(df)