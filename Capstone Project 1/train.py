import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def preparing_data():
    # Wczytaj dane z URL
    df = pd.read_csv("https://raw.githubusercontent.com/adamfdnb/course-mlzoomcamp2023/main/Capstone%20Project%201/data/water_potability.csv")
    
    # Zamień nazwy kolumn na małe litery i zastąp spacje podkreśleniami
    df.columns = df.columns.str.replace(' ', '_').str.lower()

    # Uzupełnij brakujące dane w kolumnie 'ph'
    df = fill_missing_data_ph(df)

    # Uzupełnij brakujące dane w kolumnie 'sulfate'
    df = fill_missing_sulfate_with_mean(df)

    # Uzupełnij brakujące dane w kolumnie 'trihalomethanes'
    df = fill_missing_tmh_with_mean(df)

    return df

def fill_missing_data_ph(df):
    # Utworzenie df_mean jako kopii oryginalnego DataFrame
    # df_mean_fill = df.copy()

    # Podział danych na kategorie 'hardness_threshold'
    df['hardness_threshold'] = pd.cut(df['hardness'], bins=[-float('inf'), 89, 179, 269, float('inf')],
                                           labels=['Below 89', '90-179', '180-269', 'Above 268'])

    # Warunki dla uzupełniania danych
    condition_1 = (df['hardness_threshold'] == 'Below 89') & (df['potability'] == 0)
    condition_2 = (df['hardness_threshold'] == 'Below 89') & (df['potability'] == 1)
    condition_3 = (df['hardness_threshold'] == '90-179') & (df['potability'] == 0)
    condition_4 = (df['hardness_threshold'] == '90-179') & (df['potability'] == 1)
    condition_5 = (df['hardness_threshold'] == '180-269') & (df['potability'] == 0)
    condition_6 = (df['hardness_threshold'] == '180-269') & (df['potability'] == 1)
    condition_7 = (df['hardness_threshold'] == 'Above 268') & (df['potability'] == 0)
    condition_8 = (df['hardness_threshold'] == 'Above 268') & (df['potability'] == 1)

    # Uzupełnienie danych w kolumnie 'ph' na podstawie warunków
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
    # Dodaj kategorię 'ph_category'
    df['ph_category'] = np.where(df['ph'] < 7, 'Below 7', '7 and Above')

    # Uzupełnij brakujące wartości w kolumnie 'sulfate' średnimi dla poszczególnych kategorii
    df['sulfate'] = df.groupby(['potability', 'ph_category'])['sulfate'].transform(lambda x: x.fillna(x.mean()))

    return df

def fill_missing_tmh_with_mean(df):
    # Dodaj kategorię 'ph_category'
    df['ph_category'] = np.where(df['ph'] < 7, 'Below 7', '7 and Above')

    # Uzupełnij brakujące wartości w kolumnie 'trihalomethanes' średnimi dla poszczególnych kategorii
    df['trihalomethanes'] = df.groupby(['potability', 'ph_category'])['trihalomethanes'].transform(lambda x: x.fillna(x.mean()))

    return df

def train(df):
    # Podziel DataFrame 'df' na trzy podzbiory: treningowy, walidacyjny i testowy.
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

    columns_to_drop = ['hardness_threshold', 'ph_category']

    X_train = df_train.drop(['potability'] + columns_to_drop, axis=1)
    X_val = df_val.drop(['potability'] + columns_to_drop, axis=1)
    
    y_train = df_train['potability']
    y_val = df_val['potability']
  
    # Najlepsze hiperparametry uzyskane z kroswalidacji
    best_hyperparameters = {'eta': 0.1, 'max_depth': 6, 'min_child_weight': 3, 'n_estimators': 25, 'subsample': 0.7}

    # Utwórz model XGBoost z najlepszymi hiperparametrami
    model_xgbCl = XGBClassifier(
        learning_rate=best_hyperparameters['eta'],
        max_depth=best_hyperparameters['max_depth'],
        min_child_weight=best_hyperparameters['min_child_weight'],
        n_estimators=best_hyperparameters['n_estimators'],
        subsample=best_hyperparameters['subsample']
    )

    # Trenuj model na danych treningowych
    model_xgbCl.fit(X_train, y_train)

    # Przewidywanie na danych walidacyjnych
    y_pred = model_xgbCl.predict(X_val)

    # Oblicz dokładność na zbiorze walidacyjnym
    val_accuracy = accuracy_score(y_val, y_pred)
    print(f"Accuracy on the validation dataset: {val_accuracy:.5f}")
    
    # Zapisz wytrenowany model do pliku
    output_file = "model_wpp.pkl"
    with open(output_file, "wb") as f_out:
        pickle.dump(model_xgbCl, f_out)

    # Pobierz pełną ścieżkę do zapisanego pliku modelu
    output_filepath = os.path.abspath(output_file)
    print(f"Saved the model as: {output_file}")
    print(f"Full path to the saved model: {output_filepath}")

if __name__ == "__main__":
    df = preparing_data()
    train(df)
