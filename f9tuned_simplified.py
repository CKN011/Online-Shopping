"""
Vereinfachte F9Tuned-Implementierung für die E-Commerce Kaufabsichtsprognose
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def prepare_dataset(df):
    """
    Bereitet den Datensatz für das F9Tuned-Modell vor
    
    Args:
        df (pandas.DataFrame): Der zu verarbeitende Datensatz
        
    Returns:
        pandas.DataFrame: Vorbereiteter Datensatz
    """
    # Kopieren, um das Original nicht zu verändern
    dataset = df.copy()
    
    # Zielvariable und Boolesche Spalte
    if 'Revenue' in dataset.columns and dataset['Revenue'].dtype == bool:
        dataset['Revenue'] = dataset['Revenue'].astype(int)
    
    if 'Weekend' in dataset.columns and dataset['Weekend'].dtype == bool:
        dataset['Weekend'] = dataset['Weekend'].astype(int)
    
    return dataset

def select_features(df):
    """
    Wählt die relevanten Features für das F9Tuned-Modell aus
    
    Args:
        df (pandas.DataFrame): Der vorbereitete Datensatz
        
    Returns:
        pandas.DataFrame: Datensatz mit ausgewählten Features
    """
    # Vorgegebene Features direkt auswählen
    selected_features = [
        'Informational', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
        'Month', 'OperatingSystems', 'VisitorType', 'Weekend'
    ]
    
    # Prüfen, welche der ausgewählten Features im Datensatz existieren
    available_features = [f for f in selected_features if f in df.columns]
    
    # Datensatz auf die verfügbaren ausgewählten Features verkleinern
    X_selected = df[available_features].copy()
    
    # Kategorische Spalten im verkleinerten Datensatz identifizieren
    categorical_cols = []
    for col in ['Month', 'VisitorType']:
        if col in X_selected.columns:
            if X_selected[col].dtype == 'object' or col in ['Month', 'VisitorType']:
                categorical_cols.append(col)
    
    # Kategorische Features mit One-Hot-Encoding umwandeln
    if categorical_cols:
        X_selected = pd.get_dummies(X_selected, columns=categorical_cols, drop_first=True)
    
    return X_selected

def train_f9tuned_model(df, model_name='randomforest'):
    """
    Trainiert das F9Tuned-Modell (vereinfachte Version mit RandomForest)
    
    Args:
        df (pandas.DataFrame): Der vollständige Datensatz mit Revenue-Spalte
        model_name (str, optional): Modelltyp (wird hier ignoriert, da nur RandomForest verwendet wird)
        
    Returns:
        dict: Dictionary mit dem trainierten Modell und relevanten Informationen
    """
    # Prüfe, ob der Datensatz die Zielvariable enthält
    if 'Revenue' not in df.columns:
        raise ValueError("Der Datensatz enthält keine 'Revenue'-Spalte!")
    
    # Datensatz vorbereiten
    prepared_df = prepare_dataset(df)
    
    # Features auswählen
    X_selected = select_features(prepared_df)
    
    # Zielvariable
    y = prepared_df['Revenue']
    
    # Datensatz splitten (20% Testgröße, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # RandomForest-Modell trainieren
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        max_samples=0.7,
        max_features='sqrt',  # Parameter als String für Kompatibilität
        bootstrap=True,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluierung des Modells
    y_pred = model.predict(X_test)
    
    # Metriken berechnen
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    # Modell und Informationen in Dictionary speichern
    model_info = {
        'model_name': 'randomforest',
        'model': model,
        'feature_names': list(X_train.columns),
        'metrics': metrics
    }
    
    # Modelldatei speichern
    try:
        model_filename = "randomforest_model.pkl"
        joblib.dump(model, model_filename)
        print(f"Modell in {model_filename} gespeichert.")
    except Exception as e:
        print(f"Warnung: Modell konnte nicht gespeichert werden: {str(e)}")
    
    return model_info

def predict_with_f9tuned(model_info, new_data):
    """
    Führt Vorhersagen mit dem F9Tuned-Modell durch
    
    Args:
        model_info (dict): Modellinformationen aus train_f9tuned_model()
        new_data (pandas.DataFrame): Neue Daten für die Vorhersage
        
    Returns:
        tuple: (Wahrscheinlichkeiten, binäre Vorhersagen)
    """
    if 'model' not in model_info:
        raise ValueError("Kein Modell im model_info-Dictionary gefunden!")
    
    model = model_info['model']
    
    # Features auswählen und vorbereiten
    new_data_prepared = prepare_dataset(new_data)
    X = select_features(new_data_prepared)
    
    # Prüfe, ob wir die gleichen Features haben wie beim Training
    if 'feature_names' in model_info:
        expected_features = set(model_info['feature_names'])
        actual_features = set(X.columns)
        
        # Prüfe auf fehlende Features
        missing_features = expected_features - actual_features
        if missing_features:
            for feature in missing_features:
                X[feature] = 0  # Fehlende Features mit 0 auffüllen
        
        # Begrenzen auf die beim Training verwendeten Features und die Reihenfolge sicherstellen
        X = X[model_info['feature_names']]
    
    # Vorhersagen treffen
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[:, 1]  # Wahrscheinlichkeit für Klasse 1
            predictions = model.predict(X)
        else:
            predictions = model.predict(X)
            probabilities = np.zeros(len(predictions))
            for i, pred in enumerate(predictions):
                probabilities[i] = 1.0 if pred > 0 else 0.0
    except Exception as e:
        print(f"Fehler bei der Vorhersage: {str(e)}")
        # Fallback
        predictions = np.zeros(len(X))
        probabilities = np.zeros(len(X))
    
    return probabilities, predictions

def get_f9tuned_evaluation(model_info):
    """
    Liefert Evaluationsmetriken für das F9Tuned-Modell
    
    Args:
        model_info (dict): Modellinformationen aus train_f9tuned_model()
        
    Returns:
        dict: Dictionary mit Evaluationsmetriken
    """
    if 'metrics' in model_info:
        return model_info['metrics']
    else:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }