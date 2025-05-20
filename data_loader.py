
import pandas as pd    
import io                 
import requests           # falls Datei online heruntergeladen wird
import numpy as np        # Für numerische Operationen

def load_uci_shopping_dataset():
    """
    Loads the 'Online Shoppers Purchasing Intention' dataset from the attached assets.
    
    This is the actual UCI "Online Shoppers Purchasing Intention" dataset,
    which contains real browsing data from an e-commerce website.
    
    Returns:
        pandas.DataFrame: The Online Shoppers Purchasing Intention dataset
    """
    # Dieses "Ladesystem wurde mit Hilfe von AI entwickelt"
    # 1. Versucht den Datensatz aus attached_assets zu laden (primäre Quelle)
    # 2. Wenn nicht vorhanden, versucht es ihn aus dem data-Verzeichnis zu laden
    # 3. Wenn auch das fehlschlägt, ladt es direkt vom UCI-Repository

    try:
        df = pd.read_csv('attached_assets/online_shoppers_intention.csv')
    except FileNotFoundError:
        try:
            df = pd.read_csv('data/online_shoppers_intention.csv')
        except FileNotFoundError:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
            df = pd.read_csv(url)
    
    # Konvertion in Bools von Weekend und Revenue
    df['Weekend'] = df['Weekend'].apply(lambda x: x == 'TRUE' if isinstance(x, str) else x)
    df['Revenue'] = df['Revenue'].apply(lambda x: x == 'TRUE' if isinstance(x, str) else x)
    
    # Wwichtige Informationen zum Datensatz für Übersicht

    print(f"Online Shoppers Purchasing Intention Datensatz mit {len(df)} Einträgen und {len(df.columns)} Merkmalen erfolgreich erstellt.")
    print(f"Konversionsrate im Datensatz: {df['Revenue'].mean()*100:.2f}%")
    
    return df

def preprocess_shopping_dataset(df):
    """
    Preprocess the Online Shoppers Purchasing Intention dataset
    
    Args:
        df (pandas.DataFrame): The raw dataset
        
    Returns:
        pandas.DataFrame: The preprocessed dataset
    """
    # Erstellen einef Kopie, um die Originaldaten nicht zu verändern
    
    preprocessed_df = df.copy()
    
    # Datensatz sollte eingentlich komplett sein, aber sicherstellen und ersetzen von fehlenden Werten
    
    if preprocessed_df.isnull().sum().sum() > 0:
        # Für numerische Spalten wird durch Median ersetzt
        
        numeric_cols = preprocessed_df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            preprocessed_df[col].fillna(preprocessed_df[col].median(), inplace=True)
        
        # Für kategorische Werte den Modus
        categorical_cols = preprocessed_df.select_dtypes(include=['object', 'bool']).columns
        for col in categorical_cols:
            preprocessed_df[col].fillna(preprocessed_df[col].mode()[0], inplace=True)
    
    # Sicherstellen dass die Zielvariable 'Revenue' ein Integer ist
    preprocessed_df['Revenue'] = preprocessed_df['Revenue'].astype(int)
    
    # Sicherstellen dass die 'Weekend' Spalte ein Boolean ist
    preprocessed_df['Weekend'] = preprocessed_df['Weekend'].astype(bool)
    
    return preprocessed_df
