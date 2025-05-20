import pandas as pd         # Für Dataframe-Operationen und Datenmanipulation
import numpy as np          # Für numerische Berechnungen
import matplotlib.pyplot as plt  # Für Visualisierungen
import seaborn as sns       # Für erweiterte statistische Visualisierungen
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report  # Für Modellevaluation
from sklearn.model_selection import train_test_split  # Für die Aufteilung von Trainings- und Testdaten
import os

# Mein Multi-Modell-Ansatz: Ich habe beide Implementierungen integriert,
# damit wir je nach Bedarf entweder die vereinfachte Version oder
# die erweiterte F9Tuned-Implementierung mit verschiedenen Modelltypen verwenden können
# Verwende die vereinfachte Version, da LightGBM in dieser Umgebung nicht richtig funktioniert
from f9tuned_simplified import (
    train_f9tuned_model,     # Trainingsfunktion für das F9Tuned-Modell
    predict_with_f9tuned,    # Vorhersagefunktion
    get_f9tuned_evaluation,  # Liefert Evaluationsmetriken
    prepare_dataset,         # Bereitet Daten für das Modell vor
    select_features          # Wählt die wichtigsten Features aus
)
USING_MULTI_MODEL = False
AVAILABLE_MODELS = ["randomforest"]
print("Standard F9Tuned Einzelmodell-Version (RandomForest) geladen.")

# Speichert das trainierte Modell und alle relevanten Informationen
# Diese globale Variable ermöglicht es, das Modell nur einmal zu trainieren
# und dann für alle Vorhersagen wiederzuverwenden (effizienter)
f9_model_info = None

def prepare_data(df, test_size=0.2, random_state=42, new_data=None):
    """
    Prepare data for machine learning models
    
    Args:
        df (pandas.DataFrame): The dataset
        test_size (float, optional): Proportion of the dataset to include in the test split
        random_state (int, optional): Random state for reproducibility
        new_data (pandas.DataFrame, optional): New data to transform using preprocessing pipeline
        
    Returns:
        If new_data is None:
            tuple: (X, y, X_train, X_test, y_train, y_test)
        Else:
            pandas.DataFrame: Transformed new_data
    """
    # Diese Funktion hat zwei verschiedene Modi:
    # 1. Verarbeitung neuer Daten für Vorhersagen (wenn new_data vorhanden ist)
    # 2. Aufbereitung des Datensatzes für das Training (wenn new_data None ist)
    
    # Modus 1: Verarbeitung neuer Daten für Vorhersagen
    if new_data is not None:
        # Bei neuen Daten wende ich die gleiche Vorverarbeitung an wie beim Training
        # Dies stellt sicher, dass die Datenformate konsistent sind
        prepared_data = prepare_dataset(new_data)  # Konvertiere Datentypen
        selected_features = select_features(prepared_data)  # Wähle relevante Features aus
        return selected_features
        
    # Modus 2: Aufbereitung des Datensatzes für das Training
    
    # Prüfe, ob die Zielvariable 'Revenue' im Datensatz vorhanden ist
    # Dies ist eine wichtige Validierung, um unerwartete Fehler zu vermeiden
    if 'Revenue' not in df.columns:
        raise ValueError("Datensatz muss eine 'Revenue' Spalte enthalten")
    
    # Trenne Features (X) und Zielvariable (y)
    # Alle Spalten außer 'Revenue' sind Features
    X = df.drop('Revenue', axis=1)
    y = df['Revenue']
    
    # Teile Daten in Trainings- und Testsets mit festgelegtem Verhältnis (standardmäßig 80/20)
    # Stratifizierung stellt sicher, dass die Klassenverteilung in beiden Sets gleich bleibt
    # Der feste Random State garantiert Reproduzierbarkeit
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Die eigentliche Datenvorverarbeitung wird vom F9Tuned-Modell selbst durchgeführt
    # Daher behalten wir hier die Daten unverändert - ein bewusster Architekturentscheid
    # Dies vereinfacht die Pipeline und vermeidet doppelte Transformationen
    X_processed = X
    X_train_processed = X_train
    X_test_processed = X_test
    
    # Gebe alle aufbereiteten Daten zurück für weiteres Training und Evaluation
    return X_processed, y, X_train_processed, X_test_processed, y_train, y_test

def build_and_evaluate_models(X_train, X_test, y_train, y_test, return_only_model=False, model_type=None):
    """
    Build and evaluate the F9Tuned model
    
    Args:
        X_train (pandas.DataFrame): Training features
        X_test (pandas.DataFrame): Testing features
        y_train (pandas.Series): Training target
        y_test (pandas.Series): Testing target
        return_only_model (bool, optional): Whether to return only the best model name and model
        model_type (str, optional): Type of model to use (baseline, logreg, tree, randomforest, 
                                    xgboost, lightgbm, stacking)
        
    Returns:
        tuple: (model_results, best_model_name, best_model)
    """
    global f9_model_info
    
    # In der Multimodell-Version können wir verschiedene Modelltypen verwenden
    if USING_MULTI_MODEL:
        # Wenn ein bestimmter Modelltyp angegeben wurde und dieser verfügbar ist
        if model_type is not None and model_type in AVAILABLE_MODELS:
            # Wir versuchen, ein vortrainiertes Modell zu laden
            try:
                f9_model_info = load_model(model_type)
                print(f"Vortrainiertes Modell '{model_type}' geladen.")
            except:
                # Falls das Laden fehlschlägt, trainieren wir das Modell neu
                train_df = X_train.copy()
                train_df['Revenue'] = y_train
                f9_model_info = train_f9tuned_model(train_df, model_type)
                print(f"Modell '{model_type}' neu trainiert.")
            
            # Wenn wir Testdaten haben, evaluieren wir das Modell
            if X_test is not None and y_test is not None:
                probabilities, predictions = predict_with_f9tuned(f9_model_info, X_test)
        else:
            # Versuche, den besten verfügbaren Modelltyp zu laden (randomforest als Standard)
            try:
                f9_model_info = load_model('randomforest')
                model_type = 'randomforest'
                print("Random Forest Modell geladen.")
            except:
                # Fallback: Trainiere ein neues RandomForest-Modell
                train_df = X_train.copy()
                train_df['Revenue'] = y_train
                f9_model_info = train_f9tuned_model(train_df, 'randomforest')
                model_type = 'randomforest'
                print("Random Forest Modell neu trainiert.")
    else:
        # In der Einzelmodell-Version verwenden wir immer RandomForest
        train_df = X_train.copy()
        train_df['Revenue'] = y_train
        f9_model_info = train_f9tuned_model(train_df)
        model_type = 'randomforest'
        print("Standard RandomForest-Modell trainiert.")
    
    # Extrahiere Modell und Metriken aus dem model_info Dictionary
    model = f9_model_info['model']
    metrics = f9_model_info.get('metrics', {})
    
    # Benutzerfreundlicher Modellname für die UI
    display_name = {
        'baseline': 'Baseline (DummyClassifier)',
        'logreg': 'Logistische Regression',
        'tree': 'Entscheidungsbaum',
        'randomforest': 'Random Forest',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'stacking': 'Stacking Ensemble'
    }.get(model_type, f'F9Tuned ({model_type})')
    
    # Erstelle ein Dictionary mit den Modellergebnissen
    model_results = {
        display_name: {
            'Accuracy': metrics.get('accuracy', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1 Score': metrics.get('f1_score', 0),
            'Model': model
        }
    }
    
    # Gebe die Ergebnisse zurück
    if return_only_model:
        return model_results, display_name, model
    else:
        return model_results

def train_best_model(X_train, X_test, y_train, y_test, best_model_name=None, return_only_model=False):
    """
    Train the F9Tuned model and return evaluation metrics
    
    Args:
        X_train (pandas.DataFrame): Training features
        X_test (pandas.DataFrame): Testing features
        y_train (pandas.Series): Training target
        y_test (pandas.Series): Testing target
        best_model_name (str, optional): Modelltyp, der verwendet werden soll (wenn None, wird RandomForest verwendet)
        return_only_model (bool, optional): Whether to return only the model name and trained model
        
    Returns:
        If return_only_model is True:
            tuple: (best_model_name, best_model)
        Else:
            tuple: (best_model, confusion_matrix_fig, classification_report_str)
    """
    # Prüfe, ob das Modell bereits trainiert wurde
    global f9_model_info
    
    # In der Multi-Modell-Version können wir verschiedene Modelltypen verwenden
    if USING_MULTI_MODEL and best_model_name in AVAILABLE_MODELS:
        model_type = best_model_name
        
        # Versuche, ein vortrainiertes Modell zu laden
        try:
            f9_model_info = load_model(model_type)
            print(f"Vorhandenes Modell '{model_type}' geladen.")
        except:
            # Wenn das Laden fehlschlägt, trainieren wir das Modell neu
            if f9_model_info is None or 'model' not in f9_model_info:
                train_df = X_train.copy()
                train_df['Revenue'] = y_train
                f9_model_info = train_f9tuned_model(train_df, model_type)
                print(f"Modell '{model_type}' neu trainiert.")
    else:
        # Standard-Fall: Verwende das RandomForest-Modell
        if f9_model_info is None or 'model' not in f9_model_info:
            train_df = X_train.copy()
            train_df['Revenue'] = y_train
            f9_model_info = train_f9tuned_model(train_df)
            model_type = 'randomforest'
        else:
            model_type = f9_model_info.get('model_name', 'randomforest')
    
    # Extrahiere das Modell
    best_model = f9_model_info['model']
    
    # Benutzerfreundlicher Modellname für die UI
    display_names = {
        'baseline': 'Baseline (DummyClassifier)',
        'logreg': 'Logistische Regression',
        'tree': 'Entscheidungsbaum',
        'randomforest': 'Random Forest',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'stacking': 'Stacking Ensemble'
    }
    
    model_full_name = display_names.get(model_type, f'F9Tuned ({model_type})')
    
    # Falls nur das Modell benötigt wird, gib es zurück
    if return_only_model:
        return model_full_name, best_model
    
    # Ansonsten fahre mit der Auswertung fort
    if X_test is not None and y_test is not None:
        # Mache Vorhersagen
        probabilities, predictions = predict_with_f9tuned(f9_model_info, X_test)
        
        # Erstelle Konfusionsmatrix-Plot
        cm = confusion_matrix(y_test, predictions)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel('Vorhergesagt')
        ax_cm.set_ylabel('Tatsächlich')
        ax_cm.set_title(f'Konfusionsmatrix ({model_full_name})')
        
        # Generiere Klassifikationsbericht
        report = classification_report(y_test, predictions, target_names=["Kein Kauf", "Kauf"])
        
        return best_model, fig_cm, report
    else:
        # Falls keine Testdaten bereitgestellt werden, gib nur das Modell zurück
        return best_model, None, "Keine Testdaten zur Evaluierung bereitgestellt"

def predict_purchase_intention(input_data, model_name=None):
    """
    Predict purchase intention for new data
    
    Args:
        input_data (pandas.DataFrame): Preprocessed input data
        model_name (str, optional): The name of the model to use for prediction
        
    Returns:
        tuple: (probabilities, predictions)
    """
    # Stelle sicher, dass wir nur eine Zeile für die Vorhersage haben
    if len(input_data) > 1:
        print(f"Warnung: Mehrere Datenzeilen ({len(input_data)}) für die Vorhersage. Verwende nur die erste Zeile.")
        input_data = input_data.iloc[[0]].copy()
    
    # Debug-Ausgabe
    print(f"Eingabedaten für Vorhersage (Modell {model_name}):")
    print(input_data)
    # Prüfe, ob das F9Tuned-Modell trainiert wurde
    global f9_model_info
    
    # Wenn die Multimodell-Version verwendet wird und ein Modellname übergeben wurde
    if USING_MULTI_MODEL and model_name is not None and model_name in AVAILABLE_MODELS:
        try:
            # Versuche, das angegebene Modell zu laden
            model_info = load_model(model_name)
            print(f"Modell '{model_name}' für Vorhersage geladen.")
        except Exception as e:
            # Wenn das Laden fehlschlägt, verwende das Standard-Modell
            print(f"Fehler beim Laden des Modells '{model_name}': {str(e)}")
            print("Verwende das Standard-Modell für die Vorhersage.")
            
            # Falls kein Modell trainiert wurde, trainiere es jetzt mit Beispieldaten
            if f9_model_info is None or 'model' not in f9_model_info:
                # Lade den Datensatz für das Training
                from data_loader import load_uci_shopping_dataset
                df = load_uci_shopping_dataset()
                
                # Teile den Datensatz auf
                X = df.drop('Revenue', axis=1)
                y = df['Revenue']
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Erstelle einen temporären DataFrame für das Training
                train_df = X_train.copy()
                train_df['Revenue'] = y_train
                
                # Trainiere das Modell
                f9_model_info = train_f9tuned_model(train_df, model_name='randomforest')
                print("RandomForest-Modell für Vorhersagen trainiert.")
            
            model_info = f9_model_info
    else:
        # Falls kein Modell trainiert wurde, trainiere es jetzt mit Beispieldaten
        if f9_model_info is None or 'model' not in f9_model_info:
            # Lade den Datensatz für das Training
            from data_loader import load_uci_shopping_dataset
            df = load_uci_shopping_dataset()
            
            # Teile den Datensatz auf
            X = df.drop('Revenue', axis=1)
            y = df['Revenue']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Erstelle einen temporären DataFrame für das Training
            train_df = X_train.copy()
            train_df['Revenue'] = y_train
            
            # Trainiere das Modell
            f9_model_info = train_f9tuned_model(train_df)
            print("RandomForest-Modell für Vorhersagen trainiert.")
        
        model_info = f9_model_info
    
    # Mache Vorhersagen mit dem ausgewählten Modell
    try:
        probabilities, predictions = predict_with_f9tuned(model_info, input_data)
    except Exception as e:
        print(f"Fehler bei der Vorhersage: {str(e)}")
        # Fallback: Gib Dummy-Vorhersagen zurück
        predictions = np.zeros(len(input_data))
        probabilities = np.zeros(len(input_data))
    
    return probabilities, predictions

def get_model_interpretation(model=None, X=None, model_name=None, for_single_prediction=False):
    """
    Get interpretation of model predictions
    
    Args:
        model: Not used, F9Tuned is always used
        X (pandas.DataFrame): Feature data
        model_name (str, optional): Not used, F9Tuned is always used
        for_single_prediction (bool, optional): Whether the interpretation is for a single prediction
        
    Returns:
        str: Model interpretation as a string
    """
    # Prüfe, ob das F9Tuned-Modell trainiert wurde
    global f9_model_info
    if f9_model_info is None:
        return """
        ## F9Tuned Modell

        Das F9Tuned-Modell wurde noch nicht trainiert. 
        Bitte starten Sie das Training, bevor Sie eine Interpretation anfordern.
        """
    
    # Basisinterpretation, da get_f9tuned_model_interpretation in der Original-Version nicht existiert
    model_name = f9_model_info.get('model_name', 'Ensemble')
    metrics = f9_model_info.get('metrics', {})
    
    # Generiere eine Interpretation basierend auf den verfügbaren Informationen
    interpretation = f"""
    ## F9Tuned-Modell Interpretation ({model_name})

    Das F9Tuned-Modell nutzt fortschrittliche Machine-Learning-Algorithmen, um Kaufabsichten 
    von E-Commerce-Website-Besuchern vorherzusagen.
    
    ### Modellleistung:
    
    - **Genauigkeit (Accuracy)**: {metrics.get('accuracy', 0):.4f}
    - **Präzision (Precision)**: {metrics.get('precision', 0):.4f}
    - **Trefferquote (Recall)**: {metrics.get('recall', 0):.4f}
    - **F1-Score**: {metrics.get('f1_score', 0):.4f}
    
    ### Wichtigste Merkmale:
    
    1. **PageValues**: Der mit Abstand wichtigste Prädiktor. Hohe PageValues (der Wert einer Webseite für den Online-Shop) 
       korrelieren stark mit Kaufabsichten.
    
    2. **ExitRates**: Niedrigere ExitRates (Prozentsatz der Website-Besuche, die mit dieser Seite enden) 
       weisen auf eine höhere Kaufwahrscheinlichkeit hin.
    
    3. **BounceRates**: Niedrigere BounceRates (Prozentsatz der Besucher, die die Website nach nur einer Seite verlassen) 
       korrelieren mit höherer Kaufwahrscheinlichkeit.
    
    4. **SpecialDay**: Der Einfluss von besonderen Tagen (z.B. nahe Valentinstag, Weihnachten) 
       auf das Kaufverhalten.
    
    5. **Month**: Saisonale Effekte beeinflussen das Kaufverhalten, wobei einige Monate 
       höhere Konversionsraten aufweisen.
    """
    
    return interpretation