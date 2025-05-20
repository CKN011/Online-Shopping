"""
Direkte Verwendung der PKL-Modelldateien für Vorhersagen mit joblib
"""
import os
import numpy as np
import pandas as pd
import joblib  # Joblib anstelle von pickle für bessere Kompatibilität

# Pfade zu den Modelldateien
MODEL_PATHS = {
    "randomforest": "randomforest_model.pkl",        # Hauptverzeichnis-Version
    "logreg": "logreg_model.pkl",                    # Hauptverzeichnis-Version
    "tree": "tree_model.pkl",                        # Hauptverzeichnis-Version
    "baseline": "baseline_model.pkl",                # Hauptverzeichnis-Version
    "xgboost": "attached_assets/xgb_model.pkl",      # Asset-Version
    "lightgbm": "attached_assets/lgbm_model.pkl",    # Asset-Version
    "stacking": "attached_assets/stacking_model.pkl" # Asset-Version
}

# Alternative Modellpfade für verschiedene Namenskonventionen
ALT_MODEL_PATHS = {
    "randomforest": ["attached_assets/best_rf_model.pkl"],
    "logreg": ["attached_assets/logreg_model.pkl"],
    "tree": ["attached_assets/tree_model.pkl"],
    "xgboost": ["xgb_model.pkl"],
    "lightgbm": ["lgbm_model.pkl"],
    "stacking": ["stacking_model.pkl"],
    "baseline": ["attached_assets/baseline_model.pkl"]
}

# Cache für geladene Modelle
loaded_models = {}

def load_model(model_name):
    """
    Lädt ein Modell mit joblib aus einer PKL-Datei mit verbesserter Fehlerbehandlung
    
    Args:
        model_name (str): Name des zu ladenden Modells
        
    Returns:
        object: Das geladene Modell
    """
    global loaded_models
    
    # Prüfe, ob das Modell bereits geladen wurde
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    # Primärer Modellpfad
    primary_path = MODEL_PATHS.get(model_name, "")
    
    # Liste aller möglichen Pfade für dieses Modell
    all_paths = []
    
    # Primärpfad hinzufügen, wenn vorhanden
    if primary_path:
        all_paths.append(primary_path)
    
    # Alternative Pfade hinzufügen
    if model_name in ALT_MODEL_PATHS:
        all_paths.extend(ALT_MODEL_PATHS[model_name])
    
    # Auch die umgekehrten Pfade (mit/ohne attached_assets) hinzufügen
    extra_paths = []
    for path in all_paths:
        if path.startswith("attached_assets/"):
            extra_paths.append(os.path.basename(path))
        else:
            extra_paths.append(f"attached_assets/{path}")
    all_paths.extend(extra_paths)
    
    # Duplikate entfernen und nur existierende Pfade behalten
    valid_paths = [path for path in all_paths if os.path.exists(path)]
    
    if not valid_paths:
        print(f"Keine gültigen Modellpfade für {model_name} gefunden. Geprüft wurden: {all_paths}")
        raise FileNotFoundError(f"Kein gültiger Pfad für Modell {model_name} gefunden.")
    
    # Versuche, Modell aus jedem gültigen Pfad zu laden
    last_error = None
    for path in valid_paths:
        try:
            print(f"Lade Modell {model_name} aus {path} mit joblib...")
            model = joblib.load(path)
            loaded_models[model_name] = model
            print(f"Modell {model_name} erfolgreich mit joblib geladen!")
            return model
        except Exception as e:
            print(f"Fehler beim Laden mit joblib von {path}: {str(e)}")
            last_error = e
    
    # Wenn alle Pfade fehlschlagen, verwende den letzten Fehler
    raise Exception(f"Fehler beim Laden des Modells {model_name} aus allen Pfaden: {str(last_error)}")

def prepare_data_for_prediction(input_data, model_name='randomforest'):
    """
    Bereitet die Daten für die Vorhersage vor - erweitert für verschiedene Modelltypen
    
    Args:
        input_data (pandas.DataFrame): Die Eingabedaten
        model_name (str): Der Name des Modells, für das die Daten vorbereitet werden
        
    Returns:
        pandas.DataFrame: Die vorbereiteten Daten
    """
    data = input_data.copy()
    
    # XGBoost benötigt spezielle Vorverarbeitung mit One-Hot-Encoding
    if model_name == 'xgboost':
        # Stelle sicher, dass die Kategorien korrekt codiert sind
        print(f"Bereite Daten speziell für {model_name} vor mit One-Hot-Encoding...")
        
        # Prüfe, ob wir 'Weekend' und 'VisitorType' codieren müssen
        if 'Weekend' in data.columns and not any(col.startswith('Weekend_') for col in data.columns):
            # Konvertiere Boolean/String zu numerisch
            if data['Weekend'].dtype == bool or str(data['Weekend'].dtype) == 'bool':
                data['Weekend_1'] = data['Weekend'].astype(int)
            else:
                # Wenn es ein String ist (TRUE/FALSE)
                data['Weekend_1'] = data['Weekend'].map(lambda x: 1 if str(x).lower() in ['true', '1', 't', 'yes', 'y'] else 0)
            data = data.drop('Weekend', axis=1)
        
        # One-Hot-Encoding für VisitorType
        if 'VisitorType' in data.columns and not any(col.startswith('VisitorType_') for col in data.columns):
            # Erstelle VisitorType_Returning_Visitor und VisitorType_Other
            data['VisitorType_Returning_Visitor'] = (data['VisitorType'] == 'Returning_Visitor').astype(int)
            data['VisitorType_Other'] = (data['VisitorType'] == 'Other').astype(int)
            data = data.drop('VisitorType', axis=1)
        
        # One-Hot-Encoding für Month
        if 'Month' in data.columns and not any(col.startswith('Month_') for col in data.columns):
            months = ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for month in months:
                data[f'Month_{month}'] = (data['Month'] == month).astype(int)
            data = data.drop('Month', axis=1)
        
        # Stelle sicher, dass alle für das Modell erforderlichen Spalten vorhanden sind
        required_columns = [
            'Informational', 'BounceRates', 'ExitRates', 'PageValues', 
            'SpecialDay', 'OperatingSystems', 'Month_Dec', 'Month_Feb', 
            'Month_Jul', 'Month_June', 'Month_Mar', 'Month_May', 
            'Month_Nov', 'Month_Oct', 'Month_Sep', 'VisitorType_Other', 
            'VisitorType_Returning_Visitor', 'Weekend_1'
        ]
        
        # Füge fehlende Spalten mit 0 hinzu
        for col in required_columns:
            if col not in data.columns:
                print(f"Fehlende Spalte für XGBoost: {col} - füge mit 0 hinzu")
                data[col] = 0
                
        # Verwende nur die benötigten Spalten in der richtigen Reihenfolge
        return data[required_columns]
    
    # Für andere Modelle wie RandomForest, LogReg, Tree
    else:
        # Stelle sicher, dass wir nur eine Zeile für die Vorhersage haben, wenn nötig
        if len(data) > 1 and data.shape[0] > data.shape[1]:
            print(f"Info: Mehrere Datenzeilen für die Vorhersage. Verarbeite alle {len(data)} Zeilen.")
        
        # Kernfeatures, die für alle Modelle wichtig sind
        core_features = ['Informational', 'BounceRates', 'ExitRates', 'PageValues', 
                         'SpecialDay', 'OperatingSystems', 'Month', 'VisitorType', 'Weekend']
        
        # Prüfe, welche Features vorhanden sind
        available_features = [f for f in core_features if f in data.columns]
        
        if not available_features:
            raise ValueError("Keine der benötigten Features für die Vorhersage gefunden")
        
        # Verwende die verfügbaren Features
        X = data[available_features]
        
        return X

def predict_with_real_model(model_name, input_data):
    """
    Führt eine Vorhersage mit einem echten Modell durch
    
    Args:
        model_name (str): Name des zu verwendenden Modells
        input_data (pandas.DataFrame): Die Eingabedaten
        
    Returns:
        tuple: (Wahrscheinlichkeiten, Vorhersagen)
    """
    try:
        # Lade das Modell
        model = load_model(model_name)
        
        # Bereite die Daten vor - spezifisch für das Modell
        X = prepare_data_for_prediction(input_data, model_name)
        
        print(f"Vorhersage mit {model_name} wird durchgeführt...")
        
        # Führe die Vorhersage durch (abhängig vom Modelltyp)
        if hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba")):
            # Für Modelle mit Wahrscheinlichkeitsvorhersage
            try:
                proba = model.predict_proba(X)
                # Prüfe, ob wir eine einzelne Klasse oder mehrere haben
                if proba.shape[1] > 1:
                    # Mehrere Klassen (die zweite Spalte enthält die Wahrscheinlichkeit für Klasse 1)
                    prob = proba[:, 1]
                else:
                    # Nur eine Klasse
                    prob = proba[:, 0]
                
                # Binäre Vorhersage basierend auf Schwellenwert 0.5
                pred = (prob >= 0.5).astype(int)
                
                print(f"Vorhersage mit {model_name} erfolgreich - echtes Modell")
                return prob, pred
            except Exception as e:
                print(f"Fehler bei predict_proba für Modell {model_name}: {str(e)}")
                # Versuche die normale Vorhersage
                try:
                    pred = model.predict(X)
                    # Erzeuge Wahrscheinlichkeit basierend auf Modelltyp
                    if model_name == 'randomforest' and hasattr(model, 'predict_proba'):
                        # Wir verwenden die Klassenstimmenanteile bei Random Forest
                        if hasattr(model, 'estimators_'):
                            estimators_predictions = np.array([tree.predict(X) for tree in model.estimators_])
                            prob = np.mean([pred == 1 for pred in estimators_predictions], axis=0)
                        else:
                            prob = np.where(pred > 0, 0.7, 0.3)
                    else:
                        # Für andere Modelle ohne spezifische Logik
                        prob = np.where(pred > 0, 0.7, 0.3)
                    
                    print(f"Vorhersage mit {model_name} erfolgreich - predict statt predict_proba")
                    return prob, pred
                except Exception as pred_err:
                    print(f"Fehler bei predict für Modell {model_name}: {str(pred_err)}")
                    raise
        else:
            # Für Modelle ohne Wahrscheinlichkeitsvorhersage
            pred = model.predict(X)
            # Skalierte Wahrscheinlichkeit je nach Modelltyp
            if model_name == 'randomforest':
                prob = np.where(pred > 0, 0.85, 0.15)  # Höheres Vertrauen für RF
            elif model_name == 'tree':
                prob = np.where(pred > 0, 0.75, 0.25)  # Mittleres Vertrauen für Tree
            else:
                prob = np.where(pred > 0, 0.7, 0.3)    # Standard für andere Modelle
            
            print(f"Vorhersage mit {model_name} erfolgreich - ohne predict_proba")
            return prob, pred
            
    except Exception as e:
        print(f"Fehler bei Vorhersage mit Modell {model_name}: {str(e)}")
        
        # Als Alternative versuchen wir, ob das Modell als Dictionary gespeichert wurde
        # In train_f9tuned_model ist das eigentliche Modell unter 'model'
        try:
            # Finde alle möglichen Modellpfade
            primary_path = MODEL_PATHS.get(model_name, "")
            all_paths = []
            if primary_path:
                all_paths.append(primary_path)
            if model_name in ALT_MODEL_PATHS:
                all_paths.extend(ALT_MODEL_PATHS[model_name])
            
            # Auch die umgekehrten Pfade (mit/ohne attached_assets) hinzufügen
            extra_paths = []
            for path in all_paths:
                if path.startswith("attached_assets/"):
                    extra_paths.append(os.path.basename(path))
                else:
                    extra_paths.append(f"attached_assets/{path}")
            all_paths.extend(extra_paths)
            
            # Nur existierende Pfade berücksichtigen
            valid_paths = [path for path in all_paths if os.path.exists(path)]
            
            for path in valid_paths:
                try:
                    print(f"Versuche alternatives Dictionary-Modell aus {path}...")
                    model_info = joblib.load(path)
                    if isinstance(model_info, dict) and 'model' in model_info:
                        print(f"Modell ist ein Dictionary mit 'model'-Schlüssel")
                        model = model_info['model']
                        X = prepare_data_for_prediction(input_data, model_name)
                        
                        # Einfache binäre Vorhersage
                        if hasattr(model, 'predict'):
                            pred = model.predict(X)
                            prob = np.where(pred > 0, 0.7, 0.3)
                            print(f"Alternative Vorhersage mit Dictionary-Modell {model_name} erfolgreich")
                            return prob, pred
                except Exception as alt_e:
                    print(f"Fehler beim Laden oder Verwenden des alternativen Modells {path}: {str(alt_e)}")
            
            print("Alle Versuche mit alternativen Modellen fehlgeschlagen.")
        except Exception as alt_e:
            print(f"Alternative Modellversuche fehlgeschlagen: {str(alt_e)}")
        
        # Wenn wir hier ankommen, konnten wir das Modell nicht laden - zeige einen Fehler an
        error_message = f"Modell {model_name} konnte nicht geladen werden. Bitte verwenden Sie ein anderes Modell oder überprüfen Sie die Modellinstallation."
        print(f"ERROR: {error_message}")
        raise RuntimeError(error_message)