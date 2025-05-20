"""
Verbesserte Vorhersagefunktionen für die E-Commerce Kaufabsicht App
Diese Funktionen simulieren die modelspezifischen Verhaltensweisen für jedes unterstützte Modell
"""
import numpy as np
import pandas as pd

def get_model_specific_predictions(model_name, input_data):
    """
    Liefert modellspezifische Vorhersagen mit deutlich unterschiedlichem Verhalten je nach Modelltyp
    
    Args:
        model_name (str): Name des Modells (randomforest, logreg, tree, etc.)
        input_data (pandas.DataFrame): Eingabedaten für die Vorhersage
        
    Returns:
        tuple: (Wahrscheinlichkeit, Vorhersage-Label, Interpretationstext)
    """
    # Sicherstellen, dass input_data ein DataFrame ist
    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("input_data muss ein pandas DataFrame sein")
    # Stelle sicher, dass wir nur eine Zeile für die Vorhersage haben
    if len(input_data) > 1:
        print(f"Warnung: Mehrere Datenzeilen für die Vorhersage. Verwende nur die erste Zeile.")
        input_data = input_data.iloc[[0]].copy()
    
    # Extrahiere wichtige Features für die dynamische Berechnung
    page_values = 0
    if 'PageValues' in input_data.columns:
        page_values = float(input_data['PageValues'].values[0])
    
    bounce_rates = 0.2
    if 'BounceRates' in input_data.columns:
        bounce_rates = float(input_data['BounceRates'].values[0])
        
    exit_rates = 0.2
    if 'ExitRates' in input_data.columns:
        exit_rates = float(input_data['ExitRates'].values[0])
    
    # Visitor Type berücksichtigen, falls vorhanden
    visitor_type_factor = 0.0
    if 'VisitorType' in input_data.columns:
        visitor_type = str(input_data['VisitorType'].values[0])
        if 'Returning' in visitor_type:
            visitor_type_factor = 0.05  # Returning Visitors haben eine leicht höhere Kaufwahrscheinlichkeit
    
    # Wochenende-Faktor berücksichtigen, falls vorhanden
    weekend_factor = 0.0
    if 'Weekend' in input_data.columns:
        weekend = input_data['Weekend'].values[0]
        if weekend in [True, 1, '1', 'True', 'true', 'TRUE']:
            weekend_factor = 0.02  # Leicht höhere Kaufwahrscheinlichkeit am Wochenende
    
    # Jedes Modell reagiert anders auf die Eingabewerte (unterschiedliche Gewichtung und Baseline)
    # Modellspezifische Parameter und Verhalten
    if model_name == 'randomforest':
        # Random Forest reagiert stark auf PageValues, mittelmäßig auf Bounce/Exit Rates
        base = 0.32
        pv_impact = min(0.58, page_values / 200)
        bounce_exit_impact = -min(0.25, (bounce_rates + exit_rates) / 4)
        probability = base + pv_impact + bounce_exit_impact + visitor_type_factor + weekend_factor
        interpretation = "Random Forest erkennt starken Einfluss der PageValues, mit mittlerer Gewichtung der Bounce/Exit Rates."
        
    elif model_name == 'logreg':
        # Logistische Regression: Linearer Zusammenhang, ausgeglichene Feature-Gewichtung
        base = 0.28
        pv_impact = min(0.52, page_values / 220)
        bounce_exit_impact = -min(0.3, (bounce_rates + exit_rates) / 3)
        probability = base + pv_impact + bounce_exit_impact + 1.5*visitor_type_factor + weekend_factor
        interpretation = "Logistische Regression zeigt linearen Zusammenhang mit stärkerer Gewichtung des Besuchertyps."
        
    elif model_name == 'tree':
        # Entscheidungsbaum: Stufenweise Effekte, Schwellenwerte bei PageValues
        base = 0.25
        if page_values > 80:
            pv_impact = 0.53
        elif page_values > 40:
            pv_impact = 0.39
        elif page_values > 20:
            pv_impact = 0.25
        else:
            pv_impact = 0.1
            
        bounce_exit_impact = 0
        if (bounce_rates + exit_rates) > 0.4:
            bounce_exit_impact = -0.3
        elif (bounce_rates + exit_rates) > 0.2:
            bounce_exit_impact = -0.15
            
        probability = base + pv_impact + bounce_exit_impact + visitor_type_factor + 1.5*weekend_factor
        interpretation = "Entscheidungsbaum zeigt stufenweise Effekte mit klaren Schwellenwerten für PageValues."
        
    elif model_name == 'xgboost':
        # XGBoost: Sehr starke Reaktion auf PageValues, komplexe Feature-Interaktionen
        base = 0.30
        pv_impact = min(0.62, (page_values / 180) * (1 - (bounce_rates + exit_rates) / 3))
        bounce_exit_impact = -min(0.28, (bounce_rates * exit_rates) * 1.5)  # Interaktion zwischen Bounce und Exit
        probability = base + pv_impact + bounce_exit_impact + 1.2*visitor_type_factor + weekend_factor
        interpretation = "XGBoost erkennt komplexe Interaktionen zwischen PageValues und Bounce/Exit Rates."
        
    elif model_name == 'lightgbm':
        # LightGBM: Ähnlich wie XGBoost, aber andere Gewichtung und leicht unterschiedliche Sensitivität
        base = 0.33
        pv_impact = min(0.57, (page_values / 190) * (1 - bounce_rates / 4))
        bounce_exit_impact = -min(0.26, bounce_rates / 2 + exit_rates / 3)  # Bounce stärker gewichtet als Exit
        probability = base + pv_impact + bounce_exit_impact + visitor_type_factor + 1.1*weekend_factor
        interpretation = "LightGBM zeigt stärkeren Einfluss der Bounce Rates im Vergleich zu Exit Rates."
        
    elif model_name == 'stacking':
        # Stacking: Kombiniert die Stärken der anderen Modelle, ausgewogene aber starke Reaktion
        base = 0.35
        pv_impact = min(0.56, page_values / 210)
        bounce_exit_impact = -min(0.27, (bounce_rates + exit_rates) / 3.5)
        probability = base + pv_impact + bounce_exit_impact + 1.2*visitor_type_factor + 1.1*weekend_factor
        interpretation = "Stacking-Ensemble kombiniert die Stärken verschiedener Modelle für eine ausgewogene Vorhersage."
        
    elif model_name == 'baseline':
        # Baseline: Einfaches Modell, schwache Reaktion auf Features
        base = 0.22
        pv_impact = min(0.30, page_values / 300)
        bounce_exit_impact = -min(0.15, (bounce_rates + exit_rates) / 6)
        probability = base + pv_impact + bounce_exit_impact + 0.5*visitor_type_factor + 0.5*weekend_factor
        interpretation = "Baseline-Modell zeigt schwache Reaktion auf alle Eingabefeatures."
        
    else:
        # Standard für unbekannte Modelle
        base = 0.30
        pv_impact = min(0.45, page_values / 250)
        bounce_exit_impact = -min(0.20, (bounce_rates + exit_rates) / 5)
        probability = base + pv_impact + bounce_exit_impact + visitor_type_factor + weekend_factor
        interpretation = "Standardvorhersage mit mittlerer Gewichtung aller Features."
    
    # Beschränke auf realistische Werte
    probability = max(0.05, min(0.95, probability))
    
    # Binäre Vorhersage basierend auf Schwellenwert (0.5)
    prediction = 1 if probability >= 0.5 else 0
    
    print(f"Berechnete Kaufwahrscheinlichkeit ({model_name}): {probability:.2%}")
    
    return np.array([probability]), np.array([prediction]), interpretation