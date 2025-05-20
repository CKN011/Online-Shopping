import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import os
import pickle
from sklearn.model_selection import train_test_split

from data_loader import load_uci_shopping_dataset
from data_analysis import (
    get_summary_statistics, 
    analyze_numerical_features, 
    analyze_categorical_features
)
from data_visualization import (
    plot_feature_importance,
    plot_correlation_heatmap,
    plot_categorical_distribution,
    plot_pairplot,
    plot_revenue_by_visitor_type,
    plot_purchase_by_month
)
from ml_models import (
    prepare_data,
    build_and_evaluate_models,
    train_best_model,
    predict_purchase_intention,
    get_model_interpretation
)
from utils import (
    export_dataframe_to_csv,
    create_download_link,
    save_uploaded_file
)
from f9tuned_simplified import (
    train_f9tuned_model,
    predict_with_f9tuned
)

# Integration des Trainingsablaufs (für die Integration in die App wurde ChatGPT verwendet)
def train_and_save_models_if_needed():
    """Trainiert und speichert Modelle, falls sie noch nicht existieren"""
    models_to_train = {
        'randomforest': 'randomforest_model.pkl',
        'logreg': 'logreg_model.pkl',
        'tree': 'tree_model.pkl',
        'baseline': 'baseline_model.pkl'
    }
    
    # Prüfen, ob Modelle bereits erstellt wurden
    models_exist = all(os.path.exists(file_path) for file_path in models_to_train.values())
    
    if not models_exist:
        st.info("Trainiere Modelle... Dies kann einen Moment dauern.")
        # Datensatz laden
        df = load_uci_shopping_dataset()
        
        # Für jedes Modell
        for model_name, file_path in models_to_train.items():
            try:
                # Training ausführen
                model_info = train_f9tuned_model(df, model_name)
                
                # Modell speichern
                with open(file_path, 'wb') as f:
                    pickle.dump(model_info, f)
                    
                print(f"Modell '{model_name}' in '{file_path}' gespeichert.")
            except Exception as e:
                print(f"Fehler beim Training von {model_name}: {str(e)}")

# Farbschema für die Anwendung (mit ChatGPT erstellt)
COLOR_SCHEMA = {
    'primary': '#FF4B4B',       # Rot für Primäraktionen
    'secondary': '#0083B8',     # Blau für sekundäre Aktionen
    'background': '#F9F9F9',    # Hellgrau für Hintergrund
    'text': '#333333',          # Dunkelgrau für Text
    'accent': '#FF9D00',        # Orange für Akzente
    'positive': '#00BFA5',      # Grün für positive Werte
    'negative': '#FF5252',      # Rot für negative Werte
    'neutral': '#757575'        # Grau für neutrale Werte
}

# Constants
MODEL_NAMES = {
    'F9Tuned (LightGBM)': 'F9Tuned (LightGBM) - Optimiert für maximale Leistung bei E-Commerce-Daten'
}

# Set page configuration
st.set_page_config(
    page_title="E-Commerce Kaufabsicht",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add CSS for styled components (Mit Hilfe von ChatGPT)
st.markdown("""
<style>
/* Verbesserte Metriken-Container */
.metric-container {
    background-color: #f8f9fa;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
    border-left: 4px solid #FF4B4B;
}
.metric-container:hover {
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    transform: translateY(-3px);
}
.metric-label {
    font-size: 0.9rem;
    color: #333333;
    font-weight: 600;
}

/* Marketing-freundliche Komponenten */
.highlight-box {
    background-color: #fff;
    border-radius: 8px;
    padding: 25px;
    margin: 15px 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border-left: 5px solid #0083B8;
}

.recommendation-box {
    background-color: #fff;
    border-radius: 10px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    border: 1px solid #e0e0e0;
}

.recommendation-header {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 15px;
    color: #0083B8;
}

.kpi-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #FF4B4B;
}

.kpi-label {
    font-size: 1rem;
    color: #555;
    margin-bottom: 5px;
}

/* Bessere Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    border-radius: 8px 8px 0 0;
    padding: 0 20px;
    background-color: #f5f5f5;
}

.stTabs [aria-selected="true"] {
    background-color: white !important;
    border-top: 3px solid #FF4B4B;
}

/* Bessere Buttons */
.stButton button {
    border-radius: 8px;
    padding: 3px 25px;
    font-weight: 500;
    transition: all 0.2s ease;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
/* Verbesserte Metriken */
.metric-value {
    font-size: 1.7rem;
    font-weight: bold;
    color: #1f77b4;
}

/* Verbesserte Toggle-Elemente */
div.row-widget.stRadio > div {
    display: flex;
    flex-direction: row;
    align-items: center;
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

div.row-widget.stRadio > div[role="radiogroup"] > label {
    background-color: transparent;
    border: 1px solid #e0e3e9;
    border-radius: 8px;
    padding: 10px 15px;
    margin: 3px;
    text-align: center;
    transition: all 0.2s ease;
}

div.row-widget.stRadio > div[role="radiogroup"] > label:hover {
    background-color: #f0f4f8;
    border-color: #c0cde0;
}

div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
    display: none;
}

/* Verbesserte Info-Boxen */
div.stAlert {
    border-radius: 8px;
    padding: 18px;
    margin: 15px 0;
    border-left-width: 10px !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}

/* Verbesserte Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 5px 5px 0 0;
    padding: 10px 20px;
    background-color: #f8f9fa;
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: #4e7aff;
}

/* Verbesserte Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    padding: 4px 25px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

/* Verbesserte Slider */
div.stSlider {
    padding: 10px 0;
}

div.stSlider > div > div > div > div {
    background-color: #4e7aff;
}

/* Verbesserte Checkboxen */
div.stCheckbox > label > div[data-testid="stMarkdownContainer"] > p {
    font-size: 1.05rem;
}
</style>
""", unsafe_allow_html=True)

def main():
    # Sidebar erstellen
    st.sidebar.title("E-Commerce Analyse")
    
    # Navigation Tabs erstellen
    selection = st.sidebar.radio(
        "Navigation",
        ["Datenübersicht mit Visualisierungen", "Vorhersagemodelle", "Interaktive Prognose"]
    )
    
    # Data loading
    with st.spinner("Datensatz wird geladen..."):
        df = load_uci_shopping_dataset()
    
    if selection == "Datenübersicht mit Visualisierungen":
        data_overview_page(df)
    elif selection == "Vorhersagemodelle":
        try:
            ml_models_page(df)
        except Exception as e:
            st.error(f"Fehler beim Laden der Vorhersagemodelle: {str(e)}")
            st.info("Die Modellierungsfunktionen stehen derzeit nicht zur Verfügung. Bitte versuchen Sie es später erneut oder nutzen Sie die anderen Funktionen der Anwendung.")
    elif selection == "Interaktive Prognose":
        try:
            interactive_prediction_page(df)
        except Exception as e:
            st.error(f"Fehler bei der interaktiven Prognose: {str(e)}")
            st.info("Die Prognosefunktionen stehen derzeit nicht zur Verfügung. Bitte versuchen Sie es später erneut oder nutzen Sie die Visualisierungsfunktionen der Anwendung.")


def data_overview_page(df):
    st.header("Online Shoppers Purchasing Intention Dataset")
    
    # Infos zum Datenset
    st.subheader("Übersicht zum UCI-Datensatz")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Anzahl der Datensätze", df.shape[0])
    with col2:
        st.metric("Anzahl der Merkmale", df.shape[1])
    with col3:
        purchase_rate = df['Revenue'].mean() * 100
        st.metric("Kaufrate", f"{purchase_rate:.2f}%")
    #Text mit ChatGPT erstellt
    st.info("""
    ### Über den UCI-Datensatz "Online Shoppers Purchasing Intention"
    
    Der "Online Shoppers Purchasing Intention" Datensatz stammt aus dem UCI Machine Learning Repository 
    und enthält Informationen über Besuch und Interaktionen mit einer E-Commerce-Webseite, die 
    Administrativen-, Informations- und Produktseiten enthält. Das Ziel ist die Vorhersage, ob ein Besucher 
    eine Kaufabsicht hat, was durch die binäre Revenue-Variable (0 = kein Kauf, 1 = Kauf) angezeigt wird.
    
    #### Merkmale im Datensatz:
    
    **Administrative Features:**
    - **Administrative**: Anzahl der besuchten Administrationsseiten
    - **Administrative_Duration**: Gesamtzeit auf Administrationsseiten (in Sekunden)
    
    **Informational Features:**
    - **Informational**: Anzahl der besuchten Informationsseiten
    - **Informational_Duration**: Gesamtzeit auf Informationsseiten (in Sekunden)
    
    **Product-Related Features:**
    - **ProductRelated**: Anzahl der besuchten Produktseiten
    - **ProductRelated_Duration**: Gesamtzeit auf Produktseiten (in Sekunden)
    
    **Andere Merkmale:**
    - **BounceRates**: Prozentsatz der Besucher, die die Website von dieser Seite aus verlassen haben, ohne weitere Aktionen auszuführen
    - **ExitRates**: Prozentsatz der Seitenansichten auf der Website, die die letzten in der Sitzung waren
    - **PageValues**: Durchschnittlicher Wert für die Webseite, basierend auf e-Commerce-Transaktionen
    - **SpecialDay**: Nähe des Webseitenbesuchs zu einem speziellen Tag (z.B. Muttertag, Valentinstag)
    - **Month**: Monat des Jahres
    - **OperatingSystems**: Identifikationsnummer des Betriebssystems des Besuchers
    - **Browser**: Identifikationsnummer des Browsers des Besuchers
    - **Region**: Identifikationsnummer der Region des Besuchers
    - **TrafficType**: Identifikationsnummer des Verkehrstyps (z.B. direkt, Suchmaschine)
    - **VisitorType**: Typ des Besuchers (Wiederkehrend, Neu, Andere)
    - **Weekend**: Ob der Tag ein Wochenende ist (Wahr/Falsch)
    
    **Zielvariable:**
    - **Revenue**: Ob der Besucher einen Kauf getätigt hat (Wahr/Falsch)
    """)
    
    # Interactive EDA
    st.subheader("Explorative Datenanalyse")
    
    # Tabs for different types of analysis
    tabs = st.tabs([
        "Allgemeine Statistiken", 
        "Korrelationsanalyse",
        "Besuchertyp-Analyse",
        "Zeitliche Muster"
    ])
    
    with tabs[0]:
        st.subheader("Deskriptive Statistiken")
        stats_df = get_summary_statistics(df)
        st.dataframe(stats_df, use_container_width=True)
        
        # Comparing features between converters and non-converters
        st.subheader("Feature-Vergleich: Käufer vs. Nicht-Käufer")
        num_features = analyze_numerical_features(df)
        st.dataframe(num_features)
    
    with tabs[1]:
        st.subheader("Korrelationsanalyse")
        
        try:
            # Correlation heatmap
            st.write("#### Korrelationsmatrix der numerischen Features")
            corr_fig = plot_correlation_heatmap(df)
            st.pyplot(corr_fig)
        except Exception as e:
            st.error(f"Fehler bei der Korrelationsanalyse: {str(e)}")
    
    with tabs[2]:
        st.subheader("Besuchertyp-Analyse")
        
        try:
            # Visitor type analysis
            st.write("#### Konversionsrate nach Besuchertyp")
            visitor_fig = plot_revenue_by_visitor_type(df)
            st.plotly_chart(visitor_fig, use_container_width=True)
            
            # Detailed visitor metrics
            st.write("#### Detaillierte Metriken nach Besuchertyp")
            
            # Calculate metrics for each visitor type
            visitor_metrics = {}
            
            for visitor_type in df['VisitorType'].unique():
                visitor_df = df[df['VisitorType'] == visitor_type]
                
                visitor_metrics[visitor_type] = {
                    'Anzahl': len(visitor_df),
                    'Anteil (%)': f"{len(visitor_df) / len(df) * 100:.2f}%",
                    'Konversionsrate (%)': f"{visitor_df['Revenue'].mean() * 100:.2f}%",
                    'Durchschnittl. PageValues': f"{visitor_df['PageValues'].mean():.2f}",
                    'Durchschnittl. Besuchsdauer (s)': f"{visitor_df['ProductRelated_Duration'].mean():.2f}"
                }
                
            # DataFrame für die Besuchertyp-Analyse, transponiert für bessere Darstellung
            visitor_df = pd.DataFrame(visitor_metrics).T
            # Konvertiert alle Spalten zum String, um Arrow Kompatibilität sicherzustellen
            visitor_df = visitor_df.astype(str)
            # Zeige die Besuchertyp-Analyse als Tabelle an
            st.dataframe(visitor_df, use_container_width=True)
        except Exception as e:
            st.error(f"Fehler bei der detaillierten Besuchertyp-Analyse: {str(e)}")

    with tabs[3]:
        st.subheader("Zeitliche Muster")
        
        try:
            # Monatliche patterns
            st.write("#### Kaufrate nach Monaten")
            monthly_fig = plot_purchase_by_month(df)
            st.plotly_chart(monthly_fig, use_container_width=True)
            
            # Wochenende vs Wochentag
            st.write("#### Wochenende vs. Wochentag")
            
            # Berechnung der Statistiken
            werktag_besuche = df[df['Weekend'] == False].shape[0]
            wochenend_besuche = df[df['Weekend'] == True].shape[0]
            werktag_konversion = df[df['Weekend'] == False]['Revenue'].mean() * 100
            wochenend_konversion = df[df['Weekend'] == True]['Revenue'].mean() * 100
            
            # Erstellt zwei Spalten für die Anzeige
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Besuche am Werktag", f"{werktag_besuche:,}".replace(",", "."))
                st.metric("Konversionsrate Werktag", f"{werktag_konversion:.2f}%")
            
            with col2:
                st.metric("Besuche am Wochenende", f"{wochenend_besuche:,}".replace(",", "."))
                st.metric("Konversionsrate Wochenende", f"{wochenend_konversion:.2f}%")
            
            # Balkendiagramm für Wochenende vs. Werktag
            weekend_data = pd.DataFrame({
                'Zeitraum': ['Werktag', 'Wochenende'],
                'Konversionsrate': [werktag_konversion, wochenend_konversion]
            })
            
            fig = px.bar(
                weekend_data, 
                x='Zeitraum', 
                y='Konversionsrate',
                text=weekend_data['Konversionsrate'].round(2).astype(str) + '%',
                title='Konversionsraten: Wochenende vs. Werktag',
                color='Zeitraum',
                color_discrete_map={'Werktag': '#636EFA', 'Wochenende': '#EF553B'}
            )
            
            fig.update_layout(
                xaxis_title='Zeitraum',
                yaxis_title='Konversionsrate (%)',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Fehler bei der Analyse zeitlicher Muster: {str(e)}")
        
    # Data filtering und export optionen, Funktion mit Replit trouble shooted
    st.subheader("Datenfilterung und Export")
    
    # Filter Optionen
    col1, col2 = st.columns(2)
    
    with col1:
        visitor_filter = st.multiselect(
            "Besuchertyp",
            options=sorted(df['VisitorType'].unique()),
            default=[]
        )
        
        month_filter = st.multiselect(
            "Monat",
            options=sorted(df['Month'].unique()),
            default=[]
        )
    
    with col2:
        weekend_filter = st.multiselect(
            "Wochenende",
            options=sorted(df['Weekend'].unique()),
            default=[]
        )
        
        revenue_filter = st.multiselect(
            "Kaufabsicht",
            options=sorted(df['Revenue'].unique()),
            default=[]
        )
    
    # Filter Anwendung
    filtered_df = df.copy()
    
    if visitor_filter:
        filtered_df = filtered_df[filtered_df['VisitorType'].isin(visitor_filter)]
    
    if month_filter:
        filtered_df = filtered_df[filtered_df['Month'].isin(month_filter)]
    
    if weekend_filter:
        filtered_df = filtered_df[filtered_df['Weekend'].isin(weekend_filter)]
    
    if revenue_filter:
        filtered_df = filtered_df[filtered_df['Revenue'].isin(revenue_filter)]
    
    # Anzeige der gefilterten Daten
    st.subheader("Gefilterte Daten")
    st.write(f"{len(filtered_df)} Einträge wurden ausgewählt")
    st.dataframe(filtered_df)
    
    # Gefilterte Daten als CSV exportieren (gem utils.py)
    csv_data = export_dataframe_to_csv(filtered_df)
    st.download_button(
        label="Als CSV herunterladen",
        data=csv_data,
        file_name="e_commerce_kaufabsicht_daten.csv",
        mime="text/csv"
    )

def ml_models_page(df):
    st.header("F9Tuned-Modell für Kaufabsichten")
    
    try:
        # Datenvorbereitung
        X, y, X_train, X_test, y_train, y_test = prepare_data(df)
        
        # 80/20 Split
        st.subheader("Datenaufteilung für das Training")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Trainingsdaten (80%)", f"{len(X_train)} Einträge")
        with col2:
            st.metric("Testdaten (20%)", f"{len(X_test)} Einträge")
        with col3:
            positive_rate = y.mean() * 100
            st.metric("Konversionsrate", f"{positive_rate:.2f}%")
        
        # Modellauswahl
        st.subheader("Modellauswahl")
        
        available_models = [
            "randomforest", "logreg", "tree", "xgboost", 
            "lightgbm", "stacking", "baseline"
        ]
        
        # Modell Namen
        model_display_names = {
            "randomforest": "Random Forest",
            "logreg": "Logistische Regression",
            "tree": "Entscheidungsbaum",
            "xgboost": "XGBoost",
            "lightgbm": "LightGBM",
            "stacking": "Stacking Ensemble",
            "baseline": "Baseline-Modell"
        }
        
        # Anzeige mit Checkboxen für jedes Modell
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            Diese Anwendung zeigt jeden der unterstützten Modelltypen und ihre Performance an.
            Die Modelle wurden mit einer 80/20-Datenteilung trainiert und können verwendet werden,
            um die Kaufwahrscheinlichkeit für neue E-Commerce-Besucher vorherzusagen.
            """)
            
        with col2:
            model_type = st.selectbox(
                "Modelltyp auswählen",
                options=available_models,
                format_func=lambda x: model_display_names.get(x, x),
                index=0
            )
        
        # Model results
        try:
            with st.spinner(f"Modell wird geladen: {model_type}"):
                # Lädt und evaluiert das Modell
                model_results, model_name, best_model = build_and_evaluate_models(
                    X_train, X_test, y_train, y_test, return_only_model=True, model_type=model_type
                )
                
                # Überprüfe zunächst, ob das Modell geladen werden kann
                try:
                    from model_loader import load_model
                    
                    # Versuche das Modell zu laden
                    loaded_model = load_model(model_type)
                    
                    # Wenn wir hier ankommen, konnte das Modell geladen werden
                    from model_metrics import get_model_metrics, get_classification_report, generate_confusion_matrix
                    
                    # Lade die spezifischen Metriken für dieses Modell
                    model_metrics = get_model_metrics(model_type)
                    
                    # Display model metrics
                    st.subheader("Modellmetriken")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Accuracy", 
                            f"{model_metrics.get('accuracy', 0):.2%}"
                        )
                    
                    with col2:
                        st.metric(
                            "Precision", 
                            f"{model_metrics.get('precision', 0):.2%}"
                        )
                    
                    with col3:
                        st.metric(
                            "Recall", 
                            f"{model_metrics.get('recall', 0):.2%}"
                        )
                    
                    with col4:
                        st.metric(
                            "F1 Score", 
                            f"{model_metrics.get('f1_score', 0):.2%}"
                        )
                        
                    # Zeige zusätzliche Informationen an
                    st.info(f"""
                    **Trainingsdaten:** {model_metrics.get('training_samples', 0):,} Samples  
                    **Testdaten:** {model_metrics.get('test_samples', 0):,} Samples  
                    **Datensatzgröße:** {model_metrics.get('dataset_size', 0):,} Samples
                    """)
                        
                    # Konfusionsmatrix und Klassifikationsbericht
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Konfusionsmatrix")
                        cm_values = generate_confusion_matrix(model_type)
                        cm = np.array([[cm_values['tn'], cm_values['fp']], [cm_values['fn'], cm_values['tp']]])
                        
                        # Visualisiere Konfusionsmatrix
                        plt.figure(figsize=(8, 6))
                        import seaborn as sns
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                                   xticklabels=['Kein Kauf', 'Kauf'],
                                   yticklabels=['Kein Kauf', 'Kauf'])
                        plt.xlabel('Vorhergesagt')
                        plt.ylabel('Tatsächlich')
                        plt.title(f'Konfusionsmatrix - {model_name}')
                        st.pyplot(plt.gcf())
                        
                    with col2:
                        st.subheader("Klassifikationsbericht")
                        report = get_classification_report(model_type)
                        st.text(report)
                                
                except Exception as e:
                    # Zeige Fehler als Fehlermeldung an
                    error_message = f"Fehler beim Laden des Modells {model_type}: {str(e)}"
                    print(f"ERROR: {error_message}")
                    st.error(error_message)
                    st.warning("Die Modellmetriken können nicht angezeigt werden. Bitte wählen Sie ein anderes Modell.")
        
        except Exception as e:
            st.error(f"Fehler bei der Modellevaluierung: {str(e)}")
    
    except Exception as e:
        st.error(f"Fehler bei der Datenvorbereitung: {str(e)}")

def interactive_prediction_page(df):
    st.header("Kaufabsicht vorhersagen")
    
    # Informationstext zur Verwendung der Vorhersage-Komponente
    st.info("""
    ## Modellvorhersagen im Vergleich
    
    Diese Anwendung zeigt alle verfügbaren Modellvorhersagen gleichzeitig an:
    
    - **RandomForest**
    - **Logistische Regression**
    - **Entscheidungsbaum**
    - **XGBoost**
    - **LightGBM**
    - **Stacking Ensemble**
    """)
    
    # Tabs für manuelle Eingabe und CSV-Upload
    eingabe_tab, upload_tab = st.tabs(["Manuelle Eingabe", "CSV-Datei hochladen"])
    
    with eingabe_tab:
        # Eingabefelder für die wichtigsten Merkmale
        st.subheader("Besucherdaten eingeben")
        
        try:
            # Daten vorbereiten
            X, y, X_train, X_test, y_train, y_test = prepare_data(df)
            
            # Vorbereitung des Multi-Modell-Ansatzes
            # Lade die vortrainierten Modelle einmal
            from model_loader import predict_with_real_model
            from custom_predictions import get_model_specific_predictions
            
            # Form column layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Numerische Eingaben
                page_values = st.slider(
                    "PageValues (Wert der besuchten Seiten)", 
                    min_value=0.0, 
                    max_value=150.0, 
                    value=0.0, 
                    step=5.0
                )
                
                bounce_rates = st.slider(
                    "BounceRates (Absprungrate)", 
                    min_value=0.0, 
                    max_value=0.2, 
                    value=0.02, 
                    step=0.001,
                    format="%.3f"
                )
                
                exit_rates = st.slider(
                    "ExitRates (Ausstiegsrate)", 
                    min_value=0.0, 
                    max_value=0.2, 
                    value=0.04, 
                    step=0.001,
                    format="%.3f"
                )
            
            with col2:
                # Kategorische Eingaben
                visitor_type = st.selectbox(
                    "Besuchertyp",
                    options=["Returning_Visitor", "New_Visitor", "Other"],
                    index=0
                )
                
                month = st.selectbox(
                    "Monat",
                    options=["May", "Nov", "Mar", "Dec", "Oct", "Sep", "Aug", "Jul", "June", "Feb"],
                    index=0
                )
                
                weekend = st.checkbox(
                    "Wochenende",
                    value=False
                )
                
                special_day = st.slider(
                    "SpecialDay (Nähe zu einem besonderen Tag)", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.0, 
                    step=0.1
                )
                
                operating_system = st.slider(
                    "OperatingSystem (Betriebssystem ID)", 
                    min_value=1, 
                    max_value=8, 
                    value=2
                )
            
            # Informational ist ein wichtiges Feature
            informational = st.slider(
                "Informational (Anzahl besuchter Informationsseiten)", 
                min_value=0, 
                max_value=10, 
                value=0
            )
            
            # Button zum Ausführen der Vorhersage
            if st.button("Kaufwahrscheinlichkeit vorhersagen"):
                
                # Erstellen eines DataFrames für die Eingabedaten mit den wichtigsten Features
                input_data = pd.DataFrame({
                    'Informational': [informational],
                    'BounceRates': [bounce_rates],
                    'ExitRates': [exit_rates],
                    'PageValues': [page_values],
                    'SpecialDay': [special_day],
                    'OperatingSystems': [operating_system]
                })
                
                # Hinzufügen der kategorialen Features
                if month:
                    input_data['Month'] = [month]
                
                if visitor_type:
                    input_data['VisitorType'] = [visitor_type]
                
                input_data['Weekend'] = [weekend]
                
                # Zeige die Eingabedaten an
                st.subheader("Eingabedaten für Vorhersage:")
                st.dataframe(input_data)
                
                # Versuche, die Vorhersagen mit allen Modellen zu machen
                st.subheader("Modellvorhersagen im Vergleich")
                
                # Liste der verfügbaren Modelle
                available_models = [
                    "randomforest", "logreg", "tree", "xgboost", 
                    "lightgbm", "stacking", "baseline"
                ]
                
                # Zeiger für die Fortschrittsanzeige
                progress_bar = st.progress(0)
                
                # DataFrames für die Ergebnisse vorbereiten
                results = []
                
                # Führe die Vorhersagen aus für jedes Modell
                for i, model_name in enumerate(available_models):
                    try:
                        # Aktualisiere die Fortschrittsanzeige
                        progress_bar.progress((i+1)/len(available_models))
                        
                        # Versuche zuerst, mit dem echten Modell vorherzusagen
                        try:
                            prob, pred = predict_with_real_model(model_name, input_data)
                            # Verbesserte Fehlerbehandlung für verschiedene Arrayformate
                            if isinstance(prob, np.ndarray):
                                if len(prob.shape) > 1 and prob.shape[1] > 1:
                                    # 2D-Array mit mehreren Spalten
                                    probability = float(prob[0][1])
                                else:
                                    # 1D-Array oder 2D-Array mit einer Spalte
                                    probability = float(prob[0])
                            else:
                                # Einzelner Wert
                                probability = float(prob)
                            # Korrigierte Vorhersage basierend auf Wahrscheinlichkeit
                            prediction = "Kauf" if probability >= 0.5 else "Kein Kauf"
                            print(f"Vorhersage mit {model_name} erfolgreich")
                        except Exception as model_err:
                            # Bei Fehlern setzen wir die Wahrscheinlichkeit auf 0 und markieren als "Kein Kauf"
                            probability = 0.0
                            prediction = "Kein Kauf"
                            results.append({
                                'Modell': model_name,
                                'Vorhersage': prediction,
                                'Wahrscheinlichkeit': f"ERROR: {str(model_err)}"
                            })
                            # Zur nächsten Iteration weitergehen
                            continue
                        
                        # Modellnamen für die Anzeige formatieren
                        display_names = {
                            'randomforest': 'Random Forest',
                            'logreg': 'Logistische Regression',
                            'tree': 'Entscheidungsbaum',
                            'xgboost': 'XGBoost',
                            'lightgbm': 'LightGBM',
                            'stacking': 'Stacking Ensemble',
                            'baseline': 'Baseline-Modell'
                        }
                        
                        display_name = display_names.get(model_name, model_name)
                        
                        # Sammle die Ergebnisse (ohne Quelle)
                        results.append({
                            'Modell': display_name,
                            'Wahrscheinlichkeit': probability * 100,  # In Prozent umwandeln
                            'Vorhersage': prediction
                        })
                        
                    except Exception as e:
                        st.error(f"Fehler bei der Vorhersage mit {model_name}: {str(e)}")
                
                # Entferne die Fortschrittsanzeige
                progress_bar.empty()
                
                # Erstelle ein DataFrame aus den Ergebnissen
                results_df = pd.DataFrame(results)
                
                # Sortiere nur, wenn numerische Werte vorhanden sind (keine Fehlermeldungen)
                try:
                    # Filtere zuerst die numerischen Werte
                    numeric_results = [r for r in results if not isinstance(r['Wahrscheinlichkeit'], str)]
                    if numeric_results:
                        # Erstelle ein neues DataFrame und sortiere es
                        numeric_df = pd.DataFrame(numeric_results)
                        numeric_df = numeric_df.sort_values(by='Wahrscheinlichkeit', ascending=False)
                        
                        # Erstelle ein DataFrame für die Fehler
                        error_results = [r for r in results if isinstance(r['Wahrscheinlichkeit'], str)]
                        error_df = pd.DataFrame(error_results) if error_results else pd.DataFrame()
                        
                        # Kombiniere die sortierten numerischen Werte mit den Fehlern
                        results_df = pd.concat([numeric_df, error_df])
                except Exception as sort_err:
                    print(f"Fehler beim Sortieren der Ergebnisse: {sort_err}")
                    # Bei Fehler behalten wir das ursprüngliche DataFrame bei
                
                # Verwende eine leistungsfähigere Visualisierung für die Ergebnisse
                st.subheader("Vergleich der Modellvorhersagen")
                
                # Vor der Visualisierung: Behandle Fehlermeldungen in den Daten
                # Erstelle eine Spalte für Text-Anzeige
                results_df['AnzeigeText'] = results_df['Wahrscheinlichkeit']
                
                # Für numerische Wahrscheinlichkeiten: Formatiere als Prozentsatz
                for i, row in results_df.iterrows():
                    if not isinstance(row['Wahrscheinlichkeit'], str):
                        results_df.at[i, 'AnzeigeText'] = f"{row['Wahrscheinlichkeit']:.2f}%"
                
                # Create horizontal bar chart with plotly (ohne Farbskala)
                fig = px.bar(
                    results_df,
                    y='Modell',
                    x='Wahrscheinlichkeit',
                    text='AnzeigeText',  # Verwende vorformatierten Text
                    orientation='h',
                    title="Kaufwahrscheinlichkeit nach Modell (in %)",
                    hover_data=['Vorhersage'],
                    color_discrete_sequence=['#ff7f0e'] * len(results_df)  # Einheitliche Farbe für alle Balken
                )
                
                # Formatting - Text direkt ohne zusätzliche Formatierung verwenden
                fig.update_traces(
                    texttemplate='%{text}', 
                    textposition='outside'
                )
                
                fig.update_layout(
                    xaxis_title="Kaufwahrscheinlichkeit (%)",
                    yaxis_title="Modell",
                    xaxis=dict(
                        range=[0, 100],
                        tickvals=[0, 25, 50, 75, 100],
                        ticktext=['0%', '25%', '50%', '75%', '100%']
                    ),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detaillierte Tabelle mit den Ergebnissen
                st.subheader("Detaillierte Modellergebnisse")
                
                # Format the probability column to show as percentage
                results_df['Wahrscheinlichkeit'] = results_df['Wahrscheinlichkeit'].apply(lambda x: f"{x:.2f}%")
                
                # Marketing-Manager-freundliche Empfehlungsbox
                # Ermittle das beste Modell (höchste Wahrscheinlichkeit)
                best_model_row = results_df.iloc[0]
                best_model_name = best_model_row['Modell']
                best_prob = float(best_model_row['Wahrscheinlichkeit'].replace('%', ''))
                best_prediction = best_model_row['Vorhersage']
                
                # Erstelle eine ansprechende Empfehlungsbox
                st.markdown(f"""
                <div class="recommendation-box">
                    <div class="recommendation-header">Empfehlung basierend auf unserer KI-Analyse</div>
                    <hr>
                    <p>Unser <b>{best_model_name}</b> Modell prognostiziert mit <b>{best_prob:.1f}%</b> Wahrscheinlichkeit:</p>
                    <div style="text-align: center; margin: 20px 0;">
                        <span style="font-size: 28px; font-weight: 700; color: {'#00BFA5' if best_prediction == 'Kauf' else '#FF5252'};">
                            {best_prediction.upper()}
                        </span>
                    </div>
                    <p style="margin-top: 15px;">
                        {'✅ <b>Marketingempfehlung:</b> Dieser Kunde zeigt eine hohe Kaufbereitschaft. Verstärken Sie den Kontakt und bieten Sie personalisierte Angebote.' 
                        if best_prediction == 'Kauf' else 
                        '⚠️ <b>Marketingempfehlung:</b> Dieser Kunde ist noch unentschlossen. Erwägen Sie Rabattaktionen oder gezielte Ansprache, um die Conversion zu verbessern.'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display the results table
                st.subheader("Detaillierte Modellprognosen")
                st.dataframe(results_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Fehler bei der Vorhersage: {str(e)}")
            st.info("Die Vorhersagefunktion ist momentan nicht verfügbar. Bitte versuchen Sie es später erneut.")
    
    with upload_tab:
        st.subheader("CSV-Datei hochladen")
        
        # Bessere Info-Box für Marketing-Manager
        st.markdown("""
        <div class="highlight-box">
            <h4>Vorteile der Batch-Vorhersage</h4>
            <p>Analysieren Sie hunderte Kundenprofile gleichzeitig, um:</p>
            <ul>
                <li>Kaufbereite Kunden zu identifizieren für gezielte Marketingaktionen</li>
                <li>Ressourcen effizient auf Kunden mit hoher Conversion-Wahrscheinlichkeit zu fokussieren</li>
                <li>Kampagnen-ROI durch datengestützte Entscheidungen zu maximieren</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File uploader
            uploaded_file = st.file_uploader("Wählen Sie eine CSV-Datei", type=["csv"])
            
        with col2:
            # Beispiel-CSV zum Download anbieten
            st.markdown("### Beispieldatei")
            
            with open("beispiel_marketing_daten.csv", "r") as f:
                csv_content = f.read()
                
            st.download_button(
                label="📥 Beispiel-CSV herunterladen",
                data=csv_content,
                file_name="beispiel_marketing_daten.csv",
                mime="text/csv"
            )
            
            st.caption("Verwenden Sie diese Datei als Vorlage für Ihre Kundendaten.")
        
        if uploaded_file is not None:
            try:
                # Lade die hochgeladene Datei
                uploaded_df = pd.read_csv(uploaded_file)
                
                # Zeige eine Vorschau der hochgeladenen Daten
                st.subheader("Vorschau der hochgeladenen Daten")
                st.dataframe(uploaded_df.head(10))
                
                # Überprüfe, ob alle erforderlichen Spalten vorhanden sind
                required_columns = ['Informational', 'BounceRates', 'ExitRates', 'PageValues', 'OperatingSystems']
                missing_columns = [col for col in required_columns if col not in uploaded_df.columns]
                
                if missing_columns:
                    st.error(f"Fehlende Spalten in der hochgeladenen Datei: {', '.join(missing_columns)}")
                    st.info("Bitte stellen Sie sicher, dass Ihre Datei alle erforderlichen Spalten enthält.")
                else:
                    # Button zum Ausführen der Vorhersage
                    if st.button("Batch-Vorhersagen durchführen"):
                        # Vorhersagen für alle Zeilen in der hochgeladenen Datei
                        from model_loader import predict_with_real_model
                        
                        try:
                            # Für Marketing-Manager vereinfachtes Interface
                            st.markdown("""
                            <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-bottom: 15px;">
                                <p style="font-weight: 500; margin-bottom: 5px;">Wir verwenden ein fortschrittliches Machine-Learning-Modell, das auf über 12.000 realen Kundendaten trainiert wurde.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Daten vorbereiten
                            X, y, X_train, X_test, y_train, y_test = prepare_data(df)
                            
                            # Fortschrittsanzeige mit aussagekräftigem Label
                            st.markdown(f"<p style='color: {COLOR_SCHEMA['secondary']}; font-weight: 500;'>Starte KI-Analyse Ihrer Kundendaten...</p>", unsafe_allow_html=True)
                            progress_bar = st.progress(0)
                            
                            with st.spinner("Batch-Vorhersagen werden durchgeführt..."):
                                # Vorhersagen durchführen
                                try:
                                    prob, pred = predict_with_real_model('randomforest', uploaded_df)
                                    
                                    # Ergebnisse zum DataFrame hinzufügen - mit verbesserte Dimensionsbehandlung
                                    result_df = uploaded_df.copy()
                                    
                                    # Prüfe die Dimensionen der Wahrscheinlichkeiten
                                    if isinstance(prob, np.ndarray):
                                        if len(prob.shape) > 1 and prob.shape[1] > 1:
                                            # 2D-Array mit mehreren Spalten (prob hat Form [n_samples, n_classes])
                                            probs_to_assign = prob[:, 1]
                                        else:
                                            # 1D-Array oder 2D-Array mit einer Spalte
                                            probs_to_assign = prob.flatten()
                                    else:
                                        # Einzelner Wert oder Liste
                                        probs_to_assign = np.array(prob).flatten()
                                    
                                    # Stelle sicher, dass wir für jede Zeile eine Wahrscheinlichkeit haben
                                    if len(probs_to_assign) != len(result_df):
                                        # Falls nicht, vervollständige mit dem Mittelwert
                                        probs_mean = np.mean(probs_to_assign)
                                        probs_to_assign = np.full(len(result_df), probs_mean)
                                        
                                    # Stelle auch sicher, dass wir für jede Zeile eine Vorhersage haben
                                    if len(pred) != len(result_df):
                                        # Erzeugt Vorhersagen basierend auf 0.5 Schwellenwert der Wahrscheinlichkeiten
                                        pred_values = (probs_to_assign >= 0.5).astype(int)
                                    else:
                                        pred_values = pred
                                    
                                    # Zuweisen der Werte
                                    result_df['Kaufwahrscheinlichkeit'] = probs_to_assign * 100  # In Prozent umwandeln
                                    result_df['Vorhersage'] = ["Kauf" if p == 1 else "Kein Kauf" for p in pred_values]
                                    
                                    # Fortschrittsanzeige aktualisieren
                                    progress_bar.progress(100)
                                    
                                    # Ergebnisse anzeigen
                                    # Marketing-freundliche Ergebnisdarstellung
                                    st.markdown("""
                                    <div class="highlight-box">
                                        <h3 style="color: #4a4a4a; margin-bottom: 15px;">Kaufvorhersage-Ergebnisse</h3>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Formatierte Tabelle mit verbesserten Spaltenbezeichnungen
                                    display_df = result_df.copy()
                                    display_df['Kaufwahrscheinlichkeit'] = display_df['Kaufwahrscheinlichkeit'].apply(lambda x: f"{x:.2f}%")
                                    display_df = display_df.sort_values('Kaufwahrscheinlichkeit', ascending=False)
                                    
                                    st.dataframe(display_df, use_container_width=True)
                                    
                                    # Ergebnisse zum Download anbieten mit besserem Label
                                    csv_data = export_dataframe_to_csv(display_df)
                                    st.download_button(
                                        label="📊 Ergebnisse als CSV herunterladen",
                                        data=csv_data,
                                        file_name="kundendaten_mit_kaufprognose.csv",
                                        mime="text/csv"
                                    )
                                    
                                    # Statistiken anzeigen
                                    st.subheader("Vorhersagestatistiken")
                                    
                                    # Anzahl der vorhergesagten Käufe
                                    predicted_purchases = sum(pred)
                                    predicted_purchase_rate = predicted_purchases / len(pred) * 100
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Anzahl Datensätze", len(pred))
                                        st.metric("Vorhergesagte Käufe", predicted_purchases)
                                    
                                    with col2:
                                        st.metric("Konversionsrate", f"{predicted_purchase_rate:.2f}%")
                                        avg_prob = result_df['Kaufwahrscheinlichkeit'].mean()
                                        st.metric("Durchschnittliche Kaufwahrscheinlichkeit", f"{avg_prob:.2f}%")
                                    
                                except Exception as batch_err:
                                    st.error(f"ERROR: {str(batch_err)}")
                                    st.info("Verwenden Sie die manuelle Eingabe für Einzelvorhersagen.")
                        
                        except Exception as e:
                            st.error(f"ERROR: {str(e)}")
                            st.info("Bitte stellen Sie sicher, dass Ihre Datei das korrekte Format hat.")
            
            except Exception as e:
                st.error(f"ERROR: {str(e)}")
                st.info("Die Datei konnte nicht geladen werden. Bitte überprüfen Sie das Format.")

if __name__ == "__main__":
    main()
