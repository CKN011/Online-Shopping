import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_correlation_heatmap(df):
    """
    Create a correlation heatmap for numerical features
    
    Args:
        df (pandas.DataFrame): The dataset
        
    Returns:
        matplotlib.figure.Figure: The correlation heatmap figure
    """
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap without colorbar
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        linewidths=0.5,
        ax=ax,
        cbar=False  # Entfernt die Farbskala
    )
    
    # Set title
    ax.set_title('Korrelationsmatrix der numerischen Merkmale', fontsize=16)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_feature_importance(model, feature_names, model_name=''):
    """
    Plot feature importance for a given model
    
    Args:
        model: The trained machine learning model
        feature_names (list): List of feature names
        model_name (str, optional): Name of the model
        
    Returns:
        matplotlib.figure.Figure: The feature importance plot
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get feature importances
    if model_name == 'Logistic Regression':
        # For Logistic Regression, use coefficients as importance
        importances = np.abs(model.coef_[0])
    else:
        # For tree-based models, use feature_importances_
        importances = model.feature_importances_
    
    # Sort importances
    indices = np.argsort(importances)
    
    # Plot feature importances
    plt.barh(range(len(indices)), importances[indices], align='center')
    
    # Add feature names as y-tick labels
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    
    # Add labels and title
    plt.xlabel('Relative Wichtigkeit')
    plt.title('Merkmalswichtigkeit' + (f' ({model_name})' if model_name else ''))
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_pairplot(df, selected_features, hue='Revenue'):
    """
    Create a pairplot of selected features with hue based on Revenue
    
    Args:
        df (pandas.DataFrame): The dataset
        selected_features (list): List of features to include in the pairplot
        hue (str, optional): Column to use for coloring points
        
    Returns:
        matplotlib.figure.Figure: The pairplot figure
    """
    # Create a subset of the data with selected features and hue
    subset_df = df[selected_features + [hue]].copy()
    
    # Create the pairplot
    g = sns.pairplot(
        subset_df, 
        hue=hue, 
        palette=['#636EFA', '#EF553B'],
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k', 'linewidth': 0.5},
        diag_kws={'alpha': 0.5, 'linewidth': 2}
    )
    
    # Add a title
    g.fig.suptitle('Paarplot der ausgewählten Merkmale', y=1.02, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    return g.fig

def plot_revenue_by_visitor_type(df):
    """
    Create charts showing visitor type analytics:
    1. Conversion rates by visitor type
    2. Distribution of visitor types in the dataset
    
    Args:
        df (pandas.DataFrame): The dataset
        
    Returns:
        plotly.graph_objects.Figure: The figure with both charts
    """
    # Prüfen Sie, ob die Spalte 'VisitorType' im Datensatz existiert
    if 'VisitorType' not in df.columns:
        # Erstellen Sie ein leeres Figure-Objekt mit einer Fehlermeldung
        fig = go.Figure()
        fig.update_layout(
            title="Keine Besuchertypdaten verfügbar",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        # Text in der Mitte hinzufügen
        fig.add_trace(go.Scatter(
            x=[0.5],
            y=[0.5],
            text=["Keine Besuchertypdaten verfügbar"],
            mode="text",
            textfont=dict(size=20)
        ))
        return fig
    
    # Map visitor types to more readable labels
    visitor_map = {
        'Returning_Visitor': 'Wiederkehrender Besucher',
        'New_Visitor': 'Neuer Besucher',
        'Other': 'Andere'
    }
    
    # Erstelle eine Kopie mit den übersetzten Besuchertypen
    df_temp = df.copy()
    df_temp['VisitorType_DE'] = df_temp['VisitorType'].map(visitor_map)
    
    # Erstelle Subplot mit 2 Grafiken
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Konversionsrate nach Besuchertyp", 
            "Verteilung der Besuchertypen im Datensatz"
        ),
        specs=[[{"type": "bar"}, {"type": "pie"}]]
    )
    
    # 1. Berechne Konversionsrate nach Besuchertyp
    visitor_conversion = df_temp.groupby('VisitorType_DE')['Revenue'].mean().reset_index()
    visitor_conversion['Konversionsrate'] = visitor_conversion['Revenue'] * 100  # Prozent
    
    # Füge Balkendiagramm für Konversionsraten hinzu
    fig.add_trace(
        go.Bar(
            x=visitor_conversion['VisitorType_DE'],
            y=visitor_conversion['Konversionsrate'],
            text=visitor_conversion['Konversionsrate'].round(2),
            texttemplate='%{text}%',
            textposition='outside',
            name='Konversionsrate',
            marker_color=['#66c2a5', '#fc8d62', '#8da0cb']
        ),
        row=1, col=1
    )
    
    # 2. Berechne Verteilung der Besuchertypen
    visitor_count = df_temp['VisitorType_DE'].value_counts().reset_index()
    visitor_count.columns = ['VisitorType_DE', 'Anzahl']
    visitor_count['Prozent'] = (visitor_count['Anzahl'] / visitor_count['Anzahl'].sum() * 100).round(2)
    
    # Erstelle Beschriftungen mit Prozentwerten
    labels = [f"{typ} ({prozent}%)" for typ, prozent in zip(visitor_count['VisitorType_DE'], visitor_count['Prozent'])]
    
    # Füge Tortendiagramm für Verteilung hinzu
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=visitor_count['Anzahl'],
            name='Besuchertypen',
            marker_colors=['#66c2a5', '#fc8d62', '#8da0cb'],
            textinfo='label+percent',
            insidetextorientation='radial'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Analyse nach Besuchertyp",
        height=500,
        showlegend=False
    )
    
    # Update y-axis for conversion rate
    fig.update_yaxes(title_text="Konversionsrate (%)", row=1, col=1)
    
    # Update x-axis for conversion rate
    fig.update_xaxes(title_text="Besuchertyp", row=1, col=1)
    
    # Füge eine Erklärung hinzu
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        text="Hinweis: Die Konversionsrate zeigt den Prozentsatz der Besucher jedes Typs, die einen Kauf getätigt haben.",
        showarrow=False,
        font=dict(size=12),
        bordercolor="#c7c7c7",
        borderwidth=1,
        borderpad=4,
        bgcolor="#f0f0f0",
        opacity=0.8
    )
    
    return fig

def plot_purchase_by_month(df):
    """
    Create a line chart showing purchase rates by month
    
    Args:
        df (pandas.DataFrame): The dataset
        
    Returns:
        plotly.graph_objects.Figure: The line chart figure
    """
    try:
        # Prüfen Sie, ob die Spalte 'Month' im Datensatz existiert
        if 'Month' not in df.columns:
            raise ValueError("Keine Monatsdaten verfügbar")
        
        # Vereinfachte Version der Monatsanalyse
        from plotly.subplots import make_subplots
        
        # Definiere die korrekte Monatsreihenfolge für das UCI-Dataset
        # Der UCI-Datensatz enthält nur diese 10 Monate, also müssen wir sie in die richtige Reihenfolge bringen
        month_order = ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_map = {m: i for i, m in enumerate(month_order)}
        
        # Manuell alle Monate mit ihren Werten berechnen
        data_points = []
        for month in df['Month'].unique():
            month_df = df[df['Month'] == month]
            conversion_rate = month_df['Revenue'].mean() * 100
            visit_count = len(month_df)
            sort_idx = month_map.get(month, 999)  # Standard-Index für unbekannte Monate
            
            data_points.append({
                'Month': month,
                'Konversionsrate': conversion_rate,
                'Besuchsanzahl': visit_count,
                'sort_idx': sort_idx
            })
        
        # Daten nach dem Index sortieren
        data_points.sort(key=lambda x: x['sort_idx'])
        
        # Listen für das Diagramm erstellen
        months = [p['Month'] for p in data_points]
        conversions = [p['Konversionsrate'] for p in data_points]
        visits = [p['Besuchsanzahl'] for p in data_points]
        
        # Figure mit zwei y-Achsen erstellen
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Konversionsraten-Linie hinzufügen
        fig.add_trace(
            go.Scatter(
                x=months,
                y=conversions,
                mode='lines+markers',
                name='Konversionsrate',
                line=dict(color='#636EFA', width=3),
                marker=dict(size=10)
            ),
            secondary_y=False
        )
        
        # Besuchsanzahl-Balken hinzufügen
        fig.add_trace(
            go.Bar(
                x=months,
                y=visits,
                name='Besuchsanzahl',
                marker_color='rgba(99, 110, 250, 0.3)'
            ),
            secondary_y=True
        )
        
        # Layout aktualisieren
        fig.update_layout(
            title='Konversionsrate und Besuchsanzahl nach Monat',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        # Y-Achsenbeschriftungen aktualisieren
        fig.update_yaxes(title_text="Konversionsrate (%)", secondary_y=False)
        fig.update_yaxes(title_text="Besuchsanzahl", secondary_y=True)
        
        return fig
        
    except Exception as e:
        # Bei Fehlern ein leeres Diagramm mit Fehlermeldung anzeigen
        fig = go.Figure()
        fig.update_layout(
            title="Fehler bei Monatsanalyse",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        fig.add_trace(go.Scatter(
            x=[0.5],
            y=[0.5],
            text=[f"Fehler bei der Monatsanalyse: {str(e)}"],
            mode="text",
            textfont=dict(size=16, color="red")
        ))
        return fig

def plot_conversion_by_traffic_source(df):
    """
    Create a bar chart showing conversion rates by traffic source
    
    Args:
        df (pandas.DataFrame): The dataset
        
    Returns:
        plotly.graph_objects.Figure: The bar chart figure
    """
    # Prüfen Sie, ob die Spalte 'TrafficType' im Datensatz existiert
    if 'TrafficType' not in df.columns:
        # Erstellen Sie ein leeres Figure-Objekt mit einer Fehlermeldung
        fig = go.Figure()
        fig.update_layout(
            title="Keine Traffic-Quellendaten verfügbar",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        # Text in der Mitte hinzufügen
        fig.add_trace(go.Scatter(
            x=[0.5],
            y=[0.5],
            text=["Keine Traffic-Quellendaten verfügbar"],
            mode="text",
            textfont=dict(size=20)
        ))
        return fig
    
    # Calculate conversion rate by traffic type
    traffic_revenue = df.groupby('TrafficType')['Revenue'].agg(['mean', 'count']).reset_index()
    traffic_revenue['mean'] = traffic_revenue['mean'] * 100  # Convert to percentage
    
    # Sort by conversion rate
    traffic_revenue = traffic_revenue.sort_values('mean', ascending=False)
    
    # Take top 10 traffic sources by visit count
    top_traffic = traffic_revenue.nlargest(10, 'count')
    
    # Create the bar chart
    fig = px.bar(
        top_traffic,
        x='TrafficType',
        y='mean',
        title='Konversionsrate nach Traffic-Quelle (Top 10 nach Besuchsanzahl)',
        labels={'mean': 'Konversionsrate (%)', 'TrafficType': 'Traffic-Quelle', 'count': 'Besuchsanzahl'},
        text=top_traffic['mean'].round(2),
        color='count',
        color_continuous_scale='Viridis'
    )
    
    # Update text position and format
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    
    # Update layout
    fig.update_layout(
        xaxis_title='Traffic-Quelle',
        yaxis_title='Konversionsrate (%)',
        coloraxis_colorbar_title='Besuchsanzahl'
    )
    
    return fig

def plot_categorical_distribution(df, column, hue='Revenue'):
    """
    Create a stacked bar chart showing the distribution of a categorical variable
    with segments colored by Revenue
    
    Args:
        df (pandas.DataFrame): The dataset
        column (str): The categorical column to visualize
        hue (str, optional): Column to use for coloring segments
        
    Returns:
        plotly.graph_objects.Figure: The stacked bar chart figure
    """
    # Prüfen Sie, ob die angegebene Spalte im Datensatz existiert
    if column not in df.columns or hue not in df.columns:
        # Erstellen Sie ein leeres Figure-Objekt mit einer Fehlermeldung
        fig = go.Figure()
        fig.update_layout(
            title=f"Spalte '{column}' oder '{hue}' nicht im Datensatz verfügbar",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        # Text in der Mitte hinzufügen
        fig.add_trace(go.Scatter(
            x=[0.5],
            y=[0.5],
            text=[f"Spalte '{column}' oder '{hue}' nicht im Datensatz verfügbar"],
            mode="text",
            textfont=dict(size=20)
        ))
        return fig
        
    # Calculate the count and percentage of each category
    counts = df.groupby([column, hue]).size().reset_index(name='count')
    total = counts.groupby(column)['count'].transform('sum')
    counts['percentage'] = counts['count'] / total * 100
    
    # Create the stacked bar chart
    fig = px.bar(
        counts,
        x=column,
        y='percentage',
        color=hue,
        title=f'Verteilung von {column} nach Kaufabsicht',
        labels={'percentage': 'Prozent (%)', hue: 'Kaufabsicht'},
        color_discrete_map={0: '#636EFA', 1: '#EF553B'}
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=column,
        yaxis_title='Prozent (%)',
        barmode='stack',
        legend_title='Kaufabsicht'
    )
    
    return fig
