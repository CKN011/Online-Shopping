import pandas as pd
import numpy as np

def get_summary_statistics(df):
    """
    Generate summary statistics for numerical columns in the dataset
    
    Args:
        df (pandas.DataFrame): The dataset
        
    Returns:
        pandas.DataFrame: Summary statistics
    """
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    
    # Generate summary statistics
    summary = numerical_df.describe().T
    
    # Add additional statistics
    summary['median'] = numerical_df.median()
    summary['skew'] = numerical_df.skew()
    summary['kurtosis'] = numerical_df.kurtosis()
    
    # Reorder columns
    summary = summary[['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurtosis']]
    
    return summary

def analyze_numerical_features(df):
    """
    Analyze numerical features in the dataset, comparing values between 
    visitors who made a purchase and those who didn't
    
    Args:
        df (pandas.DataFrame): The dataset
        
    Returns:
        pandas.DataFrame: Analysis of numerical features
    """
    try:
        # Sicherstellen, dass die 'Revenue' Spalte vorhanden ist
        if 'Revenue' not in df.columns:
            # Dummy-Ergebnis zurückgeben
            return pd.DataFrame({'Fehler': ['Die Revenue-Spalte fehlt im Datensatz']})
            
        # Kopieren des Dataframes, um Änderungen zu vermeiden
        df_copy = df.copy()
        
        # Select numerical features, außer Revenue
        numerical_cols = [col for col in df_copy.select_dtypes(include=['int64', 'float64']).columns 
                         if col != 'Revenue']
        
        # Wenn keine numerischen Spalten gefunden wurden
        if not numerical_cols:
            return pd.DataFrame({'Hinweis': ['Keine numerischen Merkmale zum Analysieren gefunden']})
        
        # Create a dataframe to store the analysis
        analysis = pd.DataFrame(index=numerical_cols)
        
        # Add mean values for each group
        analysis['Mean (No Purchase)'] = df_copy[df_copy['Revenue'] == 0][numerical_cols].mean()
        analysis['Mean (Purchase)'] = df_copy[df_copy['Revenue'] == 1][numerical_cols].mean()
        
        # Add median values for each group
        analysis['Median (No Purchase)'] = df_copy[df_copy['Revenue'] == 0][numerical_cols].median()
        analysis['Median (Purchase)'] = df_copy[df_copy['Revenue'] == 1][numerical_cols].median()
        
        # Calculate relative difference in means, mit Überprüfung auf Null-Division
        mean_no_purchase = analysis['Mean (No Purchase)']
        mean_purchase = analysis['Mean (Purchase)']
        
        # Sichere Berechnung der prozentualen Differenz
        pct_diff = []
        for col in numerical_cols:
            if mean_no_purchase[col] == 0:
                # Vermeidung von Division durch Null
                pct_diff.append(float('inf') if mean_purchase[col] > 0 else 0)
            else:
                pct_diff.append(((mean_purchase[col] - mean_no_purchase[col]) / mean_no_purchase[col] * 100).round(2))
        
        analysis['Mean Difference (%)'] = pct_diff
        
        # Sort by absolute difference
        try:
            analysis = analysis.sort_values(by='Mean Difference (%)', key=abs, ascending=False)
        except:
            # Falls das Sortieren fehlschlägt, unsortierten DataFrame zurückgeben
            pass
            
        return analysis
    except Exception as e:
        # Bei beliebigen Fehlern eine informative Meldung zurückgeben
        return pd.DataFrame({'Fehler': [f'Fehler bei der Analyse numerischer Merkmale: {str(e)}']})

def analyze_categorical_features(df):
    """
    Analyze categorical features in the dataset, comparing distribution
    between visitors who made a purchase and those who didn't
    
    Args:
        df (pandas.DataFrame): The dataset
        
    Returns:
        pandas.DataFrame: Analysis of categorical features
    """
    try:
        # Sicherstellen, dass die 'Revenue' Spalte vorhanden ist
        if 'Revenue' not in df.columns:
            # Dummy-Ergebnis zurückgeben
            return pd.DataFrame({'Fehler': ['Die Revenue-Spalte fehlt im Datensatz']})
            
        # Kopieren des Dataframes, um Änderungen zu vermeiden
        df_copy = df.copy()
        
        # Vorhandene kategorische Spalten überprüfen
        default_cat_cols = ['Month', 'VisitorType', 'Weekend', 'Browser', 
                          'Region', 'TrafficType', 'OperatingSystems']
        categorical_cols = [col for col in default_cat_cols if col in df_copy.columns]
        
        # Falls keine kategorischen Spalten gefunden wurden
        if not categorical_cols:
            return pd.DataFrame({'Hinweis': ['Keine kategorischen Merkmale zum Analysieren gefunden']})
            
        # Create a list to store analysis for each categorical feature
        all_analyses = []
        
        for col in categorical_cols:
            try:
                # Get purchase rate by category
                purchase_rate = df_copy.groupby(col)['Revenue'].mean().reset_index()
                purchase_rate.columns = [col, 'Purchase Rate']
                
                # Get category distribution
                category_dist = df_copy[col].value_counts(normalize=True).reset_index()
                category_dist.columns = [col, 'Distribution (%)']
                category_dist['Distribution (%)'] = (category_dist['Distribution (%)'] * 100).round(2)
                
                # Merge the two analyses
                analysis = pd.merge(purchase_rate, category_dist, on=col)
                
                # Sort by purchase rate
                analysis = analysis.sort_values(by='Purchase Rate', ascending=False)
                
                # Add the feature name
                analysis['Feature'] = col
                
                # Reorder columns
                analysis = analysis[['Feature', col, 'Purchase Rate', 'Distribution (%)']]
                
                all_analyses.append(analysis)
            except Exception as e:
                # Bei Fehler in einer Spalte diese überspringen
                continue
        
        # Falls keine Analysen generiert wurden
        if not all_analyses:
            return pd.DataFrame({'Hinweis': ['Keine verwertbaren kategorischen Merkmale gefunden']})
        
        # Combine all analyses
        combined_analysis = pd.concat(all_analyses, ignore_index=True)
        
        # Convert purchase rate to percentage
        combined_analysis['Purchase Rate'] = (combined_analysis['Purchase Rate'] * 100).round(2)
        combined_analysis.rename(columns={'Purchase Rate': 'Purchase Rate (%)'}, inplace=True)
        
        return combined_analysis
    except Exception as e:
        # Bei beliebigen Fehlern eine informative Meldung zurückgeben
        return pd.DataFrame({'Fehler': [f'Fehler bei der Analyse kategorischer Merkmale: {str(e)}']})
