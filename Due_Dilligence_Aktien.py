import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer # Neu: Importiere SimpleImputer

# 1. Datenbeschaffung (erweitert)
# Liste der Rüstungsunternehmen nach Due Diligence (Beispiel)
defense_companies = ['LMT', 'RTX', 'BA', 'GD', 'NOC']  # Lockheed Martin, RTX, Boeing, General Dynamics, Northrop Grumman

# Aktienkursdaten herunterladen
stock_data = yf.download(defense_companies, start="2010-01-01", end="2024-01-01")
stock_prices = stock_data['Close']

# Beispielhafte Konfliktdaten (aus ACLED oder OCHA, in der Realität aus einer Datei oder API)
conflict_data = {
    'Date': pd.to_datetime(['2010-01-01', '2010-06-15', '2011-03-01', '2012-11-14', '2014-07-08', '2021-05-10', '2023-10-07']),
    'Conflict_Intensity': [1, 2, 3, 4, 5, 4, 5],  # Beispiel: Intensität auf einer Skala von 1 bis 5
    'Fatalities': [10, 25, 50, 100, 200, 150, 300]
}
conflict_df = pd.DataFrame(conflict_data)
conflict_df.set_index('Date', inplace=True)

# Beispielhafte aggregierte Nachrichtendaten (aus NLP-Analyse, in der Realität viel komplexer)
news_data = {
    'Date': pd.to_datetime(['2010-01-01', '2010-03-01', '2011-01-01', '2012-05-01', '2014-01-01', '2021-01-01', '2023-01-01']),
    'News_Mentions': [50, 60, 80, 100, 120, 110, 150],  # Anzahl der Artikel über Konflikt und Rüstungsunternehmen
    'News_Sentiment': [0.2, 0.3, -0.1, 0.4, -0.2, 0.1, -0.3]  # Durchschnittliches Sentiment (-1 bis 1)
}
news_df = pd.DataFrame(news_data)
news_df.set_index('Date', inplace=True)

# 2. Datenvorverarbeitung

# Fehlende Werte behandeln
stock_prices = stock_prices.fillna(method='ffill')  # Forward fill für Aktienkurse
conflict_df = conflict_df.resample('D').ffill().fillna(0)
news_df = news_df.resample('D').ffill().fillna(0)

# Datenframes zusammenführen
df = pd.concat([stock_prices, conflict_df, news_df], axis=1)

# Option 1: Fehlende Werte durch den Mittelwert ersetzen (Imputation)
imputer = SimpleImputer(strategy='mean')  # oder 'median', 'most_frequent', 'constant'
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)


# Option 2: Zeilen mit fehlenden Werten entfernen
df_dropped = df.dropna()

# Wähle eine der Optionen für df_processed
df_processed = df_imputed # oder df_dropped


# Skalierung der Daten
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_processed), columns=df_processed.columns, index=df_processed.index)

# 3. Explorative Datenanalyse und Modellierung (Beispiel: lineare Regression)

# Kovarianzmatrix visualisieren
correlation_matrix = df_scaled.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Korrelationsmatrix')
plt.show()

# Einfache lineare Regression für ein Unternehmen (Beispiel: LMT)
X = df_scaled[['Conflict_Intensity', 'Fatalities', 'News_Mentions', 'News_Sentiment']]
y = df_scaled['LMT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Bewertung des Modells (sehr einfach, für illustrative Zwecke)
score = model.score(X_test, y_test)
print(f"Modell-Score (R^2): {score}")

# Visualisierung (optional)
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Tatsächlicher Preis (LMT)')
plt.plot(y_test.index, y_pred, label='Vorhersage')
plt.title('Aktienkurs von Lockheed Martin vs. Vorhersage')
plt.xlabel('Datum')
plt.ylabel('Skalierter Preis')
plt.legend()
plt.show()
