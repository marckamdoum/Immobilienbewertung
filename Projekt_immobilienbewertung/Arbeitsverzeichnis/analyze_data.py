# analyze_data.py
# Importiere benötigte Bibliotheken
import requests  # Zum Abrufen von Daten von der API
import pandas as pd  # Für die Datenverarbeitung mit DataFrames
from sklearn.model_selection import train_test_split  # Zum Aufteilen der Daten in Trainings- und Testsets
from sklearn.linear_model import LinearRegression  # Für das lineare Regressionsmodell
from sklearn.metrics import mean_squared_error  # Zur Bewertung des Modells (Mittlerer Quadratischer Fehler)
import sys  # Für sys.exit()
import json  # Zum Speichern der Ergebnisse als JSON
import os
print("Arbeitsverzeichnis:", os.getcwd())

print("Starte Datenabruf und Analyse für Tag 2...\n")

# =================================================================
# 1. DATEN VON DER API ABRUFEN
# =================================================================
# Definiere die URL des API-Endpunkts
API_URL = "http://127.0.0.1:5000/api/daten"
data = None  # Initialisiere data mit None

try:
    print(f"Versuche, Daten von API abzurufen: {API_URL}")
    response = requests.get(API_URL, timeout=10)  # Timeout hinzugefügt
    response.raise_for_status()  # Löst eine HTTPError-Exception für 4XX/5XX Statuscodes aus
    # Konvertiere die JSON-Antwort in einen Pandas DataFrame
    # Es wird erwartet, dass die API eine Liste von Dictionaries zurückgibt
    json_data = response.json()
    if isinstance(json_data, list) and json_data:  # Prüfen, ob es eine nicht-leere Liste ist
        data = pd.DataFrame(json_data)
        print("Daten erfolgreich von der API abgerufen:")
        print(data.head())
        print(f"\nAnzahl der abgerufenen Datensätze: {len(data)}")
    elif isinstance(json_data, dict) and json_data.get('message') == 'No data available':
        print("API meldet: Keine Daten verfügbar.")
        data = pd.DataFrame()  # Leeres DataFrame, wenn API keine Daten hat
    elif not json_data:
        print("API hat eine leere Datenliste zurückgegeben")
        data = pd.DataFrame()
    else:
        print(f"Unerwartetes Datenformat von der API erhalten: {type(json_data)}")
        data = pd.DataFrame()
except requests.exceptions.Timeout:
    print(f"Fehler: Timeout beim Versuch, die API unter {API_URL} zu erreichen.")
    sys.exit("Skript wird beendet. Bitte API-Verfügbarkeit prüfen.")
except requests.exceptions.ConnectionError:
    print(f"Fehler: Verbindung zur API unter {API_URL} fehlgeschlagen.")
    sys.exit("Stellen Sie sicher, dass die Flask-Anwendung (app.py) läuft. Skript wird beendet.")
except requests.exceptions.HTTPError as http_err:
    print(f"HTTP-Fehler beim Abrufen der Daten: {http_err}")
    sys.exit("Skript wird beendet.")
except requests.exceptions.JSONDecodeError:
    print("Fehler: Die Antwort der API konnte nicht als JSON dekodiert werden.")
    sys.exit("Skript wird beendet.")
except Exception as e:
    print(f"Ein unerwarteter Fehler ist beim API-Abruf aufgetreten: {e}")
    sys.exit("Skript wird beendet.")

# Überprüfen, ob Daten erfolgreich geladen wurden
if data is None or data.empty:
    print("\nKeine Daten zum Verarbeiten vorhanden (entweder von API oder API war leer).")
    sys.exit("Skript wird beendet, da keine Daten für die Analyse vorliegen.")

print("\nSpalten im DataFrame vor der Vorverarbeitung:", data.columns.tolist())

# =================================================================
# 2. DATENVORVERARBEITUNG
# =================================================================
print("\nStarte Datenvorverarbeitung...")
# Definiere Feature- und Zielspaltennamen für bessere Wartbarkeit
FEATURE_COLUMNS = ['Grundstücksgröße', 'Zimmeranzahl', 'Garagenanzahl']
TARGET_COLUMN = 'Hauspreis'
ALL_RELEVANT_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# Überprüfen, ob alle relevanten Spalten vorhanden sind
missing_cols = [col for col in ALL_RELEVANT_COLUMNS if col not in data.columns]
if missing_cols:
    print(f"Fehler: Folgende benötigte Spalten fehlen im DataFrame: {missing_cols}")
    print(f"Verfügbare Spalten sind: {data.columns.tolist()}")
    sys.exit("Skript wird beendet. Bitte Datenquelle überprüfen.")

# Datentypkonvertierung und Umgang mit Fehlern
for col in ALL_RELEVANT_COLUMNS:
    # Versuche, in numerische Werte umzuwandeln, Fehler werden zu NaN
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Fehlende Werte entfernen (nach der Konvertierung)
# Dies entfernt Zeilen, wenn in einer der relevanten Spalten ein NaN-Wert steht
data_cleaned = data.dropna(subset=ALL_RELEVANT_COLUMNS)
print(f"Anzahl der Datensätze vor dem Entfernen fehlender Werte: {len(data)}")
print(f"Anzahl der Datensätze nach dem Entfernen fehlender Werte: {len(data_cleaned)}")

# Überprüfe, ob nach dem Entfernen fehlender Werte noch Daten übrig sind
if data_cleaned.empty:
    print("Keine ausreichenden Daten für die Analyse nach dem Entfernen fehlender Werte vorhanden.")
    sys.exit("Bitte überprüfen Sie Ihre Datenquelle und die Konvertierung.")

# Definiere Features (unabhängige Variablen) und Zielvariable (abhängige Variable)
X = data_cleaned[FEATURE_COLUMNS]
y = data_cleaned[TARGET_COLUMN]
print("Features (X) und Zielvariable (y) definiert.")
print("Beispiel Features (X.head()):\n", X.head())
print("Beispiel Zielvariable (y.head()):\n", y.head())

# =================================================================
# 3. TRAININGS- UND TESTDATEN SPLITTEN
# =================================================================
print("\nSplitting der Daten in Trainings- und Testsets...")
if len(X) < 2:  # Benötigt mindestens 2 Samples für train_test_split
    print("Nicht genügend Daten für das Splitten in Trainings- und Testsets vorhanden.")
    sys.exit("Skript wird beendet.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Trainingsdaten-Größe (X_train): {X_train.shape}")
print(f"Testdaten-Größe (X_test): {X_test.shape}")

# =================================================================
# 4. LINEARE REGRESSION TRAINIEREN
# =================================================================
print("\nTrainiere Lineares Regressionsmodell...")
model = LinearRegression()
model.fit(X_train, y_train)
print("Modell erfolgreich trainiert.")
print(f"Modellkoeffizienten (Steigungen): {model.coef_}")
print(f"Modell-Achsenabschnitt (Intercept): {model.intercept_}")

# =================================================================
# 5. VORHERSAGE
# =================================================================
print("\nFühre Vorhersagen auf den Testdaten durch...")
y_pred = model.predict(X_test)
print("Vorhersagen erstellt.")

# =================================================================
# 6. MODELLBEWERTUNG
# =================================================================
print("\nBewerte das Modell...")
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5  # Root Mean Squared Error ist oft interpretierbarer
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# =================================================================
# 7. BEISPIELVORHERSAGE
# =================================================================
print("\nFühre Beispielvorhersagen für neue Daten durch...")
sample_data_dict = {
    'Grundstücksgröße': [150, 200, 120, 250, 180],
    'Zimmeranzahl': [3, 4, 2, 5, 3],
    'Garagenanzahl': [1, 2, 1, 2, 0]
}
sample_df = pd.DataFrame(sample_data_dict)
# Stelle sicher, dass die Spalten in der gleichen Reihenfolge wie beim Training sind
sample_df_ordered = sample_df[FEATURE_COLUMNS]
predicted_prices = model.predict(sample_df_ordered)
print("Geschätzte Hauspreise für Beispieldaten:")
for i, price in enumerate(predicted_prices):
    print(f"Beispiel {i+1}: Grundstücksgröße={sample_df_ordered.loc[i, 'Grundstücksgröße']}, "
          f"Zimmeranzahl={sample_df_ordered.loc[i, 'Zimmeranzahl']}, "
          f"Garagenanzahl={sample_df_ordered.loc[i, 'Garagenanzahl']} -> "
          f"Geschätzter Hauspreis: {price:.2f} €")

# Erstelle ein Dictionary mit den Analyseergebnissen
analysis_results = {
    'mse': mse,
    'rmse': rmse,
    'coefficients': model.coef_.tolist(),
    'intercept': model.intercept_,
    'predictions': predicted_prices.tolist()
}

# Speichere die Analyseergebnisse als JSON-Datei
with open('analysis_results.json', 'w') as f:
    json.dump(analysis_results, f)

print("\nAnalyse für Tag 2 abgeschlossen.")