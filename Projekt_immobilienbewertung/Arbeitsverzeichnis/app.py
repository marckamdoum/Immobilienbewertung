from flask import Flask, request, jsonify, render_template
import pandas as pd
import subprocess
import json
import os
import sys

# Upload-Ordner Pfad festlegen und Ordner anlegen, falls nicht vorhanden
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Pfad zur JSON-Datei für Analyseergebnisse
JSON_PATH = r"C:\Users\Administrator\Documents\07_IEDSCML2_Machine_Learning II\Übungen\Projekt_1_neu\Projekt_1\Arbeitsverzeichnis\analysis_results.json"

app = Flask(__name__)
database = pd.DataFrame()

# Pfad zur JSON-Datei (anpassen!)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = r"C:\Users\Administrator\Documents\07_IEDSCML2_Machine_Learning II\Übungen\Projekt_1_neu\Projekt_1\Arbeitsverzeichnis\analysis_results.json"


@app.route('/')
def index():
    global database
    table_html = database.to_html(index=False) if not database.empty else "No data available."

    analysis_results = None
    try:
        if os.path.exists('analysis_results.json'):
            with open('analysis_results.json', 'r') as f:
                analysis_results = json.load(f)
                print("Analyse-Ergebnisse geladen:", analysis_results)
        else:
            print("Keine analysis_results.json gefunden.")
    except Exception as e:
        print("Fehler beim Laden der Analyseergebnisse:", e)

    return render_template('index.html', table=table_html, analysis_results=analysis_results)

@app.route('/upload', methods=['POST'])
def upload_data_web():
    global database
    if 'file' not in request.files:
        return "No file provided", 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return "Only CSV files are allowed", 400

    try:
        # Speichere Datei im Upload-Ordner
        save_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(save_path)

        # Optional: Daten in DataFrame laden
        data = pd.read_csv(save_path)
        database = pd.concat([database, data], ignore_index=True)

        return "Data uploaded and saved successfully. <a href='/'>Zurück</a>", 201
    except Exception as e:
        return f"Failed to process CSV: {str(e)}. <a href='/'>Zurück</a>", 400

from flask import redirect, url_for

@app.route('/analyze', methods=['POST'])
def analyze_data():
    try:
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analyze_data.py')
        result = subprocess.run(
            [sys.executable, script_path],  # nutzt denselben Python-Interpreter wie app.py
            capture_output=True,
            text=True,
            check=True
        )
        print("Analyse erfolgreich. Ausgabe:\n", result.stdout)
        if result.stderr:
            print("Warnung/Fehlerausgabe:\n", result.stderr)

        # Nach erfolgreicher Analyse weiterleiten zur Ergebnisse-Seite
        return redirect(url_for('show_results'))

    except subprocess.CalledProcessError as e:
        print("Analyse fehlgeschlagen.")
        print("stdout:\n", e.stdout)
        print("stderr:\n", e.stderr)
        return f"""
            <h3>Analyse fehlgeschlagen</h3>
            <pre>{e.stderr}</pre>
            <a href='/'>Zurück</a>
        """, 400
    except Exception as ex:
        print("Unerwarteter Fehler beim Ausführen der Analyse:", ex)
        return f"Interner Fehler: {str(ex)}. <a href='/'>Zurück</a>", 500

@app.route('/api/daten', methods=['GET'])
def get_data():
    global database
    if database.empty:
        return jsonify({'message': 'No data available'}), 404
    return database.to_json(orient='records')

@app.route('/results')
def show_results():
    if not os.path.exists(JSON_PATH):
        return "<h2>Analyseergebnisse nicht gefunden.</h2>", 404

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    mse = round(data.get('mse', 0), 2)
    rmse = round(data.get('rmse', 0), 2)
    coefficients = data.get('coefficients', [])
    intercept = round(data.get('intercept', 0), 2)
    predictions = data.get('predictions', [])

    return render_template(
        'results.html',
        mse=mse,
        rmse=rmse,
        coefficients=coefficients,
        intercept=intercept,
        predictions=predictions
    )

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
