# pronosticos.py

from flask import Blueprint, render_template, request, jsonify
import mysql.connector
from model import model, analyze_probabilities, predict_score, historical_matches, predict_outcome

bp = Blueprint('pronosticos', __name__)

def get_teams():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="liga_pro",
        port=3306
    )
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT equipo_local FROM resultados_2024")
    teams = cursor.fetchall()
    conn.close()
    return [team[0] for team in teams]

def check_existing_result(local_team, visitor_team):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="liga_pro",
        port=3306
    )
    cursor = conn.cursor()
    query = """
    SELECT goles_local, goles_visitante 
    FROM resultados_2024 
    WHERE equipo_local = %s AND equipo_visitante = %s
    """
    cursor.execute(query, (local_team, visitor_team))
    result = cursor.fetchone()
    conn.close()
    return result

@bp.route('/pronostico', methods=['GET', 'POST'])
def pronostico():
    teams = get_teams()
    return render_template('pronostico.html', teams=teams)

@bp.route('/get_prediction', methods=['POST'])
def get_prediction():
    equipo_local = request.form['equipo_local']
    equipo_visitante = request.form['equipo_visitante']

    existing_result = check_existing_result(equipo_local, equipo_visitante)
    if existing_result:
        score = existing_result
        if score[0] > score[1]:
            prediction = 'Gana ' + equipo_local
        elif score[0] < score[1]:
            prediction = 'Gana ' + equipo_visitante
        else:
            prediction = 'Empate'
        prediction_probability = 100
        margin_of_error = 0
        result = {
            'prediction': prediction,
            'score': score,
            'prediction_probability': prediction_probability,
            'margin_of_error': margin_of_error,
            'status': 'Ya jugado'
        }
    else:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="liga_pro",
            port=3306
        )
        cursor = conn.cursor()
        cursor.execute("SELECT Altura FROM estadios_2024 WHERE Equipos_2024 = %s", (equipo_local,))
        altura_local = cursor.fetchone()[0]
        cursor.execute("SELECT Altura FROM estadios_2024 WHERE Equipos_2024 = %s", (equipo_visitante,))
        altura_visitante = cursor.fetchone()[0]

        cursor.execute("SELECT victorias, derrotas FROM tecnicos_2024 WHERE equipo = %s", (equipo_local,))
        local_tecnico = cursor.fetchone()
        local_victorias_tecnico = local_tecnico[0]
        local_derrotas_tecnico = local_tecnico[1]

        cursor.execute("SELECT victorias, derrotas FROM tecnicos_2024 WHERE equipo = %s", (equipo_visitante,))
        visitante_tecnico = cursor.fetchone()
        visitante_victorias_tecnico = visitante_tecnico[0]
        visitante_derrotas_tecnico = visitante_tecnico[1]

        conn.close()

        prediction, probabilities = predict_outcome(equipo_local, equipo_visitante, altura_local, altura_visitante, local_victorias_tecnico, visitante_victorias_tecnico)
        prediction_probability = max(probabilities.values())
        analysis = analyze_probabilities(equipo_local, equipo_visitante)
        score = predict_score(equipo_local, equipo_visitante, prediction)
        history = historical_matches(equipo_local, equipo_visitante)
        margin_of_error = 100 - prediction_probability

        if prediction == 0:
            prediction_text = 'Gana ' + equipo_local
        elif prediction == 1:
            prediction_text = 'Gana ' + equipo_visitante
        else:
            prediction_text = 'Empate'

        result = {
            'prediction': prediction_text,
            'score': score,
            'prediction_probability': prediction_probability,
            'margin_of_error': margin_of_error,
            'analysis': analysis,
            'history': history,
            'status': ''
        }

    return jsonify(result)
