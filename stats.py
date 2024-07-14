from flask import Blueprint, render_template, request, jsonify
import mysql.connector
import matplotlib.pyplot as plt
import io
import base64
import logging

bp = Blueprint('stats', __name__)

# Configurar logging
logging.basicConfig(level=logging.DEBUG)

def get_teams():
    table_name = "resultados_2024"
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="liga_pro",
        port=3306
    )
    cursor = conn.cursor()
    query = f"SELECT DISTINCT equipo_local FROM {table_name}"
    cursor.execute(query)
    teams = cursor.fetchall()
    conn.close()
    return [team[0] for team in teams]

def get_encounters_stats(local_team, visitor_team, order):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="liga_pro",
        port=3306
    )
    cursor = conn.cursor()
    stats = []
    years = ['2019', '2020', '2021', '2022', '2023', '2024']
    for year in years:
        table_name = f"resultados_{year}"
        if order == "local_first":
            query = f"""
            SELECT 
                %s AS year,
                equipo_local,
                goles_local,
                goles_visitante,
                equipo_visitante
            FROM {table_name}
            WHERE equipo_local = %s AND equipo_visitante = %s
            """
        else:
            query = f"""
            SELECT 
                %s AS year,
                equipo_visitante AS equipo_local,
                goles_visitante AS goles_local,
                goles_local AS goles_visitante,
                equipo_local AS equipo_visitante
            FROM {table_name}
            WHERE equipo_visitante = %s AND equipo_local = %s
            """
        cursor.execute(query, (year, local_team, visitor_team))
        year_stats = cursor.fetchall()
        for stat in year_stats:
            stats.append({
                'year': year,
                'equipo_local': stat[1],
                'goles_local': stat[2],
                'goles_visitante': stat[3],
                'equipo_visitante': stat[4]
            })
    conn.close()

    logging.debug(f"Encounter stats for {local_team} vs {visitor_team}: {stats}")
    return stats

def get_team_stats(team):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="liga_pro",
        port=3306
    )
    cursor = conn.cursor()
    stats = {
        'victorias': 0,
        'derrotas': 0,
        'empates': 0,
        'total': 0
    }
    years = ['2019', '2020', '2021', '2022', '2023', '2024']
    for year in years:
        table_name = f"resultados_{year}"
        query_local = f"""
        SELECT 
            COUNT(*) AS total,
            SUM(CASE WHEN equipo_local = %s AND goles_local > goles_visitante THEN 1 ELSE 0 END) AS victorias,
            SUM(CASE WHEN equipo_local = %s AND goles_local < goles_visitante THEN 1 ELSE 0 END) AS derrotas,
            SUM(CASE WHEN equipo_local = %s AND goles_local = goles_visitante THEN 1 ELSE 0 END) AS empates
        FROM {table_name}
        WHERE equipo_local = %s OR equipo_visitante = %s
        """
        cursor.execute(query_local, (team, team, team, team, team))
        local_stats = cursor.fetchone()
        if local_stats:
            stats['total'] += local_stats[0] if local_stats[0] else 0
            stats['victorias'] += local_stats[1] if local_stats[1] else 0
            stats['derrotas'] += local_stats[2] if local_stats[2] else 0
            stats['empates'] += local_stats[3] if local_stats[3] else 0

        query_visitor = f"""
        SELECT 
            SUM(CASE WHEN equipo_visitante = %s AND goles_visitante > goles_local THEN 1 ELSE 0 END) AS victorias,
            SUM(CASE WHEN equipo_visitante = %s AND goles_visitante < goles_local THEN 1 ELSE 0 END) AS derrotas,
            SUM(CASE WHEN equipo_visitante = %s AND goles_visitante = goles_local THEN 1 ELSE 0 END) AS empates
        FROM {table_name}
        WHERE equipo_local = %s OR equipo_visitante = %s
        """
        cursor.execute(query_visitor, (team, team, team, team, team))
        visitor_stats = cursor.fetchone()
        if visitor_stats:
            stats['victorias'] += visitor_stats[0] if visitor_stats[0] else 0
            stats['derrotas'] += visitor_stats[1] if visitor_stats[1] else 0
            stats['empates'] += visitor_stats[2] if visitor_stats[2] else 0

    conn.close()

    logging.debug(f"Stats for team {team}: {stats}")
    return stats

def calculate_result(encounters_stats):
    local_wins = sum(1 for stat in encounters_stats if stat['goles_local'] > stat['goles_visitante'])
    visitor_wins = sum(1 for stat in encounters_stats if stat['goles_visitante'] > stat['goles_local'])
    draws = sum(1 for stat in encounters_stats if stat['goles_local'] == stat['goles_visitante'])

    if local_wins > visitor_wins:
        result = "Gana Local"
    elif visitor_wins > local_wins:
        result = "Gana Visitante"
    else:
        result = "Empate"

    return result

def calculate_score(encounters_stats):
    local_goals = sum(stat['goles_local'] for stat in encounters_stats)
    visitor_goals = sum(stat['goles_visitante'] for stat in encounters_stats)
    num_matches = len(encounters_stats)

    if num_matches == 0:
        return (0, 0)

    average_local_goals = round(local_goals / num_matches)
    average_visitor_goals = round(visitor_goals / num_matches)

    return (average_local_goals, average_visitor_goals)

def generate_pie_chart(labels, sizes, title):
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FF5722', '#FFC107'])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@bp.route('/stats', methods=['GET'])
def stats():
    return render_template('stats.html')

@bp.route('/get_teams', methods=['POST'])
def get_teams_endpoint():
    teams = get_teams()
    return jsonify(teams)

@bp.route('/get_stats', methods=['POST'])
def get_stats_endpoint():
    local_team = request.form['equipo_local']
    visitor_team = request.form['equipo_visitante']
    order = request.form['order']
    
    encounters_stats = get_encounters_stats(local_team, visitor_team, order)
    local_stats = get_team_stats(local_team)
    visitor_stats = get_team_stats(visitor_team)
    
    result = calculate_result(encounters_stats)
    score = calculate_score(encounters_stats)

    response = {
        'encounters_stats': encounters_stats,
        'result': result,
        'score': score
    }

    if local_stats['total'] > 0 or visitor_stats['total'] > 0:
        if local_stats['total'] > 0:
            local_pie_chart = generate_pie_chart(
                ['Victorias', 'Derrotas', 'Empates'],
                [local_stats['victorias'], local_stats['derrotas'], local_stats['empates']],
                f"{local_team} (Total)"
            )
            response['local_pie_chart'] = local_pie_chart

        if visitor_stats['total'] > 0:
            visitor_pie_chart = generate_pie_chart(
                ['Victorias', 'Derrotas', 'Empates'],
                [visitor_stats['victorias'], visitor_stats['derrotas'], visitor_stats['empates']],
                f"{visitor_team} (Total)"
            )
            response['visitor_pie_chart'] = visitor_pie_chart
    else:
        response['message'] = "No hay historial para estos equipos."

    logging.debug(f"Response data: {response}")
    return jsonify(response)
