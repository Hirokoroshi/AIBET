# model.py

# Importaciones necesarias
import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Definir la función get_historical_data, balance_data, get_last_five_games, train_initial_model, etc.

def get_historical_data():
    engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/liga_pro')
    query = """
    SELECT r.fecha, r.equipo_local, r.equipo_visitante, r.goles_local, r.goles_visitante, 
           e_local.Altura AS altura_local, e_visitante.Altura AS altura_visitante, 
           s_local.porcentaje_victorias AS local_porcentaje_victorias, 
           s_local.tiros_a_puerta AS local_tiros_a_puerta, 
           s_local.posesion AS local_posesion,
           s_visitante.porcentaje_victorias AS visitante_porcentaje_victorias, 
           s_visitante.tiros_a_puerta AS visitante_tiros_a_puerta, 
           s_visitante.posesion AS visitante_posesion,
           t_local.victorias AS local_victorias_tecnico, 
           t_local.derrotas AS local_derrotas_tecnico,
           t_visitante.victorias AS visitante_victorias_tecnico, 
           t_visitante.derrotas AS visitante_derrotas_tecnico
    FROM (
        SELECT fecha, equipo_local, equipo_visitante, goles_local, goles_visitante FROM resultados_2019
        UNION
        SELECT fecha, equipo_local, equipo_visitante, goles_local, goles_visitante FROM resultados_2020
        UNION
        SELECT fecha, equipo_local, equipo_visitante, goles_local, goles_visitante FROM resultados_2021
        UNION
        SELECT fecha, equipo_local, equipo_visitante, goles_local, goles_visitante FROM resultados_2022
        UNION
        SELECT fecha, equipo_local, equipo_visitante, goles_local, goles_visitante FROM resultados_2023
        UNION
        SELECT fecha, equipo_local, equipo_visitante, goles_local, goles_visitante FROM resultados_2024
    ) AS r
    JOIN estadios_2024 AS e_local ON r.equipo_local = e_local.Equipos_2024
    JOIN estadios_2024 AS e_visitante ON r.equipo_visitante = e_visitante.Equipos_2024
    JOIN (
        SELECT equipo, AVG(porcentaje_victorias) AS porcentaje_victorias, AVG(tiros_a_puerta) AS tiros_a_puerta, AVG(posesion) AS posesion 
        FROM (
            SELECT equipo, porcentaje_victorias, tiros_a_puerta, posesion FROM stats_2020
            UNION
            SELECT equipo, porcentaje_victorias, tiros_a_puerta, posesion FROM stats_2021
            UNION
            SELECT equipo, porcentaje_victorias, tiros_a_puerta, posesion FROM stats_2022
            UNION
            SELECT equipo, porcentaje_victorias, tiros_a_puerta, posesion FROM stats_2023
            UNION
            SELECT equipo, porcentaje_victorias, tiros_a_puerta, posesion FROM stats_2024
        ) AS stats
        GROUP BY equipo
    ) AS s_local ON r.equipo_local = s_local.equipo
    JOIN (
        SELECT equipo, AVG(porcentaje_victorias) AS porcentaje_victorias, AVG(tiros_a_puerta) AS tiros_a_puerta, AVG(posesion) AS posesion 
        FROM (
            SELECT equipo, porcentaje_victorias, tiros_a_puerta, posesion FROM stats_2020
            UNION
            SELECT equipo, porcentaje_victorias, tiros_a_puerta, posesion FROM stats_2021
            UNION
            SELECT equipo, porcentaje_victorias, tiros_a_puerta, posesion FROM stats_2022
            UNION
            SELECT equipo, porcentaje_victorias, tiros_a_puerta, posesion FROM stats_2023
            UNION
            SELECT equipo, porcentaje_victorias, tiros_a_puerta, posesion FROM stats_2024
        ) AS stats
        GROUP BY equipo
    ) AS s_visitante ON r.equipo_visitante = s_visitante.equipo
    JOIN (
        SELECT equipo, AVG(victorias) AS victorias, AVG(derrotas) AS derrotas 
        FROM tecnicos_2024
        GROUP BY equipo
    ) AS t_local ON r.equipo_local = t_local.equipo
    JOIN (
        SELECT equipo, AVG(victorias) AS victorias, AVG(derrotas) AS derrotas 
        FROM tecnicos_2024
        GROUP BY equipo
    ) AS t_visitante ON r.equipo_visitante = t_visitante.equipo
    """
    df = pd.read_sql(query, engine)
    return df

def balance_data(df):
    df_local_win = df[df['result'] == 0]
    df_visitor_win = df[df['result'] == 1]
    df_draw = df[df['result'] == 2]

    min_size = min(len(df_local_win), len(df_visitor_win), len(df_draw))

    df_local_win_balanced = df_local_win.sample(n=min_size, random_state=42, replace=False)
    df_visitor_win_balanced = df_visitor_win.sample(n=min_size, random_state=42, replace=False)
    df_draw_balanced = df_draw.sample(n=min_size, random_state=42, replace=False)

    df_balanced = pd.concat([df_local_win_balanced, df_visitor_win_balanced, df_draw_balanced])
    return df_balanced

def get_last_five_games(df, team, is_local=True):
    if is_local:
        matches = df[(df['equipo_local'] == team)].sort_values(by='fecha', ascending=False).head(5)
    else:
        matches = df[(df['equipo_visitante'] == team)].sort_values(by='fecha', ascending=False).head(5)
    return matches

def train_initial_model():
    df = get_historical_data()
    df['result'] = df.apply(lambda row: 0 if row['goles_local'] > row['goles_visitante'] else (1 if row['goles_visitante'] > row['goles_local'] else 2), axis=1)

    df = balance_data(df)

    X = []
    y = []
    
    teams = df['equipo_local'].unique()
    for team in teams:
        local_games = get_last_five_games(df, team, is_local=True)
        visitor_games = get_last_five_games(df, team, is_local=False)
        
        if len(local_games) == 5:
            X.append(local_games[['altura_local', 'altura_visitante', 'local_porcentaje_victorias', 'local_tiros_a_puerta', 'local_posesion', 'visitante_porcentaje_victorias', 'visitante_tiros_a_puerta', 'visitante_posesion', 'local_victorias_tecnico', 'local_derrotas_tecnico', 'visitante_victorias_tecnico', 'visitante_derrotas_tecnico']].mean().values)
            y.append(local_games['result'].mode()[0])
        
        if len(visitor_games) == 5:
            X.append(visitor_games[['altura_local', 'altura_visitante', 'local_porcentaje_victorias', 'local_tiros_a_puerta', 'local_posesion', 'visitante_porcentaje_victorias', 'visitante_tiros_a_puerta', 'visitante_posesion', 'local_victorias_tecnico', 'local_derrotas_tecnico', 'visitante_victorias_tecnico', 'visitante_derrotas_tecnico']].mean().values)
            y.append(visitor_games['result'].mode()[0])

    X = pd.DataFrame(X, columns=['altura_local', 'altura_visitante', 'local_porcentaje_victorias', 'local_tiros_a_puerta', 'local_posesion', 'visitante_porcentaje_victorias', 'visitante_tiros_a_puerta', 'visitante_posesion', 'local_victorias_tecnico', 'local_derrotas_tecnico', 'visitante_victorias_tecnico', 'visitante_derrotas_tecnico'])
    y = pd.Series(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True]
    }

    model = RandomForestClassifier(random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=rf_param_grid, cv=skf, n_jobs=-1, verbose=2, scoring='accuracy')
    
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    accuracy = best_model.score(X, y)
    print(f"Best RandomForest Model Accuracy: {accuracy}")
    joblib.dump(best_model, "best_model.pkl")

    return best_model, df

model, df = train_initial_model()

def analyze_probabilities(local_team, visitor_team):
    local_wins = df[(df['equipo_local'] == local_team) & (df['result'] == 0)].shape[0]
    visitor_wins = df[(df['equipo_visitante'] == visitor_team) & (df['result'] == 1)].shape[0]
    draws = df[(df['equipo_local'] == local_team) & (df['equipo_visitante'] == visitor_team) & (df['result'] == 2)].shape[0]

    total_matches = local_wins + visitor_wins + draws
    if total_matches == 0:
        return {"local": 0, "visitor": 0, "draw": 0}

    local_prob = round((local_wins / total_matches) * 100, 2)
    visitor_prob = round((visitor_wins / total_matches) * 100, 2)
    draw_prob = round((draws / total_matches) * 100, 2)

    return {"local": local_prob, "visitor": visitor_prob, "draw": draw_prob}

def predict_outcome(local_team, visitor_team, altura_local, altura_visitante, local_victorias_tecnico, visitante_victorias_tecnico):
    probabilities = analyze_probabilities(local_team, visitor_team)

    total_prob = probabilities['local'] + probabilities['visitor'] + probabilities['draw']
    if total_prob > 0:
        probabilities['local'] = (probabilities['local'] / total_prob) * 100
        probabilities['visitor'] = (probabilities['visitor'] / total_prob) * 100
        probabilities['draw'] = (probabilities['draw'] / total_prob) * 100

    if altura_local > altura_visitante:
        probabilities['local'] += 5
        probabilities['visitor'] -= 5
    elif altura_visitante > altura_local:
        probabilities['visitor'] += 10
        probabilities['local'] -= 10

    probabilities['local'] += local_victorias_tecnico * 0.1
    probabilities['visitor'] += visitante_victorias_tecnico * 0.1

    total_prob = probabilities['local'] + probabilities['visitor'] + probabilities['draw']
    probabilities['local'] = (probabilities['local'] / total_prob) * 100
    probabilities['visitor'] = (probabilities['visitor'] / total_prob) * 100
    probabilities['draw'] = (probabilities['draw'] / total_prob) * 100

    if probabilities['local'] >= 75:
        prediction = 0
    elif probabilities['visitor'] >= 75:
        prediction = 1
    elif probabilities['draw'] >= 75:
        prediction = 2
    else:
        if probabilities['local'] > probabilities['visitor'] and probabilities['local'] > probabilities['draw']:
            prediction = 0
        elif probabilities['visitor'] > probabilities['local'] and probabilities['visitor'] > probabilities['draw']:
            prediction = 1
        else:
            prediction = 2
    return prediction, probabilities

def predict_score(local_team, visitor_team, prediction):
    if prediction == 0:
        local_goals = df[df['equipo_local'] == local_team]['goles_local'].mean()
        visitor_goals = df[df['equipo_visitante'] == visitor_team]['goles_visitante'].mean()
        return max(int(round(local_goals)), 1), min(int(round(visitor_goals)), int(round(local_goals)) - 1)
    elif prediction == 1:
        local_goals = df[df['equipo_local'] == local_team]['goles_local'].mean()
        visitor_goals = df[df['equipo_visitante'] == visitor_team]['goles_visitante'].mean()
        return min(int(round(local_goals)), int(round(visitor_goals)) - 1), max(int(round(visitor_goals)), 1)
    else:
        average_goals = int(round((df[df['equipo_local'] == local_team]['goles_local'].mean() + df[df['equipo_visitante'] == visitor_team]['goles_visitante'].mean()) / 2))
        return average_goals, average_goals

def historical_matches(local_team, visitor_team):
    matches = df[(df['equipo_local'] == local_team) & (df['equipo_visitante'] == visitor_team)]
    local_wins = matches[matches['result'] == 0].shape[0]
    visitor_wins = matches[matches['result'] == 1].shape[0]
    draws = matches[matches['result'] == 2].shape[0]
    total_matches = matches.shape[0]

    return {"total": total_matches, "local_wins": local_wins, "visitor_wins": visitor_wins, "draws": draws}

# Asegúrate de que predict_score esté definida antes de ser importada en pronosticos.py.
    