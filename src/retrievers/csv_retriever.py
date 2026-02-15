# src/retrievers/csv_retriever.py

import json

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage


"""
# Carga CSV

Carga una copia del csv como dataframe
con un formato mejor establecido de fechas
"""


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=",")

    df["PERIODO"] = pd.to_datetime(df["PERIODO"])
    df["Ano"] = df["PERIODO"].dt.year
    df["Mes"] = df["PERIODO"].dt.month

    return df


"""
### Prompt Parser

LLM convierte pregunta -> consulta estructurada
"""

PARSER_SYSTEM_PROMPT = """
Eres un asistente que convierte preguntas en consultas estructuradas
para un dataset climatico de Mexico.

Reglas estrictas:
- NO respondas la pregunta.
- NO expliques nada.
- SOLO devuelve un JSON valido que cumpla el schema.
- Usa unicamente las operaciones permitidas.
- Si falta informacion, infiere solo lo explicito en la pregunta.

Operaciones permitidas:
- top_n_max_temperature
- top_n_min_temperature
- top_n_avg_temperature

Datos a devolver:
- Entidad
- Operacion
- Mes
- Ano

No devuelvas extras
Meses deben devolverse como numero (1-12).
"""


"""
Convierte la salida cruda del LLM
en un query limpio y usable.
"""


def llm_output_to_query(llm_response: list) -> dict:
    raw_text = llm_response[0]["text"]
    parsed = json.loads(raw_text)

    query = {}
    for key, value in parsed.items():
        query[key] = value

    return query


"""
### Retrieval Core

Ejecuta operacion sobre el dataframe
"""


def top_n_temperatures(df: pd.DataFrame, query: dict, top_n: int = 5):
    operation_config = {
        "top_n_max_temperature": {"column": "TEMP_MAXIMA", "ascending": False},
        "top_n_min_temperature": {"column": "TEMP_MINIMA", "ascending": True},
        "top_n_avg_temperature": {"column": "TEMP_MEDIA", "ascending": False},
    }

    operation = query.get("Operacion")

    if operation not in operation_config:
        return {"error": f"Operacion no soportada: {operation}"}

    config = operation_config[operation]
    filtered = df.copy()

    year = query.get("Ano", query.get("Año"))
    if year is not None:
        filtered = filtered[filtered["Ano"] == year]

    if query.get("Mes") is not None:
        filtered = filtered[filtered["Mes"] == query["Mes"]]

    if query.get("Entidad") is not None:
        filtered = filtered[filtered["ENTIDAD"] == query["Entidad"]]

    if filtered.empty:
        return []

    result = (
        filtered.sort_values(config["column"], ascending=config["ascending"])
        .head(top_n)[["ENTIDAD", config["column"], "Ano", "Mes"]]
    )

    return result.to_dict(orient="records")


"""
### Retriever Wrapper

Funcion que:

1. Llama parser LLM
2. Ejecuta retrieval
3. Devuelve contexto
"""


def retrieve_from_csv(chat_model, data, user_question: str) -> str:
    df = data

    messages = [
        SystemMessage(content=PARSER_SYSTEM_PROMPT),
        HumanMessage(content=user_question),
    ]

    response = chat_model.invoke(messages)

    query = llm_output_to_query(response.content)
    context = top_n_temperatures(df, query)

    return context
