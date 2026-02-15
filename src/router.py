# src/router.py

import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import DATA_SOURCES, VALID_FILES, sources_context


"""
Prompt para el Router
"""

router_system_prompt = f"""
Eres un Decision Maker encargado de seleccionar que fuente de datos usar
para responder una pregunta del usuario.

Fuentes disponibles:

{sources_context}

Reglas:

- Responde SOLO con el nombre exacto del archivo.
- No expliques tu decision.
- No agregues texto extra.
- No inventes archivos fuera de la lista.
"""


"""
### Router Decision
"""


def router_decision(chat_model, user_question):
    messages = [
        SystemMessage(content=router_system_prompt),
        HumanMessage(content=user_question),
    ]

    response = chat_model.invoke(messages)

    # Dependiendo del proveedor/modelo, content puede llegar
    # como str o como lista de bloques.
    raw_content = response.content
    if isinstance(raw_content, str):
        raw_text = raw_content
    elif isinstance(raw_content, list):
        parts = []
        for item in raw_content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        raw_text = "\n".join(parts)
    else:
        raw_text = str(raw_content)

    normalized = raw_text.strip().strip("`'\"")

    # 1) match exacto
    if normalized in VALID_FILES:
        return normalized

    # 2) match por inclusion (ej: "Archivo: datos_clima_mexico.csv")
    lowered = normalized.lower()
    for valid_file in VALID_FILES:
        if valid_file.lower() in lowered:
            return valid_file

    # 3) limpieza adicional de caracteres raros
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "", normalized)
    if cleaned in VALID_FILES:
        return cleaned

    return "ERROR_INVALID_FILE"
