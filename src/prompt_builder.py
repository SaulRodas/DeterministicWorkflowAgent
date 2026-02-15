# src/prompt_builder.py


from src.retrievers.txt_retriever import retrieve_txt
from src.retrievers.csv_retriever import retrieve_from_csv
from src.retrievers.img_retriever import retrieve_img


"""
## Prompts dinámicos
"""

img_prompt = """
Actúa como un observador visual de alta precisión.
Analizarás la imagen proporcionada para responder a la consulta del usuario.

Reglas:
- Describe o responde basándote solo en lo que es claramente visible en la imagen.
- Si el usuario pregunta por un detalle que no aparece, que está fuera de cuadro o que es ilegible (texto borroso, objetos oscuros), responde: 'La imagen no proporciona información suficiente sobre [detalle específico]'.
- No realices suposiciones sobre lo que podría haber sucedido antes o después de la captura, ni sobre elementos que no están explícitamente presentes.
- Si se te pide identificar un objeto y no hay certeza visual absoluta, describe sus características físicas en lugar de darle un nombre definitivo.
"""


txt_prompt = """
Actúa como un especialista en extracción de información. Se te proporcionará un texto como contexto único para responder a las dudas del usuario.

Reglas:
- Tu respuesta debe provenir únicamente del texto suministrado.
- Si la respuesta no está presente de forma explícita o implícita directa en el texto, debes decir: 'No puedo responder a esto porque la información no está disponible en el texto de referencia'.
- No añadas información adicional, aunque sepas que es cierta en el mundo real.
- Mantén una postura neutral y objetiva.
"""


csv_prompt = """
Actúa como un analista de datos riguroso. Tu única fuente de información son los datos filtrados de un CSV que te proporcionan.

Reglas:
- Responde preguntas basándote exclusivamente en las filas y columnas proporcionadas.
- Si la consulta requiere comparar datos que no están presentes o si la respuesta no se puede deducir de los datos suministrados, responde exactamente: 'La información solicitada no se encuentra en los registros proporcionados'.
- No utilices conocimiento externo ni inventes tendencias.
- Si los datos están incompletos para realizar un cálculo, indícalo claramente.
"""


"""
### Build Messages

Construye:

System prompt dinámico  
Human message con contexto
"""

def build_messages(
    router_output: str,
    user_question: str,
    chat_model,
    data,
    index,
    texts,
    metadatas,
    image_path
):

    if router_output.endswith(".txt"):

        context = retrieve_txt(
            user_question,
            index,
            texts,
            metadatas
        )

        content_system = txt_prompt

        context_str = "\n\n".join(
            [item["content"] for item in context]
        )

        content_human = [
          {"type": "text", "text": user_question},
          {"type": "text", "text": context_str}
        ]


    elif router_output.endswith(".csv"):

        context = retrieve_from_csv(
            chat_model,
            data,
            user_question
        )

        content_system = csv_prompt

        context_str = str(context)

        content_human = [
          {"type": "text", "text": user_question},
          {"type": "text", "text": context_str}
        ]


    elif router_output.endswith(".jpg"):

        context = retrieve_img(image_path)

        content_system = img_prompt

        content_human = [
          {"type": "text", "text": user_question},
          context
        ]


    else:
        raise ValueError(
            f"Tipo de archivo no soportado: {router_output}"
        )


    return content_system, content_human
