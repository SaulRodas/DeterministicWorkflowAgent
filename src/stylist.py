# src/stylist.py

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


"""
#3.- Stylist

Structured Output Schema
"""

class StyledAnswer(BaseModel):
  """Esquema para la salida estructurada del Stylist."""

  original_answer: str = Field(
      description="La respuesta original generada en el paso anterior."
  )

  dry_answer: str = Field(
      description="Una versión muy concisa, directa y profesional."
  )

  funny_answer: str = Field(
      description="Una versión con humor y mucha personalidad."
  )


"""
System Prompt Stylist
"""

Stilyst_system_prompt = """
Eres un editor de estilo experto ("The Stylist").
Tu tarea es recibir información cruda y transformarla en tres formatos distintos según el esquema solicitado.

1. Original: Mantén exactamente el mismo texto crudo, no hagas ningun cambio o variacion.
2. Dry: Sé extremadamente conciso, elimina adornos, ve al grano (estilo corporativo/científico).
3. Funny: Usa jerga mexicana, emojis y un tono divertido/sarcástico, pero mantén la veracidad de los datos.
"""


"""
Stylist Agent
"""

def stylist_agent(
    chat_model,
    raw_context: str
):

    structured_llm = chat_model.with_structured_output(
        StyledAnswer
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
          Eres un editor de estilo experto ("The Stylist").
          Tu tarea es recibir información cruda y transformarla en tres formatos distintos según el esquema solicitado.

          1. Original: Mantén exactamente el mismo texto crudo, no hagas ningun cambio o variacion.
          2. Dry: Sé extremadamente conciso, elimina adornos, ve al grano (estilo corporativo/científico).
          3. Funny: Usa jerga mexicana, emojis y un tono divertido/sarcástico, pero mantén la veracidad de los datos.
          """
        ),
        (
            "human",
            "Aquí está la información cruda para procesar: {input_text}"
        )
    ])

    chain = prompt | structured_llm

    return chain.invoke({
        "input_text": raw_context
    })
