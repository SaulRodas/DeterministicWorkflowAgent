# src/generator.py

from langchain_core.messages import SystemMessage, HumanMessage


"""
### Response LLM

Genera la respuesta cruda del modelo
usando los mensajes construidos
"""

def raw_response_llm(
    chat_model,
    content_system,
    content_human
):

  messages = [
    SystemMessage(content=content_system),
    HumanMessage(content=content_human)
  ]

  response = chat_model.invoke(messages)

  return response.content[0]['text']
