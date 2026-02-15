# src/workflow.py


from src.router import router_decision
from src.prompt_builder import build_messages
from src.generator import raw_response_llm
from src.stylist import stylist_agent


"""
# Workflow Determinístico

Integra:

1. Router
2. Retriever + Prompt Builder
3. Generator
4. Stylist
"""

def deterministic_workflow(
    user_question: str,
    chat_model,
    data,
    index,
    texts,
    metadatas,
    image_path
):

  # 1. Router
  router_output = router_decision(
      chat_model,
      user_question
  )


  # 2. Retriever + Prompt Builder
  content_system, content_human = build_messages(
      router_output,
      user_question,
      chat_model,
      data,
      index,
      texts,
      metadatas,
      image_path
  )


  # 3. Generator
  raw_context = raw_response_llm(
      chat_model,
      content_system,
      content_human
  )


  # 4. Stylist
  stylist_output = stylist_agent(
      chat_model,
      raw_context
  )


  return stylist_output
