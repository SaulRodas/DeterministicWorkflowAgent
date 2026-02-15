# src/config.py

"""
# Data Sources

Diccionario con las fuentes de datos y una descripcion
de que contiene, en caso de querer escalar a mas documentos
"""

DATA_SOURCES = {
  "datos_clima_mexico.csv":
    "Dataset con temperaturas mensuales por estado en México, incluyendo meses, años y promedios climáticos.",

  "GPT-41_PromptingGuide.txt":
    "Guía técnica sobre prompting para GPT-4.1, incluyendo uso de etiquetas XML, structured prompting y buenas prácticas." ,

  "maiz_info.jpg":
    "Imagen con información visual sobre las razas de maíz nativas de México y su diversidad."
}

VALID_FILES = list(DATA_SOURCES.keys())


# Transforma dict a txt plano
def format_data_sources(data_sources):

  formatted_text = ""

  for file, description in data_sources.items():
    formatted_text += f"{file} → {description}\n"

  return formatted_text


sources_context = format_data_sources(DATA_SOURCES)
