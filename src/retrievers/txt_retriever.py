# src/retrievers/txt_retriever.py

import re

from sentence_transformers import SentenceTransformer
import faiss


"""
## TXT Loader
"""

def load_txt(txt_path: str) -> str:

  try:
    with open(txt_path, "r", encoding="utf-8") as archivo:
      contenido = archivo.read()
      return contenido

  except FileNotFoundError:
    print("Error: El archivo no existe.")
    return None

  except PermissionError:
    print("Error: No tienes permisos para leer este archivo.")
    return None

  except Exception as e:
    print(f"Ocurrió un error: {e}")
    return None


"""
### Markdown Chunking

Chunking basado en:

- Headers markdown
- Code blocks
- Tamaño
- Overlap
"""

def markdown_chunking(
    text,
    chunk_size=2000,
    overlap=200
):

  lines = text.split('\n')
  chunks = []

  # Estado actual
  current_chunk = []
  current_size = 0
  in_code_block = False
  current_headers = {"h1": "", "h2": "", "h3": ""}

  header_pattern = re.compile(r'^(#{1,3})\s+(.*)')


  for line in lines:

    # Detectar code block
    if line.strip().startswith('```'):
      in_code_block = not in_code_block
      current_chunk.append(line)
      current_size += len(line)
      continue


    # Detectar headers
    match = header_pattern.match(line)

    if match and not in_code_block:

      level = len(match.group(1))
      title = match.group(2)

      if current_chunk:
        chunks.append({
          "content": "\n".join(current_chunk),
          "metadata": current_headers.copy()
        })

        current_chunk = []
        current_size = 0


      if level == 1:
        current_headers = {
          "h1": title,
          "h2": "",
          "h3": ""
        }

      elif level == 2:
        current_headers["h2"] = title
        current_headers["h3"] = ""

      elif level == 3:
        current_headers["h3"] = title


      current_chunk.append(line)
      current_size += len(line)


    else:

      if (
          current_size > chunk_size
          and not in_code_block
          and line.strip() == ""
      ):

        chunks.append({
          "content": "\n".join(current_chunk),
          "metadata": current_headers.copy()
        })

        overlap_text = (
          current_chunk[-5:]
          if len(current_chunk) > 5
          else []
        )

        current_chunk = list(overlap_text)

        current_size = sum(
          len(l) for l in current_chunk
        )

      else:
        current_chunk.append(line)
        current_size += len(line)


  if current_chunk:
    chunks.append({
      "content": "\n".join(current_chunk),
      "metadata": current_headers.copy()
    })

  return chunks


"""
### Embedding Model
"""

embedding_model = SentenceTransformer(
    "all-mpnet-base-v2"
)


"""
### Embed Chunks
"""

def embed_chunks(chunks):

  texts = []
  metadatas = []

  for c in chunks:

    text = f"""
    {c['metadata']['h1']}
    {c['metadata']['h2']}
    {c['metadata']['h3']}

    {c['content']}
    """

    texts.append(text.strip())
    metadatas.append(c['metadata'])


  vectors = embedding_model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True
  )

  return texts, vectors, metadatas


"""
### FAISS Index
"""

def build_faiss_index(vectors):

  dim = vectors.shape[1]

  index = faiss.IndexFlatIP(dim)

  index.add(vectors.astype("float32"))

  return index


"""
### RAG Setup Wrapper

Carga txt → chunk → embed → index
"""

def rag_txt(txt_path):

  text_content = load_txt(txt_path)

  chunks = markdown_chunking(text_content)

  texts, vectors, metadatas = embed_chunks(chunks)

  index = build_faiss_index(vectors)

  return texts, index, metadatas


"""
### Retrieval
"""

def retrieve_txt(
    query,
    index,
    texts,
    metadatas,
    k=1
):

  query_vector = embedding_model.encode(
    [query],
    normalize_embeddings=True
  )

  scores, indices = index.search(
    query_vector.astype("float32"),
    k
  )

  context = []

  for idx in indices[0]:

    context.append({
      "content": texts[idx],
      "metadata": metadatas[idx]
    })

  return context
