import os

from src.llm import get_llm
from src.retrievers.csv_retriever import load_csv
from src.retrievers.txt_retriever import rag_txt
from src.workflow import deterministic_workflow


def build_pipeline_assets(data_dir: str = "data"):
    csv_path = os.path.join(data_dir, "datos_clima_mexico.csv")
    txt_path = os.path.join(data_dir, "GPT-41_PromptingGuide.txt")
    image_path = os.path.join(data_dir, "maiz_info.jpg")

    data = load_csv(csv_path)
    texts, index, metadatas = rag_txt(txt_path)

    return data, texts, index, metadatas, image_path


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No se encontro GOOGLE_API_KEY. Define la variable de entorno antes de ejecutar."
        )

    chat_model = get_llm(api_key)
    data, texts, index, metadatas, image_path = build_pipeline_assets()

    while True:
        user_question = input("\nTu pregunta (o 'salir'): ").strip()
        if not user_question:
            continue

        if user_question.lower() in {"salir", "exit", "quit"}:
            break

        result = deterministic_workflow(
            user_question=user_question,
            chat_model=chat_model,
            data=data,
            index=index,
            texts=texts,
            metadatas=metadatas,
            image_path=image_path,
        )

        print("\n=== Original ===")
        print(result.original_answer)
        print("\n=== Dry ===")
        print(result.dry_answer)
        print("\n=== Funny ===")
        print(result.funny_answer)


if __name__ == "__main__":
    main()
