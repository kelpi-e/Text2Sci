import sys
import os

# Добавляем корень проекта в sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embedding.embedder import TextEmbedder
from retrieval.retriever import VectorRetriever
from llm.client import AIClient

def load_text(path: str) -> str:
    """Читает весь текст из файла."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def split_text_into_chunks(text: str, chunk_size: int = 3) -> list[str]:
    """
    Разбивает текст на абзацы и объединяет их по chunk_size для индексации.
    Это позволяет получать более информативные фрагменты.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for i in range(0, len(paragraphs), chunk_size):
        chunk = " ".join(paragraphs[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def main():
    # Путь к файлу
    base_path = os.path.join("src", "unit_tests", "files_for_testing", "1984.txt")
    text = load_text(base_path)

    # Разбиваем текст на крупные фрагменты
    chunks = split_text_into_chunks(text, chunk_size=5)  # объединяем по 5 абзацев

    # Инициализируем эмбеддер
    embeder = TextEmbedder()
    embeddings = embeder.encode(chunks)

    # Создаём FAISS индекс
    dim = embeddings.shape[1]
    retriever = VectorRetriever(dim=dim)
    retriever.add_embeddings(embeddings, chunks)

    query = "Uinston Smith visiting O'Brien"
    query_vec = embeder.encode([query])

    # Поиск топ-5 фрагментов
    results = retriever.search(query_vec, top_k=5)

    print("Топ-5 фрагментов по запросу:")
    for i, (chunk_text, dist) in enumerate(results, 1):
        # Выводим первые 1000 символов для наглядности
        snippet = chunk_text[:1000]
        print(f"\n--- Фрагмент {i} ---\n{snippet}\nРасстояние: {dist:.4f}")

if __name__ == "__main__":
    #main()
    client = AIClient()
    print(client.generate("привет"))
