from pathlib import Path

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

from src.config import VECTORSTORE_DIR, OLLAMA_EMBED_MODEL, OLLAMA_CHAT_MODEL


SYSTEM_PROMPT = """
You are a trading performance analyst.

You analyze CSV-based backtest and trade execution data.

Rules:
- Use only the provided retrieved trade context.
- Be specific and numeric when possible.
- Mention patterns by setup, direction, hour, result, R, fees, cost, and drawdown.
- Do not invent trades or statistics.
- If the retrieved context is insufficient, say what additional analysis is needed.
- Keep answers practical and focused on improving the trading algorithm.
"""


def load_vectorstore():
    index_file = Path(VECTORSTORE_DIR) / "index.faiss"

    if not index_file.exists():
        raise FileNotFoundError(
            "FAISS index not found. In the Streamlit sidebar, click 'Rebuild RAG Index' first."
        )

    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def ask_rag(question: str, k: int = 8) -> str:
    vectorstore = load_vectorstore()

    docs = vectorstore.similarity_search(question, k=k)

    context = "\n\n--- TRADE CONTEXT ---\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                """
Question:
{question}

Retrieved trade context:
{context}

Answer:
""",
            ),
        ]
    )

    llm = ChatOllama(
        model=OLLAMA_CHAT_MODEL,
        temperature=0.1,
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    return response.content
