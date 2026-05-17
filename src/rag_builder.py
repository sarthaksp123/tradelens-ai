from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import VECTORSTORE_DIR, OLLAMA_EMBED_MODEL
from src.data_loader import load_trades


def trade_row_to_text(row, idx: int) -> str:
    parts = [
        f"Trade number: {idx}",
        f"Entry time: {row.get('entry_time')}",
        f"Exit time: {row.get('exit_time')}",
        f"Direction: {row.get('direction')}",
        f"Side: {row.get('side')}",
        f"Setup: {row.get('setup')}",
        f"Source timeframe: {row.get('source_tf')}",
        f"Entry price: {row.get('entry')}",
        f"Initial stop: {row.get('initial_stop')}",
        f"Target: {row.get('target')}",
        f"Initial risk: {row.get('initial_risk')}",
        f"Projected RR: {row.get('projected_rr')}",
        f"Target type: {row.get('target_type')}",
        f"Liquidity target used: {row.get('liquidity_target_used')}",
        f"Score: {row.get('score')}",
        f"Daily bias: {row.get('daily_bias')}",
        f"CISD OK: {row.get('cisd_ok')}",
        f"Directional OK: {row.get('directional_ok')}",
        f"Break even moved: {row.get('be_moved')}",
        f"Result: {row.get('result')}",
        f"PNL points: {row.get('pnl_points')}",
        f"R result: {row.get('r')}",
        f"Fee R: {row.get('fee_r')}",
        f"R after fee: {row.get('r_after_fee')}",
        f"R after cost: {row.get('r_after_cost')}",
        f"Equity R after fee: {row.get('equity_r_after_fee')}",
        f"Equity R after cost: {row.get('equity_r_after_cost')}",
        f"Drawdown R after fee: {row.get('drawdown_r_after_fee')}",
        f"Drawdown R after cost: {row.get('drawdown_r_after_cost')}",
        f"Hour: {row.get('hour')}",
        f"Month: {row.get('month')}",
        f"Year: {row.get('year')}",
    ]

    return "\n".join(parts)


def safe_float(value):
    try:
        if value != value:
            return None
        return float(value)
    except Exception:
        return None


def build_vectorstore():
    df = load_trades()

    documents = []

    for idx, row in df.iterrows():
        text = trade_row_to_text(row, idx)

        metadata = {
            "row_id": int(idx),
            "direction": str(row.get("direction")),
            "setup": str(row.get("setup")),
            "result": str(row.get("result")),
            "hour": str(row.get("hour")),
            "month": str(row.get("month")),
            "year": str(row.get("year")),
            "r": safe_float(row.get("r")),
            "r_after_fee": safe_float(row.get("r_after_fee")),
            "r_after_cost": safe_float(row.get("r_after_cost")),
        }

        documents.append(Document(page_content=text, metadata=metadata))

    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

    vectorstore = FAISS.from_documents(documents, embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    print(f"Saved FAISS index to: {VECTORSTORE_DIR}")
    print(f"Indexed {len(documents)} trade rows.")
