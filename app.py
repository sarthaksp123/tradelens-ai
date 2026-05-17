import streamlit as st
import plotly.express as px
import pandas as pd

from src.config import CSV_PATH
from src.data_loader import load_trades
from src.analytics import (
    performance_summary,
    group_performance,
    worst_trades,
    best_trades,
)
from src.rag_query import ask_rag
from src.rag_builder import build_vectorstore


st.set_page_config(
    page_title="TradeLens AI",
    page_icon="📈",
    layout="wide",
)

st.title("📈 TradeLens AI")
st.caption("Local AI-powered CSV trade analysis using Python, Streamlit, Ollama, LangChain, and FAISS")


def save_uploaded_file(uploaded_file):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CSV_PATH, "wb") as f:
        f.write(uploaded_file.getbuffer())


def load_uploaded_preview(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        if df.shape[1] == 1:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep="\t")
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep="\t")

    uploaded_file.seek(0)
    return df


def is_direct_stats_question(question: str) -> bool:
    q = question.lower().strip()

    direct_keywords = [
        "max dd",
        "maximum dd",
        "max drawdown",
        "maximum drawdown",
        "drawdown",
        "win rate",
        "profit factor",
        "total trades",
        "how many trades",
        "number of trades",
        "gross r",
        "net r",
        "average r",
        "avg r",
        "average win",
        "avg win",
        "average loss",
        "avg loss",
        "rows",
        "row count",
        "how many rows",
        "rows of data",
        "csv rows",
        "data rows",
    ]

    return any(keyword in q for keyword in direct_keywords)


def answer_direct_stats_question(question: str, df: pd.DataFrame) -> str:
    q = question.lower().strip()
    summary = performance_summary(df)

    if (
        "max dd" in q
        or "maximum dd" in q
        or "max drawdown" in q
        or "maximum drawdown" in q
        or "drawdown" in q
    ):
        return (
            "### Max Drawdown\n\n"
            f"The max drawdown is **{summary['max_drawdown_r']} R**.\n\n"
            "This was calculated directly from your trade data, not from the RAG index."
        )

    if "win rate" in q:
        return (
            "### Win Rate\n\n"
            f"The win rate is **{summary['win_rate_pct']}%**.\n\n"
            f"Wins: **{summary['wins']}**\n\n"
            f"Losses: **{summary['losses']}**\n\n"
            f"Breakeven: **{summary.get('breakeven', 0)}**\n\n"
            f"Total trades: **{summary['total_trades']}**"
        )

    if "profit factor" in q:
        return (
            "### Profit Factor\n\n"
            f"The profit factor is **{summary['profit_factor']}**."
        )

    if (
        "rows" in q
        or "row count" in q
        or "how many rows" in q
        or "rows of data" in q
        or "csv rows" in q
        or "data rows" in q
    ):
        return (
            "### CSV Row Count\n\n"
            f"The uploaded CSV has **{len(df)} rows of data**.\n\n"
            f"In trading terms, this currently equals **{summary['total_trades']} trades**."
        )

    if "total trades" in q or "how many trades" in q or "number of trades" in q:
        return (
            "### Total Trades\n\n"
            f"Total trades: **{summary['total_trades']}**"
        )

    if "gross r" in q:
        return (
            "### Gross R\n\n"
            f"Gross R: **{summary['gross_r']} R**"
        )

    if "net r" in q:
        return (
            "### Net R\n\n"
            f"Net R: **{summary['net_r']} R**"
        )

    if "average win" in q or "avg win" in q:
        return (
            "### Average Win\n\n"
            f"Average winning trade: **{summary['avg_win_r']} R**"
        )

    if "average loss" in q or "avg loss" in q:
        return (
            "### Average Loss\n\n"
            f"Average losing trade: **{summary['avg_loss_r']} R**"
        )

    if "average r" in q or "avg r" in q:
        return (
            "### Average R\n\n"
            f"Average R per trade: **{summary['avg_r']} R**"
        )

    return (
        "### Performance Summary\n\n"
        f"Total trades: **{summary['total_trades']}**\n\n"
        f"Rows: **{len(df)}**\n\n"
        f"Wins: **{summary['wins']}**\n\n"
        f"Losses: **{summary['losses']}**\n\n"
        f"Breakeven: **{summary.get('breakeven', 0)}**\n\n"
        f"Win rate: **{summary['win_rate_pct']}%**\n\n"
        f"Gross R: **{summary['gross_r']} R**\n\n"
        f"Net R: **{summary['net_r']} R**\n\n"
        f"Average R: **{summary['avg_r']} R**\n\n"
        f"Average Win: **{summary['avg_win_r']} R**\n\n"
        f"Average Loss: **{summary['avg_loss_r']} R**\n\n"
        f"Profit Factor: **{summary['profit_factor']}**\n\n"
        f"Max Drawdown: **{summary['max_drawdown_r']} R**"
    )


def does_question_need_ai(question: str) -> bool:
    q = question.lower().strip()

    analysis_keywords = [
        "why",
        "explain",
        "analyze",
        "analysis",
        "pattern",
        "patterns",
        "weakness",
        "weaknesses",
        "strength",
        "strengths",
        "improve",
        "filter",
        "filters",
        "recommend",
        "suggest",
        "compare",
        "worst",
        "best",
        "losing",
        "winning",
        "losses",
        "winners",
        "losers",
        "setup",
        "setups",
        "direction",
        "short",
        "long",
        "hour",
        "month",
        "score",
        "bias",
        "liquidity",
        "target",
        "stop loss",
        "sl",
        "take profit",
        "tp",
        "risk reward",
        "rr",
    ]

    return any(keyword in q for keyword in analysis_keywords)


def unclear_question_response(question: str) -> str:
    return (
        "### I do not understand that question yet.\n\n"
        "Try asking a direct stats question like:\n\n"
        "- What is the max DD?\n"
        "- How many rows are in the CSV?\n"
        "- What is the win rate?\n"
        "- What is the profit factor?\n\n"
        "Or ask a deeper analysis question like:\n\n"
        "- Why are the worst trades losing?\n"
        "- Which setup performs best?\n"
        "- What filters should I test next?\n\n"
        "For deeper analysis questions, click **Rebuild RAG Index** first."
    )


with st.sidebar:
    st.header("Upload")

    uploaded_file = st.file_uploader(
        "Upload your trades CSV",
        type=["csv", "txt", "tsv"],
    )

    if uploaded_file is not None:
        preview_df = load_uploaded_preview(uploaded_file)

        st.success("File uploaded successfully.")
        st.caption(f"Rows detected: {len(preview_df)}")
        st.caption(f"Columns detected: {len(preview_df.columns)}")

        if st.button("Use This CSV"):
            save_uploaded_file(uploaded_file)
            st.cache_data.clear()
            st.success("CSV saved to project data folder.")
            st.rerun()

    st.divider()

    rebuild_index = st.button("Rebuild RAG Index")

    st.divider()

    rag_k = st.slider("RAG retrieved trades", min_value=3, max_value=20, value=8)

    st.caption("Model: llama3.2:3b")
    st.caption("Embeddings: nomic-embed-text")


@st.cache_data
def load_data_from_saved_csv():
    return load_trades()


if not CSV_PATH.exists():
    st.warning("Upload a trades CSV from the sidebar to begin.")
    st.stop()


df = load_data_from_saved_csv()


if rebuild_index:
    with st.spinner("Building FAISS index from uploaded CSV rows..."):
        build_vectorstore()
    st.success("RAG index rebuilt successfully.")


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Overview",
        "Breakdowns",
        "Equity Curve",
        "Best/Worst Trades",
        "Ask AI",
    ]
)


with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(50), width="stretch")

    st.subheader("Performance Summary")

    try:
        summary = performance_summary(df)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Trades", summary["total_trades"])
        col2.metric("Win Rate", f"{summary['win_rate_pct']}%")
        col3.metric("Gross R", summary["gross_r"])
        col4.metric("Net R", summary["net_r"])

        col5, col6, col7, col8 = st.columns(4)

        col5.metric("Avg R", summary["avg_r"])
        col6.metric("Avg Win R", summary["avg_win_r"])
        col7.metric("Avg Loss R", summary["avg_loss_r"])
        col8.metric("Profit Factor", summary["profit_factor"])

        col9, col10, col11, col12 = st.columns(4)

        col9.metric("Wins", summary["wins"])
        col10.metric("Losses", summary["losses"])
        col11.metric("Breakeven", summary.get("breakeven", 0))
        col12.metric("Max DD", f"{summary['max_drawdown_r']} R")

        st.json(summary)

    except Exception as e:
        st.error(f"Could not calculate performance summary: {e}")


with tab2:
    st.subheader("Performance Breakdowns")

    possible_cols = [
        "direction",
        "setup",
        "result",
        "hour",
        "month",
        "year",
        "score",
        "daily_bias",
        "cisd_ok",
        "directional_ok",
        "be_moved",
        "target_type",
    ]

    existing_cols = [c for c in possible_cols if c in df.columns]

    if not existing_cols:
        st.warning("No supported grouping columns found.")
    else:
        group_col = st.selectbox("Group by", existing_cols)

        try:
            grouped = group_performance(df, group_col)

            st.dataframe(grouped, width="stretch")

            if not grouped.empty:
                y_col = "net_r" if "net_r" in grouped.columns else "total_r"

                fig = px.bar(
                    grouped,
                    x=group_col,
                    y=y_col,
                    title=f"{y_col} by {group_col}",
                )
                st.plotly_chart(fig, width="stretch")

        except Exception as e:
            st.error(f"Could not calculate breakdown: {e}")


with tab3:
    st.subheader("Equity Curve")

    equity_col = None

    if "equity_r_after_cost" in df.columns:
        equity_col = "equity_r_after_cost"
    elif "equity_r_after_fee" in df.columns:
        equity_col = "equity_r_after_fee"
    elif "equity_r" in df.columns:
        equity_col = "equity_r"

    if equity_col:
        fig = px.line(
            df.reset_index(),
            x="index",
            y=equity_col,
            title=f"Equity Curve: {equity_col}",
        )
        st.plotly_chart(fig, width="stretch")

    else:
        st.warning("No equity column found.")


with tab4:
    st.subheader("Worst Trades")

    try:
        st.dataframe(worst_trades(df, 20), width="stretch")
    except Exception as e:
        st.error(f"Could not show worst trades: {e}")

    st.subheader("Best Trades")

    try:
        st.dataframe(best_trades(df, 20), width="stretch")
    except Exception as e:
        st.error(f"Could not show best trades: {e}")


with tab5:
    st.subheader("Ask AI About Your Trades")

    st.info(
        "Simple stats like max drawdown, row count, win rate, profit factor, and total trades "
        "are answered directly from pandas. Deeper pattern questions use the RAG index."
    )

    st.caption(
        "For deeper AI questions, upload the CSV, click 'Use This CSV', then click 'Rebuild RAG Index'."
    )

    question = st.text_area(
        "Question",
        value="What is the max dd?",
        height=100,
    )

    if st.button("Ask AI"):
        if not question.strip():
            st.markdown(unclear_question_response(question))

        elif is_direct_stats_question(question):
            answer = answer_direct_stats_question(question, df)
            st.markdown(answer)

        elif does_question_need_ai(question):
            try:
                with st.spinner("Thinking through retrieved trade examples..."):
                    answer = ask_rag(question, k=rag_k)

                st.markdown(answer)

            except FileNotFoundError:
                st.warning(
                    "FAISS index not found. Click **Rebuild RAG Index** in the sidebar first, then ask the AI again."
                )

            except Exception as e:
                st.error(f"AI question failed: {e}")

        else:
            st.markdown(unclear_question_response(question))
