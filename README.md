# TradeLens AI

TradeLens AI is a local AI-powered trade analytics dashboard for analyzing CSV trading and backtest results.

It lets users upload a private CSV file, calculate performance metrics, visualize trading results, and ask natural-language questions about the trades.

## Features

- Upload trade CSV files directly through the Streamlit UI
- Calculate key trading metrics:
  - Total trades
  - Win rate
  - Gross R
  - Net R
  - Average R
  - Average win
  - Average loss
  - Profit factor
  - Max drawdown
- View performance breakdowns by:
  - Direction
  - Setup
  - Result
  - Hour
  - Month
  - Score
  - Bias
- Display equity curve charts
- Show best and worst trades
- Ask questions about the trade data using local AI and RAG
- Runs locally with Ollama, LangChain, FAISS, pandas, Plotly, and Streamlit

## Tech Stack

- Python
- Streamlit
- pandas
- Plotly
- Ollama
- LangChain
- FAISS
- Local LLM: `llama3.2:3b`
- Embedding model: `nomic-embed-text`

## Project Structure

```text
tradelens-ai/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── analytics.py
│   ├── rag_builder.py
│   └── rag_query.py
│
├── data/
│   └── trades.csv
│
├── vectorstore/
│   └── FAISS index files
│
└── reports/
