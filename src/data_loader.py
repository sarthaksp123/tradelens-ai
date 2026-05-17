import pandas as pd
from src.config import CSV_PATH


def load_trades(csv_path=CSV_PATH) -> pd.DataFrame:
    """
    Load trade CSV or TSV file and clean basic columns.

    Handles:
    - comma-separated CSV
    - tab-separated CSV/TSV
    - timezone-aware datetime columns with mixed offsets
    """

    try:
        df = pd.read_csv(csv_path)
        if df.shape[1] == 1:
            df = pd.read_csv(csv_path, sep="\t")
    except Exception:
        df = pd.read_csv(csv_path, sep="\t")

    df.columns = [c.strip() for c in df.columns]

    datetime_cols = [
        "entry_time",
        "candidate_created_time",
        "armed_time",
        "be_time",
        "exit_time",
    ]

    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    numeric_cols = [
        "entry",
        "initial_stop",
        "stop",
        "target",
        "initial_risk",
        "projected_rr",
        "liquidity_rr",
        "score",
        "exit_price",
        "pnl_points",
        "r",
        "dollars_per_contract",
        "equity_r",
        "peak_r",
        "drawdown_r",
        "fee_r",
        "r_after_fee",
        "r_after_cost",
        "equity_r_after_fee",
        "equity_r_after_cost",
        "peak_r_after_fee",
        "peak_r_after_cost",
        "drawdown_r_after_fee",
        "drawdown_r_after_cost",
        "hour",
        "year",
        "bars_after_setup",
        "bars_after_armed",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    bool_cols = [
        "liquidity_target_used",
        "cisd_ok",
        "directional_ok",
        "be_moved",
    ]

    for col in bool_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.upper()
                .map({"TRUE": True, "FALSE": False})
            )

    return df


if __name__ == "__main__":
    df = load_trades()
    print(df.head())
    print(df.info())
