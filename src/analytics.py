import pandas as pd


def get_net_r_column(df: pd.DataFrame) -> str:
    """
    Detect the best net R column available.
    """
    if "r_after_cost" in df.columns:
        return "r_after_cost"
    if "r_after_fee" in df.columns:
        return "r_after_fee"
    if "r" in df.columns:
        return "r"
    raise ValueError("No R column found. Expected r, r_after_fee, or r_after_cost.")


def get_equity_column(df: pd.DataFrame) -> str | None:
    """
    Detect the best equity curve column.
    """
    if "equity_r_after_cost" in df.columns:
        return "equity_r_after_cost"
    if "equity_r_after_fee" in df.columns:
        return "equity_r_after_fee"
    if "equity_r" in df.columns:
        return "equity_r"
    return None


def calculate_max_drawdown_from_equity(equity: pd.Series) -> float:
    """
    Calculate max drawdown from an equity curve.
    """
    equity = pd.to_numeric(equity, errors="coerce").dropna()

    if equity.empty:
        return 0.0

    running_peak = equity.cummax()
    drawdown = equity - running_peak

    return float(drawdown.min())


def get_max_drawdown(df: pd.DataFrame) -> float:
    """
    Prefer existing drawdown columns if available.
    Otherwise calculate max drawdown from equity curve.
    """

    drawdown_candidates = [
        "drawdown_r_after_cost",
        "drawdown_r_after_fee",
        "drawdown_r",
    ]

    for col in drawdown_candidates:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if not series.empty:
                return float(series.min())

    equity_col = get_equity_column(df)

    if equity_col is not None:
        return calculate_max_drawdown_from_equity(df[equity_col])

    net_col = get_net_r_column(df)
    equity = pd.to_numeric(df[net_col], errors="coerce").fillna(0).cumsum()

    return calculate_max_drawdown_from_equity(equity)


def performance_summary(df: pd.DataFrame) -> dict:
    total_trades = len(df)

    if total_trades == 0:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "win_rate_pct": 0,
            "gross_r": 0,
            "net_r": 0,
            "avg_r": 0,
            "avg_win_r": 0,
            "avg_loss_r": 0,
            "profit_factor": None,
            "max_drawdown_r": 0,
        }

    if "r" not in df.columns:
        raise ValueError("CSV must contain an 'r' column for performance analysis.")

    net_col = get_net_r_column(df)

    r = pd.to_numeric(df["r"], errors="coerce").fillna(0)
    net_r = pd.to_numeric(df[net_col], errors="coerce").fillna(0)

    wins = df[r > 0]
    losses = df[r < 0]
    breakeven = df[r == 0]

    gross_r = r.sum()
    net_r_total = net_r.sum()

    win_rate = len(wins) / total_trades * 100

    avg_r = r.mean()
    avg_win = r[r > 0].mean() if len(wins) else 0
    avg_loss = r[r < 0].mean() if len(losses) else 0

    gross_win = r[r > 0].sum()
    gross_loss = abs(r[r < 0].sum())
    profit_factor = gross_win / gross_loss if gross_loss else None

    max_drawdown = get_max_drawdown(df)

    return {
        "total_trades": total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(breakeven),
        "win_rate_pct": round(win_rate, 2),
        "gross_r": round(float(gross_r), 4),
        "net_r": round(float(net_r_total), 4),
        "avg_r": round(float(avg_r), 4),
        "avg_win_r": round(float(avg_win), 4),
        "avg_loss_r": round(float(avg_loss), 4),
        "profit_factor": round(float(profit_factor), 4) if profit_factor else None,
        "max_drawdown_r": round(float(max_drawdown), 4),
    }


def group_performance(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()

    if "r" not in df.columns:
        return pd.DataFrame()

    net_col = get_net_r_column(df)

    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            trades=("r", "count"),
            total_r=("r", "sum"),
            avg_r=("r", "mean"),
            win_rate=("r", lambda x: (x > 0).mean() * 100),
            net_r=(net_col, "sum"),
        )
        .reset_index()
    )

    grouped["total_r"] = grouped["total_r"].round(3)
    grouped["avg_r"] = grouped["avg_r"].round(3)
    grouped["win_rate"] = grouped["win_rate"].round(2)
    grouped["net_r"] = grouped["net_r"].round(3)

    return grouped.sort_values("net_r", ascending=False)


def worst_trades(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    sort_col = get_net_r_column(df)
    return df.sort_values(sort_col).head(n)


def best_trades(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    sort_col = get_net_r_column(df)
    return df.sort_values(sort_col, ascending=False).head(n)
