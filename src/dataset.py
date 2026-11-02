"""Dataset ingestion. Fraud labels and personal columns are intentionally never loaded."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

INPUT_COLS = ["trans_date_trans_time", "cc_num", "merchant", "category", "amt", "lat", "long", "merch_lat", "merch_long", "trans_num"]


def _anonymize_account(card: str) -> str:
    return hashlib.sha256(str(card).encode("utf-8")).hexdigest()[:16]


def clean_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a minimal, pseudonymous, fraud-label-free canonical transaction frame."""
    absent = [c for c in INPUT_COLS if c not in frame.columns]
    if absent:
        raise ValueError(f"Missing Kaggle columns: {absent}")
    df = frame[INPUT_COLS].copy()  # Explicit allowlist blocks is_fraud leakage.
    df = df.rename(columns={
        "trans_date_trans_time": "time", "cc_num": "account", "trans_num": "tx_id",
        "amt": "amount", "lat": "home_lat", "long": "home_lon",
        "merch_lat": "merchant_lat", "merch_long": "merchant_lon",
    })
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in ("amount", "home_lat", "home_lon", "merchant_lat", "merchant_lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["account", "time", "tx_id", "amount"]).copy()
    df = df.loc[df["amount"] >= 0].copy()
    df["account"] = df["account"].astype(str).map(_anonymize_account)
    for c in ("category", "merchant"):
        df[c] = df[c].fillna("unknown").astype(str)
    # 1-degree grid of merchant *transaction* locations, NOT cardholder city.
    latgrid = df["merchant_lat"].round(0).astype("Int64").astype(str)
    longrid = df["merchant_lon"].round(0).astype("Int64").astype(str)
    df["merchant_region"] = np.where(df["merchant_lat"].notna() & df["merchant_lon"].notna(), latgrid + ":" + longrid, "unknown")
    df = df.drop_duplicates(subset="tx_id", keep="first")
    return df.sort_values(["account", "time", "tx_id"]).reset_index(drop=True)


def load_kaggle(path: str, accounts: int = 25, per_account: int = 100, seed: int = 42) -> pd.DataFrame:
    """Two streaming passes: select accounts, then preserve their chronological histories.

    Account selection never uses labels or amounts. A bounded pilot sample is used for CPU runs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset missing: {path}. Download Kaggle fraudTrain.csv first.")
    if accounts < 0 or (0 < accounts < 4) or (per_account != 0 and per_account < 8):
        raise ValueError("Choose --accounts >=4 and --per-account >=8, or 0 for full history.")
    # Read only account IDs in pass one to sample accounts from the complete dataset.
    account_ids = set()
    for chunk in pd.read_csv(path, usecols=["cc_num"], dtype={"cc_num":"string"}, chunksize=100_000):
        account_ids.update(chunk["cc_num"].dropna().unique().tolist())
    rng = np.random.default_rng(seed)
    chosen = rng.choice(sorted(account_ids), size=min(accounts, len(account_ids)) if accounts else len(account_ids), replace=False)
    selected = set(chosen.tolist())
    frames = []
    for chunk in pd.read_csv(path, usecols=INPUT_COLS, dtype={"cc_num":"string"}, chunksize=100_000):
        part = chunk[chunk["cc_num"].isin(selected)]
        if not part.empty:
            frames.append(part)
    if not frames:
        raise ValueError("No transactions for sampled accounts")
    raw = pd.concat(frames, ignore_index=True)
    raw["trans_date_trans_time"] = pd.to_datetime(raw["trans_date_trans_time"], errors="coerce")
    raw = raw.sort_values(["cc_num", "trans_date_trans_time", "trans_num"])
    # Per-account limit keeps the *earliest contiguous portion* of each history.
    # For a larger experiment use a high cap or full histories with --per-account 0.
    if per_account:
        raw = raw.groupby("cc_num", sort=False, group_keys=False).head(per_account)
    return clean_transactions(raw)


def demo_transactions(accounts: int = 30, per_account: int = 30, seed: int = 42) -> pd.DataFrame:
    """Self-contained synthetic dataset for smoke testing; NOT Kaggle measurements."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(accounts):
        now = pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(rng.integers(0, 100)))
        home_lat = float(rng.uniform(30, 46))
        home_lon = float(rng.uniform(-113, -73))
        for j in range(per_account):
            minutes = float(rng.exponential(80)) + 3
            if j in (per_account // 2, per_account // 2 + 1) and i % 5 == 0:
                minutes = 0.2
            now += pd.Timedelta(minutes=minutes)
            lat = home_lat + rng.normal(0, .7)
            lon = home_lon + rng.normal(0, .7)
            amount = float(rng.lognormal(3.5, .8))
            if j == per_account // 2 and i % 5 == 0:
                amount *= 19
                lat, lon = lat + 13, lon + 17
            rows.append({"trans_date_trans_time":now, "cc_num":str(4000000000000000 + i),
                         "merchant":f"shop_{int(rng.integers(0, 30))}",
                         "category":rng.choice(["grocery", "gas", "travel", "shopping"]),
                         "amt":amount, "lat":home_lat,"long":home_lon,
                         "merch_lat":lat,"merch_long":lon,"trans_num":f"demo-{i}-{j}",
                         "is_fraud":int(i % 5 == 0 and j == per_account//2)})
    return clean_transactions(pd.DataFrame(rows))
