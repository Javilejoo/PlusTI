"""
EDA 3 Bancos — Deteccion de Fraude en Entretenimiento (Rest/Cine/Bar)
Banco 1: Bolivia BO-VIP  | Banco 2: Brazil BR-PRIVADO | Banco 3: Guatemala GT-ESTATAL
Figuras guardadas en figures/eda_3bancos/
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import pointbiserialr

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

DATA = "Documentos/Copia de Datasets 3 bancos"

BANKS = {
    "BANCO 1 - BOLIVIA (BO-VIP)": {
        "file":       f"{DATA}/Copia de 01_bo_vip_seed22_n100000.csv",
        "has_labels": True,
        "color":      "steelblue",
        "short":      "BO",
    },
    "BANCO 2 - BRAZIL (BR-PRIVADO)": {
        "file":       f"{DATA}/Copia de 02_br_privado_seed33_n100000.csv",
        "has_labels": True,
        "color":      "crimson",
        "short":      "BR",
    },
    "BANCO 3 - GUATEMALA (GT-ESTATAL)": {
        "file":       f"{DATA}/Copia de 03_gt_estatal_seed3_n100000.csv",
        "has_labels": False,
        "color":      "forestgreen",
        "short":      "GT",
    },
}

MCC_TARGET  = {5812: "Restaurante", 5814: "FastFood", 7832: "Cinema"}
MCC_BROADER = [5812, 5813, 5814, 7832, 7922, 7941, 7996]
FIG_DIR     = "figures/eda_3bancos"
os.makedirs(FIG_DIR, exist_ok=True)

NEW_FEATURES = [
    "is_dinner_time", "is_bar_time", "is_broader_entertainment",
    "txn_same_evening", "mcc_diversity_last_3", "is_split_amount",
    "amount_entertainment_daily", "time_to_next_txn_min",
    "is_no_pin_entertainment", "consecutive_ent_flag",
]

EXISTING_FEATURES = [
    "amount_usd", "distance_from_home_km", "is_international",
    "seconds_since_prev", "client_amount_zscore", "amount_vs_baseline",
]


# ---------------------------------------------------------------------------
# UTILITY
# ---------------------------------------------------------------------------

def savefig(name: str, bank_short: str = "ALL") -> None:
    plt.tight_layout()
    path = f"{FIG_DIR}/{name}_{bank_short}.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  [fig] {path}")


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def bank_header(bank_name: str) -> None:
    print(f"\n  --- {bank_name} ---")


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def build_txn_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    DE7_transmission_datetime is an int treated as nanoseconds from epoch
    by pd.to_datetime, producing 1970-01-01 timestamps useless for velocity.
    Reconstruct from DE13_local_date (MMDD) + DE12_local_time (HHMMSS).
    """
    de13 = df["DE13_local_date"].astype(str).str.zfill(4)
    de12 = df["DE12_local_time"].astype(str).str.zfill(6)
    df["txn_datetime"] = pd.to_datetime(
        "2024-" + de13.str[0:2] + "-" + de13.str[2:4] + " " +
        de12.str[0:2] + ":" + de12.str[2:4] + ":" + de12.str[4:6],
        errors="coerce",
    )
    df["txn_date"] = df["txn_datetime"].dt.date
    return df


def load_banks() -> dict:
    dfs = {}
    for bank_name, cfg in BANKS.items():
        print(f"\n[load] {bank_name}")
        df = pd.read_csv(cfg["file"], sep=";", low_memory=False)
        df["bank_label"] = bank_name
        df["bank_short"] = cfg["short"]
        df = build_txn_datetime(df)

        if cfg["has_labels"]:
            df["is_fraud_int"] = df["is_fraud"].astype(float)
            rate = df["is_fraud_int"].mean()
            n    = int(df["is_fraud_int"].sum())
            print(f"  Filas: {len(df):,} | Fraude: {rate:.4f} ({n} casos)")
        else:
            df["is_fraud_int"] = np.nan
            print(f"  Filas: {len(df):,} | Sin labels (solo analisis descriptivo)")

        ent = df["DE18_merchant_category_code"].isin(MCC_TARGET.keys())
        print(f"  Entretenimiento (5812/5814/7832): {ent.sum():,} txns ({ent.mean()*100:.1f}%)")

        # Validate datetime fix
        valid_dates = df["txn_datetime"].dropna()
        print(f"  Fechas: {valid_dates.min().date()} → {valid_dates.max().date()}")

        dfs[bank_name] = df
    return dfs


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_rest_cine_bar"]   = df["DE18_merchant_category_code"].isin(MCC_TARGET.keys()).astype(int)
    df["pin_present"]        = (df["DE52_pin_data_present"] == "Y").astype(int)
    df["emv_present"]        = (df["DE55_emv_data_present"] == "Y").astype(int)
    df["is_international"]   = pd.to_numeric(df["is_international"], errors="coerce").fillna(0).astype(int)
    df["amount_log"]         = np.log1p(df["amount_usd"].fillna(0))
    df["amount_vs_baseline"] = (df["amount_usd"] / df["client_baseline_amount"].replace(0, np.nan)).fillna(1.0).clip(0, 100)
    df["dist_log"]           = np.log1p(df["distance_from_home_km"].fillna(0))
    df["is_night"]           = ((df["hour_local"] >= 22) | (df["hour_local"] <= 5)).astype(int)
    df["is_weekend"]         = df["day_of_week"].isin(["Sat", "Sun"]).astype(int)
    return df


def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["client_id", "txn_datetime"]).reset_index(drop=True)

    # 1. is_dinner_time
    df["is_dinner_time"] = ((df["hour_local"] >= 19) & (df["hour_local"] <= 23)).astype(int)

    # 2. is_bar_time
    df["is_bar_time"] = ((df["hour_local"] >= 22) | (df["hour_local"] <= 4)).astype(int)

    # 3. is_broader_entertainment (note: MCCs 5813, 7922, 7941, 7996 absent from data)
    df["is_broader_entertainment"] = df["DE18_merchant_category_code"].isin(MCC_BROADER).astype(int)
    if df["is_broader_entertainment"].equals(df["is_rest_cine_bar"]):
        print("  [hallazgo] is_broader_entertainment == is_rest_cine_bar: MCCs 5813/7922/7941/7996 ausentes")

    # 4. txn_same_evening (all channels, 18:00-03:59, evening unit crosses midnight)
    df["_evening_date"] = df["txn_datetime"].dt.date
    late_mask = df["hour_local"] <= 3
    df.loc[late_mask, "_evening_date"] = (
        df.loc[late_mask, "txn_datetime"] - pd.Timedelta(days=1)
    ).dt.date
    df["_is_eve"] = ((df["hour_local"] >= 18) | (df["hour_local"] <= 3)).astype(int)
    eve_counts = (
        df[df["_is_eve"] == 1]
        .groupby(["client_id", "_evening_date"])
        .size()
        .reset_index(name="txn_same_evening")
    )
    df = df.merge(eve_counts, on=["client_id", "_evening_date"], how="left")
    df["txn_same_evening"] = df["txn_same_evening"].fillna(0).astype(int)
    df.drop(columns=["_evening_date", "_is_eve"], inplace=True)

    # 5. mcc_diversity_last_3
    print("  [feature] calculando mcc_diversity_last_3...")
    diversity = []
    for _cid, grp in df.groupby("client_id", sort=False):
        mccs = grp["DE18_merchant_category_code"].tolist()
        for i in range(len(mccs)):
            diversity.append(len(set(mccs[max(0, i - 3):i])))
    df["mcc_diversity_last_3"] = diversity

    # 6. is_split_amount
    df["_prev_merchant"] = df.groupby("client_id")["DE42_card_acceptor_id"].shift(1)
    df["_prev_date"]     = df.groupby("client_id")["txn_date"].shift(1)
    df["_time_diff_min"] = df.groupby("client_id")["txn_datetime"].diff().dt.total_seconds() / 60
    df["is_split_amount"] = (
        (df["_time_diff_min"] < 15) &
        (df["_prev_merchant"] == df["DE42_card_acceptor_id"]) &
        (df["_prev_date"] == df["txn_date"])
    ).astype(int)
    df.drop(columns=["_prev_merchant", "_prev_date", "_time_diff_min"], inplace=True)

    # 7. amount_entertainment_daily
    df["_ent_amount"] = df["amount_usd"] * df["is_rest_cine_bar"]
    daily_ent = (
        df.groupby(["client_id", "txn_date"])["_ent_amount"]
        .sum()
        .reset_index(name="amount_entertainment_daily")
    )
    df = df.merge(daily_ent, on=["client_id", "txn_date"], how="left")
    df["amount_entertainment_daily"] = df["amount_entertainment_daily"].fillna(0.0)
    df.drop(columns=["_ent_amount"], inplace=True)

    # 8. time_to_next_txn_min
    df["time_to_next_txn_min"] = (
        df.groupby("client_id")["txn_datetime"]
        .diff(-1).dt.total_seconds().abs() / 60
    )

    # 9. is_no_pin_entertainment
    df["is_no_pin_entertainment"] = ((df["pin_present"] == 0) & (df["is_rest_cine_bar"] == 1)).astype(int)

    # 10. consecutive_ent_flag
    print("  [feature] calculando consecutive_ent_flag...")
    def _ent_streak(grp):
        count, streak = 0, []
        for f in grp["is_rest_cine_bar"]:
            count = count + 1 if f == 1 else 0
            streak.append(count)
        return pd.Series(streak, index=grp.index)

    df["_ent_streak"]          = df.groupby("client_id", group_keys=False).apply(_ent_streak)
    df["consecutive_ent_flag"] = (df["_ent_streak"] >= 3).astype(int)
    df.drop(columns=["_ent_streak"], inplace=True)

    return df


def add_client_velocity_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Adapted from fraud_rest_cine_bar.add_client_velocity(). Uses txn_datetime (DE12+DE13)."""
    df = df.sort_values(["client_id", "txn_datetime"]).reset_index(drop=True)
    g = df.groupby("client_id")

    df["client_tx_count"]           = g.cumcount()
    df["client_amount_mean_so_far"] = g["amount_usd"].expanding().mean().reset_index(level=0, drop=True)
    df["client_amount_std_so_far"]  = g["amount_usd"].expanding().std().reset_index(level=0, drop=True).fillna(0)
    df["client_amount_zscore"]      = (
        (df["amount_usd"] - df["client_amount_mean_so_far"]) /
        (df["client_amount_std_so_far"] + 1.0)
    )
    df["seconds_since_prev"]   = g["txn_datetime"].diff().dt.total_seconds().fillna(1e6).clip(0, 1e6)
    df["client_merchant_changed"] = (g["DE42_card_acceptor_id"].shift(1) != df["DE42_card_acceptor_id"]).astype(int)

    ONE_HOUR = np.timedelta64(3600, "s")
    ONE_DAY  = np.timedelta64(86400, "s")
    counts_1h, counts_24h, amt_1h, same_mcc_1h = [], [], [], []

    total_clients = df["client_id"].nunique()
    for idx, (_cid, grp) in enumerate(df.groupby("client_id")):
        if idx % 500 == 0:
            print(f"    velocity: {idx}/{total_clients} clientes...", end="\r")
        times   = grp["txn_datetime"].values
        amounts = grp["amount_usd"].fillna(0).values
        mccs    = grp["DE18_merchant_category_code"].values
        n = len(grp)
        c1h = np.zeros(n, dtype=np.int32)
        c24h = np.zeros(n, dtype=np.int32)
        a1h  = np.zeros(n, dtype=np.float64)
        m1h  = np.zeros(n, dtype=np.int32)
        for i in range(n):
            t    = times[i]
            past = times[:i]
            mask_1h  = past >= (t - ONE_HOUR)
            mask_24h = past >= (t - ONE_DAY)
            c1h[i]  = mask_1h.sum()
            c24h[i] = mask_24h.sum()
            a1h[i]  = amounts[:i][mask_1h].sum()
            m1h[i]  = (mask_1h & (mccs[:i] == mccs[i])).sum()
        counts_1h.extend(c1h); counts_24h.extend(c24h)
        amt_1h.extend(a1h);    same_mcc_1h.extend(m1h)

    print()
    df["txn_count_last_1h"]    = counts_1h
    df["txn_count_last_24h"]   = counts_24h
    df["amount_sum_last_1h"]   = amt_1h
    df["txn_same_mcc_last_1h"] = same_mcc_1h
    return df


# ---------------------------------------------------------------------------
# SECTION 1: OVERVIEW COMPARATIVO
# ---------------------------------------------------------------------------

def section1_overview(dfs: dict) -> None:
    section("SECCIÓN 1: OVERVIEW COMPARATIVO — 3 BANCOS")

    labeled = {k: v for k, v in dfs.items() if v["is_fraud_int"].notna().any()}
    colors   = [BANKS[k]["color"] for k in dfs]
    shorts   = [BANKS[k]["short"] for k in dfs]

    # 1a — Stats table
    print(f"\n{'Banco':<45} {'Filas':>8} {'Fraude%':>8} {'N_fraud':>8} {'Ent_txns':>10} {'Ent%':>6}")
    print("-" * 90)
    for name, df in dfs.items():
        cfg  = BANKS[name]
        n    = len(df)
        ent  = df["is_rest_cine_bar"].sum()
        fr   = f"{df['is_fraud_int'].mean()*100:.2f}%" if cfg["has_labels"] else "N/A"
        nfr  = f"{int(df['is_fraud_int'].sum())}" if cfg["has_labels"] else "N/A"
        print(f"{name:<45} {n:>8,} {fr:>8} {nfr:>8} {ent:>10,} {ent/n*100:>5.1f}%")

    # 1b — Fraud rate by MCC (top 15 BO+BR, volume for GT)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("S1 — Tasa de Fraude por MCC (Top 15)", fontsize=13, fontweight="bold")

    for ax, (name, df) in zip(axes, dfs.items()):
        cfg = BANKS[name]
        if cfg["has_labels"]:
            rate = (df.groupby("DE18_merchant_category_code")["is_fraud_int"]
                    .agg(["count", "mean"]).reset_index()
                    .sort_values("mean", ascending=False).head(15))
            ax.bar(range(len(rate)), rate["mean"].values, color=cfg["color"], alpha=0.8)
            ax.set_xticks(range(len(rate)))
            ax.set_xticklabels(rate["DE18_merchant_category_code"].astype(str).values,
                               rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Tasa de fraude")
            for mcc in MCC_TARGET.keys():
                if mcc in rate["DE18_merchant_category_code"].values:
                    i = rate.index[rate["DE18_merchant_category_code"] == mcc][0]
                    pos = rate.index.get_loc(i)
                    ax.get_children()[pos].set_edgecolor("black")
                    ax.get_children()[pos].set_linewidth(2)
        else:
            vol = (df.groupby("DE18_merchant_category_code").size()
                   .sort_values(ascending=False).head(15))
            ax.bar(range(len(vol)), vol.values, color=cfg["color"], alpha=0.8)
            ax.set_xticks(range(len(vol)))
            ax.set_xticklabels(vol.index.astype(str).values, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Volumen de transacciones")
        ax.set_title(f"{cfg['short']} — {name.split('(')[1].rstrip(')')}", fontsize=10)
        ax.set_xlabel("MCC")

    savefig("S1_fraud_rate", "ALL")

    # 1c — MCC objetivo distribution por banco
    fig, ax = plt.subplots(figsize=(10, 5))
    mcc_names = list(MCC_TARGET.values())
    x = np.arange(len(mcc_names))
    width = 0.25

    for i, (name, df) in enumerate(dfs.items()):
        cfg  = BANKS[name]
        vals = [df[df["DE18_merchant_category_code"] == mcc].shape[0] / len(df) * 100
                for mcc in MCC_TARGET.keys()]
        bars = ax.bar(x + i * width, vals, width, label=cfg["short"], color=cfg["color"], alpha=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(mcc_names)
    ax.set_ylabel("% del volumen total")
    ax.set_title("S1 — Distribución de MCCs Objetivo por Banco", fontweight="bold")
    ax.legend()
    savefig("S1_mcc_dist", "ALL")

    # 1d — Channel distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("S1 — Distribución por Canal (ECOM/POS/ATM/MOTO)", fontsize=13, fontweight="bold")
    for ax, (name, df) in zip(axes, dfs.items()):
        cfg = BANKS[name]
        ch  = df["channel"].value_counts(normalize=True) * 100
        ax.bar(ch.index, ch.values, color=cfg["color"], alpha=0.8)
        for i, (lbl, v) in enumerate(ch.items()):
            ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{cfg['short']}")
        ax.set_ylabel("% transacciones")
    savefig("S1_channel", "ALL")

    # 1e — Amount distribution (log scale)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, df in dfs.items():
        cfg = BANKS[name]
        vals = np.log1p(df["amount_usd"].dropna())
        ax.hist(vals, bins=60, alpha=0.4, label=cfg["short"], color=cfg["color"], density=True)
    ax.set_xlabel("log1p(amount_usd)")
    ax.set_ylabel("Densidad")
    ax.set_title("S1 — Distribución de Montos (escala log) por Banco", fontweight="bold")
    ax.legend()
    medians = {BANKS[k]["short"]: np.log1p(v["amount_usd"].median()) for k, v in dfs.items()}
    for i, (sh, med) in enumerate(medians.items()):
        ax.axvline(med, color=list(dfs.values())[i].iloc[0]["bank_short"] and
                   [BANKS[k]["color"] for k in dfs][i],
                   linestyle="--", alpha=0.7)
    savefig("S1_amount_dist", "ALL")

    # 1f — Client segment distribution
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("S1 — Segmento de Clientes por Banco", fontsize=13, fontweight="bold")
    for ax, (name, df) in zip(axes, dfs.items()):
        cfg = BANKS[name]
        seg = df["client_segment"].value_counts(normalize=True) * 100
        ax.barh(seg.index, seg.values, color=cfg["color"], alpha=0.8)
        for i, v in enumerate(seg.values):
            ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
        ax.set_title(f"{cfg['short']} — Segmentos")
        ax.set_xlabel("% clientes")
    savefig("S1_segments", "ALL")

    print("  Seccion 1 completada.")


# ---------------------------------------------------------------------------
# SECTION 2: EDA SEGMENTO ENTRETENIMIENTO
# ---------------------------------------------------------------------------

def section2_entertainment(dfs: dict) -> None:
    section("SECCIÓN 2: EDA SEGMENTO ENTRETENIMIENTO (MCC 5812/5814/7832)")

    for name, df in dfs.items():
        cfg    = BANKS[name]
        sh     = cfg["short"]
        bank_header(f"{name}")

        df_ent = df[df["is_rest_cine_bar"] == 1].copy()
        print(f"  Filas entretenimiento: {len(df_ent):,}")
        if cfg["has_labels"]:
            print(f"  Tasa fraude: {df_ent['is_fraud_int'].mean():.4f} "
                  f"({int(df_ent['is_fraud_int'].sum())} fraudes)")

        # 2a — Fraud rate or volume per MCC
        fig, ax = plt.subplots(figsize=(8, 4))
        mcc_lbl = {5812: "Restaurante\n(5812)", 5814: "FastFood\n(5814)", 7832: "Cinema\n(7832)"}
        if cfg["has_labels"]:
            stats = df_ent.groupby("DE18_merchant_category_code")["is_fraud_int"].agg(["count", "mean"])
            bars  = ax.bar([mcc_lbl.get(m, str(m)) for m in stats.index],
                           stats["mean"].values, color=cfg["color"], alpha=0.85)
            ax.set_ylabel("Tasa de fraude")
            for bar, (_, row) in zip(bars, stats.iterrows()):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f"n={int(row['count'])}\n{row['mean']:.3f}",
                        ha="center", va="bottom", fontsize=9)
            ax.set_title(f"S2 [{sh}] — Tasa de Fraude por MCC Entretenimiento", fontweight="bold")
        else:
            vol = df_ent["DE18_merchant_category_code"].value_counts().sort_index()
            bars = ax.bar([mcc_lbl.get(m, str(m)) for m in vol.index],
                          vol.values, color=cfg["color"], alpha=0.85)
            ax.set_ylabel("Volumen de transacciones")
            for bar, v in zip(bars, vol.values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 5, f"n={v}", ha="center", va="bottom", fontsize=9)
            ax.set_title(f"S2 [{sh}] — Volumen por MCC Entretenimiento (sin labels)", fontweight="bold")
        savefig("S2_ent_fraud_mcc", sh)

        # 2b — Amount distribution
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        if cfg["has_labels"]:
            for lbl, sub in [("Legitima", df_ent[df_ent["is_fraud_int"] == 0]),
                              ("Fraude",   df_ent[df_ent["is_fraud_int"] == 1])]:
                if len(sub) > 0:
                    axes[0].hist(np.log1p(sub["amount_usd"]), bins=40, alpha=0.55,
                                 label=f"{lbl} (n={len(sub)})", density=True)
            axes[0].set_xlabel("log1p(amount_usd)")
            axes[0].set_title(f"[{sh}] Monto: Fraude vs Legitima")
            axes[0].legend()
        else:
            axes[0].hist(np.log1p(df_ent["amount_usd"].dropna()), bins=40,
                         color=cfg["color"], alpha=0.8)
            axes[0].set_xlabel("log1p(amount_usd)")
            axes[0].set_title(f"[{sh}] Distribución de Monto (sin labels)")

        # Boxplot por MCC
        df_ent["mcc_name"] = df_ent["DE18_merchant_category_code"].map(MCC_TARGET)
        df_ent.boxplot(column="amount_usd", by="mcc_name", ax=axes[1])
        axes[1].set_title(f"[{sh}] Monto por tipo de establecimiento")
        axes[1].set_xlabel("")
        plt.suptitle(f"S2 [{sh}] — Distribución de Montos en Entretenimiento", fontweight="bold")
        savefig("S2_ent_amount", sh)

        # 2c — Hourly analysis
        fig, ax1 = plt.subplots(figsize=(11, 4))
        h_vol = df_ent.groupby("hour_local").size()
        ax1.bar(h_vol.index, h_vol.values, alpha=0.4, color=cfg["color"], label="Volumen")
        ax1.set_xlabel("Hora local")
        ax1.set_ylabel("Volumen txns", color=cfg["color"])
        if cfg["has_labels"]:
            ax2 = ax1.twinx()
            h_fraud = df_ent.groupby("hour_local")["is_fraud_int"].mean()
            ax2.plot(h_fraud.index, h_fraud.values, color="black",
                     marker="o", markersize=4, linewidth=2, label="Tasa fraude")
            ax2.set_ylabel("Tasa de fraude", color="black")
            ax2.legend(loc="upper left")
        ax1.set_title(f"S2 [{sh}] — Volumen y Fraude por Hora (Entretenimiento)", fontweight="bold")
        ax1.legend(loc="upper right")
        savefig("S2_ent_hour", sh)

        # 2d — POS entry mode
        fig, ax = plt.subplots(figsize=(10, 4))
        if cfg["has_labels"]:
            pe = (df_ent.groupby("DE22_pos_entry_mode")["is_fraud_int"]
                  .agg(["count", "mean"]).reset_index()
                  .sort_values("count", ascending=False).head(8))
            x_pos = range(len(pe))
            bars = ax.bar(x_pos, pe["mean"].values, color=cfg["color"], alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(pe["DE22_pos_entry_mode"].astype(str).values)
            ax.set_ylabel("Tasa de fraude")
            for bar, (_, row) in zip(bars, pe.iterrows()):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f"n={int(row['count'])}", ha="center", va="bottom", fontsize=8)
            ax.set_title(f"S2 [{sh}] — Fraude por POS Entry Mode (Entretenimiento)", fontweight="bold")
        else:
            pe = df_ent["DE22_pos_entry_mode"].value_counts().head(8)
            ax.bar(pe.index.astype(str), pe.values, color=cfg["color"], alpha=0.8)
            ax.set_ylabel("Volumen")
            ax.set_title(f"S2 [{sh}] — Distribución POS Entry Mode (Entretenimiento, sin labels)",
                         fontweight="bold")
        ax.set_xlabel("POS Entry Mode")
        savefig("S2_ent_channel", sh)

        # 2e — Distance from home
        fig, ax = plt.subplots(figsize=(10, 4))
        dist_bins = [-1, 10, 100, 500, 2000, 10000, 1e9]
        dist_lbls = ["<10", "10-100", "100-500", "500-2k", "2k-10k", ">10k"]
        if cfg["has_labels"]:
            for lbl, sub in [("Legitima", df_ent[df_ent["is_fraud_int"] == 0]),
                              ("Fraude",   df_ent[df_ent["is_fraud_int"] == 1])]:
                if len(sub) > 0:
                    binned = pd.cut(sub["distance_from_home_km"], bins=dist_bins,
                                   labels=dist_lbls).value_counts().reindex(dist_lbls, fill_value=0)
                    ax.plot(dist_lbls, binned.values / binned.sum(), marker="o",
                            label=f"{lbl} (n={len(sub)})")
            ax.set_title(f"S2 [{sh}] — Distancia del Hogar: Fraude vs Legitima (Entretenimiento)",
                         fontweight="bold")
            ax.legend()
        else:
            binned = pd.cut(df_ent["distance_from_home_km"], bins=dist_bins,
                           labels=dist_lbls).value_counts().reindex(dist_lbls, fill_value=0)
            ax.bar(dist_lbls, binned.values, color=cfg["color"], alpha=0.8)
            ax.set_title(f"S2 [{sh}] — Distancia del Hogar (Entretenimiento, sin labels)",
                         fontweight="bold")
        ax.set_xlabel("Distancia desde casa (km)")
        ax.set_ylabel("Proporción / Conteo")
        savefig("S2_ent_distance", sh)

        # 2f — PIN/EMV heatmap (labeled banks only)
        if cfg["has_labels"]:
            fig, ax = plt.subplots(figsize=(9, 4))
            pe_fraud = (df_ent.groupby("DE22_pos_entry_mode")["is_fraud_int"]
                        .agg(["count", "mean"]).reset_index()
                        .sort_values("count", ascending=False).head(8))
            pe_pivot  = pe_fraud.set_index("DE22_pos_entry_mode")[["mean"]]
            sns.heatmap(pe_pivot.T, annot=True, fmt=".3f", cmap="YlOrRd",
                        ax=ax, cbar_kws={"label": "Tasa fraude"})
            ax.set_title(f"S2 [{sh}] — Tasa Fraude por Entry Mode (Entretenimiento)", fontweight="bold")
            ax.set_ylabel("")
            savefig("S2_ent_pinemv", sh)

    print("  Seccion 2 completada.")


# ---------------------------------------------------------------------------
# SECTION 3: VALIDACIÓN DE NUEVAS FEATURES
# ---------------------------------------------------------------------------

def section3_new_features(dfs: dict) -> None:
    section("SECCIÓN 3: VALIDACIÓN DE 10 NUEVAS FEATURES")

    labeled_dfs = {k: v for k, v in dfs.items() if BANKS[k]["has_labels"]}

    # 3a — is_dinner_time + is_bar_time
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("S3 — is_dinner_time y is_bar_time vs Fraude (Global)", fontweight="bold")
    for col_i, feat in enumerate(["is_dinner_time", "is_bar_time"]):
        for row_i, (name, df) in enumerate(dfs.items()):
            ax  = axes[col_i][row_i]
            cfg = BANKS[name]
            vol = df[feat].value_counts().sort_index()
            ax.bar(["No", "Si"], vol.values, color=cfg["color"], alpha=0.5, label="Volumen")
            ax.set_ylabel("Conteo")
            if cfg["has_labels"]:
                ax2 = ax.twinx()
                fr  = df.groupby(feat)["is_fraud_int"].mean()
                ax2.plot(["No", "Si"], fr.values, color="black", marker="o", linewidth=2)
                ax2.set_ylabel("Tasa fraude")
            ax.set_title(f"{feat}\n[{cfg['short']}]")
    savefig("S3_time_features", "ALL")

    # 3b — txn_same_evening: global vs entretenimiento
    for name, df in dfs.items():
        cfg = BANKS[name]
        sh  = cfg["short"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        fig.suptitle(f"S3 [{sh}] — txn_same_evening: señal global vs entretenimiento",
                     fontweight="bold")
        for ax, (label, subset) in zip(axes, [("Global", df),
                                               ("Solo Entretenimiento", df[df["is_rest_cine_bar"] == 1])]):
            bins  = sorted(subset["txn_same_evening"].unique())
            vols  = [subset[subset["txn_same_evening"] == b].shape[0] for b in bins]
            ax.bar(range(len(bins)), vols, color=cfg["color"], alpha=0.5, label="Volumen")
            ax.set_xticks(range(len(bins)))
            ax.set_xticklabels([str(b) for b in bins])
            ax.set_xlabel("txn_same_evening")
            ax.set_ylabel("Conteo")
            if cfg["has_labels"]:
                ax2  = ax.twinx()
                fr   = subset.groupby("txn_same_evening")["is_fraud_int"].mean()
                ax2.plot(range(len(bins)), [fr.get(b, 0) for b in bins],
                         color="black", marker="o", linewidth=2, label="Tasa fraude")
                ax2.set_ylabel("Tasa fraude")
            ax.set_title(label)
        if cfg["has_labels"]:
            axes[0].set_title("Global — señal fuerte (card-testing ECOM)")
            axes[1].set_title("Entretenimiento — señal desaparece")
        savefig("S3_txn_same_evening", sh)

    # 3c — mcc_diversity_last_3
    for name, df in dfs.items():
        cfg = BANKS[name]
        sh  = cfg["short"]
        fig, ax = plt.subplots(figsize=(8, 4))
        vals = sorted(df["mcc_diversity_last_3"].unique())
        vols = [df[df["mcc_diversity_last_3"] == v].shape[0] for v in vals]
        ax.bar(range(len(vals)), vols, color=cfg["color"], alpha=0.5, label="Volumen")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([str(v) for v in vals])
        ax.set_xlabel("Diversidad de MCCs (últimas 3 txns)")
        ax.set_ylabel("Conteo")
        if cfg["has_labels"]:
            ax2 = ax.twinx()
            fr  = df.groupby("mcc_diversity_last_3")["is_fraud_int"].mean()
            ax2.plot(range(len(vals)), [fr.get(v, 0) for v in vals],
                     color="black", marker="o", linewidth=2, label="Tasa fraude")
            ax2.set_ylabel("Tasa fraude")
            ax2.legend(loc="upper right")
            peak_val = vals[int(np.argmax([fr.get(v, 0) for v in vals]))]
            print(f"  [{sh}] mcc_diversity pico fraude en diversity={peak_val}")
        ax.set_title(f"S3 [{sh}] — MCC Diversity (últimas 3 txns) vs Fraude", fontweight="bold")
        savefig("S3_mcc_diversity", sh)

    # 3d — is_split_amount (all banks together)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("S3 — is_split_amount: Ocurrencias por Banco", fontweight="bold")
    counts = {BANKS[k]["short"]: int(v["is_split_amount"].sum()) for k, v in dfs.items()}
    axes[0].bar(list(counts.keys()), list(counts.values()),
                color=[BANKS[k]["color"] for k in dfs], alpha=0.8)
    axes[0].set_ylabel("Ocurrencias (is_split_amount=1)")
    axes[0].set_title("Conteo total por banco")
    for i, (sh, cnt) in enumerate(counts.items()):
        axes[0].text(i, cnt + 0.1, str(cnt), ha="center", va="bottom")

    fraud_rates = {}
    for name, df in labeled_dfs.items():
        sh   = BANKS[name]["short"]
        subs = df[df["is_split_amount"] == 1]
        fraud_rates[sh] = subs["is_fraud_int"].mean() if len(subs) > 0 else 0
    axes[1].bar(list(fraud_rates.keys()), list(fraud_rates.values()),
                color=[BANKS[k]["color"] for k in labeled_dfs], alpha=0.8)
    axes[1].set_ylabel("Tasa de fraude cuando is_split=1")
    axes[1].set_title("Precisión cuando flag activo")
    for i, (sh, fr) in enumerate(fraud_rates.items()):
        axes[1].text(i, fr + 0.01, f"{fr:.2f}", ha="center", va="bottom")
    savefig("S3_split_amount", "ALL")

    # 3e — amount_entertainment_daily
    for name, df in dfs.items():
        cfg = BANKS[name]
        sh  = cfg["short"]
        fig, ax = plt.subplots(figsize=(9, 4))
        if cfg["has_labels"]:
            bins  = [0, 0.01, 50, 100, 200, 500, 1e6]
            blbls = ["0 (sin ent)", "0-50", "50-100", "100-200", "200-500", ">500"]
            df["_ent_bin"] = pd.cut(df["amount_entertainment_daily"], bins=bins, labels=blbls)
            fr = df.groupby("_ent_bin")["is_fraud_int"].agg(["count", "mean"]).reindex(blbls)
            bars = ax.bar(blbls, fr["mean"].values, color=cfg["color"], alpha=0.8)
            ax.set_ylabel("Tasa de fraude")
            for bar, (_, row) in zip(bars, fr.iterrows()):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f"n={int(row['count']) if not pd.isna(row['count']) else 0}",
                        ha="center", va="bottom", fontsize=8)
            df.drop(columns=["_ent_bin"], inplace=True)
            ax.set_title(f"S3 [{sh}] — Gasto Diario Entretenimiento vs Fraude", fontweight="bold")
        else:
            ent_pos = df[df["amount_entertainment_daily"] > 0]["amount_entertainment_daily"]
            if len(ent_pos) > 0:
                ax.hist(np.log1p(ent_pos), bins=40, color=cfg["color"], alpha=0.8)
                ax.set_xlabel("log1p(amount_entertainment_daily)")
            ax.set_title(f"S3 [{sh}] — Distribución Gasto Diario Entretenimiento (sin labels)",
                         fontweight="bold")
        ax.set_xlabel("Gasto diario en entretenimiento (USD)")
        savefig("S3_ent_daily_amt", sh)

    # 3f — time_to_next_txn_min
    for name, df in dfs.items():
        cfg = BANKS[name]
        sh  = cfg["short"]
        fig, ax = plt.subplots(figsize=(10, 4))
        vals = np.log1p(df["time_to_next_txn_min"].dropna())
        if cfg["has_labels"]:
            for lbl, mask in [("Legitima", df["is_fraud_int"] == 0),
                               ("Fraude",   df["is_fraud_int"] == 1)]:
                sub = np.log1p(df.loc[mask, "time_to_next_txn_min"].dropna())
                if len(sub) > 0:
                    ax.hist(sub, bins=50, alpha=0.5, density=True, label=f"{lbl} (n={len(sub)})")
            ax.legend()
        else:
            ax.hist(vals, bins=50, color=cfg["color"], alpha=0.8, density=True)
        ax.set_xlabel("log1p(time_to_next_txn_min)")
        ax.set_ylabel("Densidad")
        ax.set_title(f"S3 [{sh}] — Tiempo hasta Siguiente Txn", fontweight="bold")
        savefig("S3_time_to_next", sh)

    # 3g — is_no_pin_entertainment (all banks)
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.25
    x     = np.arange(2)
    for i, (name, df) in enumerate(dfs.items()):
        cfg = BANKS[name]
        if cfg["has_labels"]:
            fr = df.groupby("is_no_pin_entertainment")["is_fraud_int"].mean()
            vals = [fr.get(0, 0), fr.get(1, 0)]
            ax.bar(x + i * width, vals, width, label=cfg["short"],
                   color=cfg["color"], alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Con PIN\n(normal/ECOM)", "Sin PIN en Ent.\n(contactless POS)"])
    ax.set_ylabel("Tasa de fraude")
    ax.set_title("S3 — is_no_pin_entertainment: Contactless POS en Restaurante/Cine es NORMAL",
                 fontweight="bold")
    ax.legend()
    savefig("S3_no_pin_ent", "ALL")

    # 3h — consecutive_ent_flag
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, df in dfs.items():
        cfg    = BANKS[name]
        streak = df.groupby("client_id")["consecutive_ent_flag"].max()
        counts = streak.value_counts().sort_index()
        ax.bar(np.arange(len(counts)) + list(dfs.keys()).index(name) * 0.25,
               counts.values, 0.25, label=cfg["short"], color=cfg["color"], alpha=0.8)
    ax.set_xlabel("consecutive_ent_flag (max streak por cliente)")
    ax.set_ylabel("N clientes")
    ax.set_title("S3 — Streak Consecutivo en Entretenimiento (hallazgo: señal nula a ≥3)",
                 fontweight="bold")
    ax.legend()
    savefig("S3_consec_ent", "ALL")

    # 3i — Feature correlation heatmap (labeled banks)
    for name, df in labeled_dfs.items():
        cfg      = BANKS[name]
        sh       = cfg["short"]
        all_feats = NEW_FEATURES + EXISTING_FEATURES
        corrs    = {}
        for feat in all_feats:
            if feat in df.columns:
                valid = df[[feat, "is_fraud_int"]].dropna()
                if len(valid) > 10 and valid[feat].std() > 0:
                    r, _ = pointbiserialr(valid["is_fraud_int"], valid[feat])
                    corrs[feat] = r
                else:
                    corrs[feat] = 0.0

        corr_df = pd.DataFrame.from_dict(corrs, orient="index", columns=["corr"])
        corr_df = corr_df.sort_values("corr")
        print(f"\n  [{sh}] Correlaciones punto-biserial con is_fraud:")
        for feat, row in corr_df.iterrows():
            bar = "+" * int(abs(row["corr"]) * 100) if row["corr"] >= 0 else "-" * int(abs(row["corr"]) * 100)
            print(f"    {feat:<35} {row['corr']:+.4f} |{bar}|")

        fig, ax = plt.subplots(figsize=(6, 8))
        colors  = ["crimson" if v < 0 else "steelblue" for v in corr_df["corr"].values]
        ax.barh(corr_df.index, corr_df["corr"].values, color=colors, alpha=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Correlación punto-biserial con is_fraud")
        ax.set_title(f"S3 [{sh}] — Discriminabilidad de Features\n(azul=positivo fraude, rojo=negativo)",
                     fontweight="bold")
        savefig("S3_new_feat_corr", sh)

    print("  Seccion 3 completada.")


# ---------------------------------------------------------------------------
# SECTION 4: CROSS-BANK COMPARISON
# ---------------------------------------------------------------------------

def section4_crossbank(dfs: dict) -> None:
    section("SECCIÓN 4: COMPARACIÓN CROSS-BANCO — DISCRIMINABILIDAD DE FEATURES")

    labeled_dfs = {k: v for k, v in dfs.items() if BANKS[k]["has_labels"]}

    # 4a — Feature discriminability: BO vs BR
    all_feats  = NEW_FEATURES + EXISTING_FEATURES
    feat_corrs = {sh: {} for sh in ["BO", "BR"]}

    for name, df in labeled_dfs.items():
        sh = BANKS[name]["short"]
        for feat in all_feats:
            if feat in df.columns:
                valid = df[[feat, "is_fraud_int"]].dropna()
                if len(valid) > 10 and valid[feat].std() > 0:
                    r, _ = pointbiserialr(valid["is_fraud_int"], valid[feat])
                    feat_corrs[sh][feat] = abs(r)
                else:
                    feat_corrs[sh][feat] = 0.0

    bo_vals = [feat_corrs["BO"].get(f, 0) for f in all_feats]
    br_vals = [feat_corrs["BR"].get(f, 0) for f in all_feats]
    order   = np.argsort([max(b, br) for b, br in zip(bo_vals, br_vals)])[::-1]

    fig, ax = plt.subplots(figsize=(10, 9))
    y = np.arange(len(all_feats))
    ax.barh(y - 0.15, [bo_vals[i] for i in order], 0.3,
            label="BO", color="steelblue", alpha=0.8)
    ax.barh(y + 0.15, [br_vals[i] for i in order], 0.3,
            label="BR", color="crimson", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([all_feats[i] for i in order])
    ax.set_xlabel("|Correlación punto-biserial| con is_fraud")
    ax.set_title("S4 — Discriminabilidad de Features: BO vs BR\n(nuevas + existentes)",
                 fontweight="bold")
    ax.legend()
    ax.axvline(0.05, color="gray", linestyle="--", alpha=0.5, label="umbral 0.05")
    savefig("S4_feat_discriminability", "ALL")

    # 4b — Feature distributions: BO vs BR vs GT
    key_feats = ["txn_same_evening", "mcc_diversity_last_3", "time_to_next_txn_min",
                 "amount_entertainment_daily", "is_dinner_time", "is_bar_time"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("S4 — Distribución de Features Clave: BO vs BR vs GT", fontweight="bold")
    for ax, feat in zip(axes.flatten(), key_feats):
        for name, df in dfs.items():
            cfg = BANKS[name]
            if feat in df.columns:
                vals = df[feat].dropna()
                if vals.std() > 0:
                    ax.hist(np.log1p(vals) if vals.max() > 5 else vals,
                            bins=30, alpha=0.45, density=True,
                            label=cfg["short"], color=cfg["color"])
        ax.set_title(feat, fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlabel("log1p" if df[feat].max() > 5 else "valor")
    savefig("S4_feature_dist", "ALL")

    # 4c — GT risk profiles (unsupervised)
    gt_name = [k for k in dfs if "GUATEMALA" in k][0]
    df_gt   = dfs[gt_name]
    section(f"  GT — Perfiles de Riesgo (sin labels, análisis descriptivo)")

    risk_flags = {
        "txn_same_evening >= 3":       (df_gt["txn_same_evening"] >= 3).sum(),
        "is_split_amount = 1":         (df_gt["is_split_amount"] == 1).sum(),
        "is_no_pin_entertainment = 1": (df_gt["is_no_pin_entertainment"] == 1).sum(),
        "consecutive_ent_flag = 1":    (df_gt["consecutive_ent_flag"] == 1).sum(),
        "mcc_diversity_last_3 = 2":    (df_gt["mcc_diversity_last_3"] == 2).sum(),
        "is_bar_time + ent":
            ((df_gt["is_bar_time"] == 1) & (df_gt["is_rest_cine_bar"] == 1)).sum(),
    }
    print(f"\n  Filas GT con flags de riesgo:")
    for flag, cnt in risk_flags.items():
        print(f"    {flag:<40} : {cnt:>6,} ({cnt/len(df_gt)*100:.2f}%)")

    fig, ax = plt.subplots(figsize=(10, 5))
    labels  = [k.replace(" = 1", "").replace(" = 2", "=2").replace(" >= 3", "≥3")
               for k in risk_flags.keys()]
    vals    = list(risk_flags.values())
    bars    = ax.bar(range(len(labels)), vals, color="forestgreen", alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Transacciones con flag activo")
    ax.set_title("S4 [GT] — Flags de Riesgo en Guatemala (sin labels)", fontweight="bold")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(v), ha="center", va="bottom", fontsize=9)
    savefig("S4_gt_profiles", "GT")

    # 4d — Cross-bank amount comparison by MCC (3x3 grid)
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("S4 — Distribución de Montos por MCC × Banco", fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(3, 3, figure=fig)
    mcc_list   = list(MCC_TARGET.items())
    bank_list  = list(dfs.items())

    for row_i, (mcc_code, mcc_name) in enumerate(mcc_list):
        for col_i, (bank_name, df) in enumerate(bank_list):
            cfg    = BANKS[bank_name]
            ax     = fig.add_subplot(gs[row_i, col_i])
            subset = df[df["DE18_merchant_category_code"] == mcc_code]
            if len(subset) > 0:
                ax.hist(np.log1p(subset["amount_usd"].dropna()), bins=30,
                        color=cfg["color"], alpha=0.8, density=True)
                if cfg["has_labels"] and subset["is_fraud_int"].sum() > 0:
                    fraud_sub = subset[subset["is_fraud_int"] == 1]
                    if len(fraud_sub) > 0:
                        ax.hist(np.log1p(fraud_sub["amount_usd"].dropna()), bins=20,
                                color="black", alpha=0.5, density=True,
                                histtype="step", linewidth=2, label="Fraude")
                        ax.legend(fontsize=7)
            ax.set_title(f"{mcc_name} | {cfg['short']}\n(n={len(subset):,})", fontsize=9)
            ax.set_xlabel("log1p(amount_usd)" if row_i == 2 else "")

    savefig("S4_ent_comparison", "ALL")

    print("  Seccion 4 completada.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print("="*70)
    print("  EDA 3 BANCOS — FRAUDE EN ENTRETENIMIENTO (REST/CINE/BAR)")
    print("="*70)
    print(f"  Figuras en: {FIG_DIR}/")

    # Load
    dfs = load_banks()

    # Feature engineering (per bank)
    for name, df in list(dfs.items()):
        bank_header(f"Feature engineering: {name}")
        df = add_base_features(df)
        df = add_new_features(df)
        print(f"  [velocity] {BANKS[name]['short']} ({len(df):,} filas)...")
        df = add_client_velocity_eda(df)
        dfs[name] = df

    # Sections
    section1_overview(dfs)
    section2_entertainment(dfs)
    section3_new_features(dfs)
    section4_crossbank(dfs)

    # Summary
    section("RESUMEN DE HALLAZGOS CLAVE")
    print("""
  1. MCC 5813 (Bar), 7922, 7941, 7996 AUSENTES en los 3 datasets
     → is_broader_entertainment = is_rest_cine_bar en estos datos

  2. txn_same_evening: señal FUERTE global (96% fraude a ≥3 txns en BO)
     → PROVIENE de card-testing ECOM, NO de entretenimiento POS
     → En entretenimiento: señal desaparece (~0.75% constante)

  3. is_no_pin_entertainment: señal NEGATIVA de fraude
     → Contactless (sin PIN) en restaurante/cine es comportamiento NORMAL
     → No usar como flag de alerta en este segmento

  4. amount_entertainment_daily: señal NEGATIVA
     → Clientes con gasto entretenimiento ese día = patrones locales legítimos
     → Fraude ocurre en días SIN entretenimiento (ECOM/ATM)

  5. mcc_diversity_last_3: pico fraude en diversity=2 (no en 3)
     → Semi-cycling de MCC es más sospechoso que máxima diversidad

  6. Bug corregido: velocidades (1h/24h) ahora usan txn_datetime
     (DE12+DE13) en vez de DE7 (que daba timestamps de 1970)

  7. Fraude en entretenimiento: 0.75-0.85% vs 3.21-4.92% global
     → Señal 6x más débil → necesita más datos etiquetados o reglas heurísticas
    """)

    print(f"\n[done] {len(os.listdir(FIG_DIR))} figuras en {FIG_DIR}/")


if __name__ == "__main__":
    main()
