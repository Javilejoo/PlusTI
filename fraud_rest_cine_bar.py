"""
Fraud detection: restaurants (MCC 5812), fast food (5814), cinema (7832), bars (5813).
Train on BO-VIP + BR-PRIVADO. GT-ESTATAL has no labels.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, recall_score, precision_score,
    average_precision_score,
)
from lightgbm import LGBMClassifier, early_stopping

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
FIG = "figures"
os.makedirs(FIG, exist_ok=True)

MCC_TARGET = {5812: "Restaurant", 5813: "Bar", 5814: "FastFood", 7832: "Cinema"}

DATA = "Documentos/Copia de Datasets 3 bancos"
FILES = [
    f"{DATA}/Copia de 01_bo_vip_seed22_n100000.csv",
]

def load():
    dfs = []
    for f in FILES:
        d = pd.read_csv(f, sep=";", low_memory=False)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["is_fraud"].notna()].copy()
    df["is_fraud"] = df["is_fraud"].astype(int)
    return df

def filter_mcc(df):
    return df[df["DE18_merchant_category_code"].isin(MCC_TARGET.keys())].copy()

def viz(df_all, df_t):
    # 1) Fraud rate by MCC (global vs target)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    rate_all = (df_all.groupby("DE18_merchant_category_code")["is_fraud"]
                .agg(["count", "mean"]).reset_index()
                .sort_values("mean", ascending=False).head(15))
    rate_all["mcc_lbl"] = rate_all["DE18_merchant_category_code"].astype(str)
    sns.barplot(data=rate_all, x="mcc_lbl", y="mean", ax=axes[0], color="steelblue")
    axes[0].set_title("Top-15 MCC por tasa de fraude (todos los datos)")
    axes[0].set_ylabel("Fraud rate"); axes[0].set_xlabel("MCC")
    axes[0].tick_params(axis="x", rotation=45)

    df_t["mcc_name"] = df_t["DE18_merchant_category_code"].map(MCC_TARGET)
    rate_t = df_t.groupby("mcc_name")["is_fraud"].agg(["count", "mean"]).reset_index()
    sns.barplot(data=rate_t, x="mcc_name", y="mean", ax=axes[1], color="coral")
    axes[1].set_title("Tasa de fraude: Restaurantes/Cine/Bares")
    axes[1].set_ylabel("Fraud rate"); axes[1].set_xlabel("")
    for i, r in rate_t.iterrows():
        axes[1].text(i, r["mean"], f"n={int(r['count'])}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{FIG}/01_fraud_rate_mcc.png", dpi=110); plt.close()

    # 2) Hourly fraud rate
    fig, ax = plt.subplots(figsize=(10, 4))
    h = df_t.groupby("hour_local")["is_fraud"].mean()
    h_cnt = df_t.groupby("hour_local")["is_fraud"].count()
    ax.bar(h.index, h.values, color="indianred", alpha=0.85)
    ax2 = ax.twinx(); ax2.plot(h_cnt.index, h_cnt.values, color="navy", marker="o", label="Volumen")
    ax.set_xlabel("Hora local"); ax.set_ylabel("Fraud rate", color="indianred")
    ax2.set_ylabel("Volumen tx", color="navy")
    ax.set_title("Fraude por hora local (Rest/Cine/Bar)")
    plt.tight_layout(); plt.savefig(f"{FIG}/02_fraud_by_hour.png", dpi=110); plt.close()

    # 3) Amount distribution: fraud vs no fraud (log scale)
    fig, ax = plt.subplots(figsize=(10, 4))
    for lbl, sub in [("Legitima", df_t[df_t.is_fraud == 0]), ("Fraude", df_t[df_t.is_fraud == 1])]:
        sns.kdeplot(np.log1p(sub["amount_usd"]), ax=ax, label=lbl, fill=True, alpha=0.4)
    ax.set_xlabel("log1p(amount_usd)"); ax.set_title("Distribucion de monto (log) fraude vs legitima")
    ax.legend(); plt.tight_layout(); plt.savefig(f"{FIG}/03_amount_dist.png", dpi=110); plt.close()

    # 4) Fraud rate by channel + POS entry mode
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    ch = df_t.groupby("channel")["is_fraud"].agg(["count", "mean"]).reset_index().sort_values("mean", ascending=False)
    sns.barplot(data=ch, x="channel", y="mean", ax=axes[0], color="teal")
    axes[0].set_title("Fraude por canal"); axes[0].set_ylabel("Fraud rate")
    for i, r in ch.iterrows():
        axes[0].text(i, r["mean"], f"n={int(r['count'])}", ha="center", va="bottom", fontsize=8)
    pe = df_t.groupby("DE22_pos_entry_mode")["is_fraud"].agg(["count", "mean"]).reset_index().sort_values("mean", ascending=False).head(10)
    pe["mode"] = pe["DE22_pos_entry_mode"].astype(str)
    sns.barplot(data=pe, x="mode", y="mean", ax=axes[1], color="purple")
    axes[1].set_title("Fraude por POS entry mode")
    axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout(); plt.savefig(f"{FIG}/04_channel_pos.png", dpi=110); plt.close()

    # 5) International vs domestic + distance from home
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    intl = df_t.groupby("is_international")["is_fraud"].agg(["count", "mean"]).reset_index()
    intl["is_international"] = intl["is_international"].astype(str)
    sns.barplot(data=intl, x="is_international", y="mean", ax=axes[0], color="goldenrod")
    axes[0].set_title("Fraude internacional vs domestico"); axes[0].set_ylabel("Fraud rate")
    df_t["dist_bin"] = pd.cut(df_t["distance_from_home_km"], bins=[-1, 10, 100, 500, 2000, 10000, 1e9],
                              labels=["<10", "10-100", "100-500", "500-2k", "2k-10k", ">10k"])
    db = df_t.groupby("dist_bin")["is_fraud"].agg(["count", "mean"]).reset_index()
    sns.barplot(data=db, x="dist_bin", y="mean", ax=axes[1], color="crimson")
    axes[1].set_title("Fraude por distancia desde casa (km)")
    plt.tight_layout(); plt.savefig(f"{FIG}/05_intl_distance.png", dpi=110); plt.close()

    # 6) EMV/PIN + bank tier
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    df_t["emv_pin"] = df_t["DE55_emv_data_present"].astype(str) + "/" + df_t["DE52_pin_data_present"].astype(str)
    ep = df_t.groupby("emv_pin")["is_fraud"].agg(["count", "mean"]).reset_index().sort_values("mean", ascending=False)
    sns.barplot(data=ep, x="emv_pin", y="mean", ax=axes[0], color="darkgreen")
    axes[0].set_title("Fraude por EMV/PIN")
    axes[0].set_xlabel("EMV_present / PIN_present"); axes[0].tick_params(axis="x", rotation=20)
    bt = df_t.groupby(["bank_tier", "mcc_name"])["is_fraud"].mean().reset_index()
    sns.barplot(data=bt, x="mcc_name", y="is_fraud", hue="bank_tier", ax=axes[1])
    axes[1].set_title("Fraude por MCC y nivel de banca")
    plt.tight_layout(); plt.savefig(f"{FIG}/06_emv_tier.png", dpi=110); plt.close()

    print("[viz] 6 figuras guardadas en", FIG)

def build_features(df):
    df = df.copy()
    df["DE7_transmission_datetime"] = pd.to_datetime(df["DE7_transmission_datetime"], errors="coerce")
    df["hour_utc"] = df["DE7_transmission_datetime"].dt.hour.fillna(-1).astype(int)
    df["is_weekend"] = df["day_of_week"].isin(["Sat", "Sun"]).astype(int)
    df["is_night"] = ((df["hour_local"] >= 22) | (df["hour_local"] <= 5)).astype(int)
    df["amount_log"] = np.log1p(df["amount_usd"].fillna(0))
    df["amount_vs_baseline"] = df["amount_usd"] / (df["client_baseline_amount"].replace(0, np.nan))
    df["amount_vs_baseline"] = df["amount_vs_baseline"].fillna(1.0).clip(0, 100)
    df["dist_log"] = np.log1p(df["distance_from_home_km"].fillna(0))
    df["is_international"] = df["is_international"].astype(int)
    df["approved"] = df["approved"].astype(int)
    df["pin_present"] = (df["DE52_pin_data_present"] == "Y").astype(int)
    df["emv_present"] = (df["DE55_emv_data_present"] == "Y").astype(int)
    # Contexto MCC rest/cine/bar
    df["is_rest_cine_bar"] = df["DE18_merchant_category_code"].isin([5812, 5813, 5814, 7832]).astype(int)
    df["is_night_out"] = ((df["is_night"] == 1) & (df["is_rest_cine_bar"] == 1)).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_local"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_local"] / 24)
    return df

FEATURES_NUM = [
    "amount_usd", "amount_log", "amount_vs_baseline",
    "distance_from_home_km", "dist_log",
    "hour_local", "hour_utc", "hour_sin", "hour_cos",
    "is_weekend", "is_night", "is_rest_cine_bar", "is_night_out",
    "is_international", "pin_present", "emv_present",
    "DE18_merchant_category_code", "DE22_pos_entry_mode", "DE25_pos_condition_code",
    "DE19_acquirer_country_code", "DE9_conversion_rate_billing",
    "client_baseline_amount",
    # velocity features (added in add_client_velocity)
    "time_since_last_txn_min",
    "txn_count_last_1h", "txn_count_last_24h",
    "amount_sum_last_1h", "txn_same_mcc_last_1h",
]
FEATURES_CAT = [
    "bank_tier", "client_segment", "channel", "card_brand",
    "DE60_pos_terminal_type", "day_of_week", "client_home_city",
]

def add_client_velocity(df):
    df = df.sort_values(["client_id", "DE7_transmission_datetime"]).copy()
    df = df.reset_index(drop=True)
    g = df.groupby("client_id")

    # --- features existentes ---
    df["client_tx_count"] = g.cumcount()
    df["client_amount_mean_so_far"] = g["amount_usd"].expanding().mean().reset_index(level=0, drop=True)
    df["client_amount_std_so_far"] = g["amount_usd"].expanding().std().reset_index(level=0, drop=True).fillna(0)
    df["client_amount_zscore"] = (df["amount_usd"] - df["client_amount_mean_so_far"]) / (df["client_amount_std_so_far"] + 1.0)
    df["seconds_since_prev"] = g["DE7_transmission_datetime"].diff().dt.total_seconds().fillna(1e6).clip(0, 1e6)
    df["time_since_last_txn_min"] = df["seconds_since_prev"] / 60.0
    df["client_merchant_changed"] = (g["DE42_card_acceptor_id"].shift(1) != df["DE42_card_acceptor_id"]).astype(int)

    # --- ventanas temporales por cliente (1h y 24h) ---
    ONE_HOUR  = np.timedelta64(3600,  "s")
    ONE_DAY   = np.timedelta64(86400, "s")

    counts_1h, counts_24h, amt_1h, same_mcc_1h = [], [], [], []

    for _cid, grp in df.groupby("client_id"):
        times   = grp["DE7_transmission_datetime"].values
        amounts = grp["amount_usd"].fillna(0).values
        mccs    = grp["DE18_merchant_category_code"].values
        n = len(grp)
        c1h  = np.zeros(n, dtype=np.int32)
        c24h = np.zeros(n, dtype=np.int32)
        a1h  = np.zeros(n, dtype=np.float64)
        m1h  = np.zeros(n, dtype=np.int32)
        for i in range(n):
            t = times[i]
            past = times[:i]
            mask_1h  = past >= (t - ONE_HOUR)
            mask_24h = past >= (t - ONE_DAY)
            c1h[i]  = mask_1h.sum()
            c24h[i] = mask_24h.sum()
            a1h[i]  = amounts[:i][mask_1h].sum()
            m1h[i]  = (mask_1h & (mccs[:i] == mccs[i])).sum()
        counts_1h.extend(c1h)
        counts_24h.extend(c24h)
        amt_1h.extend(a1h)
        same_mcc_1h.extend(m1h)

    df["txn_count_last_1h"]    = counts_1h
    df["txn_count_last_24h"]   = counts_24h
    df["amount_sum_last_1h"]   = amt_1h
    df["txn_same_mcc_last_1h"] = same_mcc_1h
    return df


def train_eval_specialist(df_all, seed=42):
    """Specialist: train ONLY on rest/cine/bar, no global noise."""
    df_all = build_features(df_all)
    df_all = add_client_velocity(df_all)
    df_t = df_all[df_all["DE18_merchant_category_code"].isin(MCC_TARGET.keys())].copy()
    cols = FEATURES_NUM + FEATURES_CAT + [
        "client_tx_count", "client_amount_mean_so_far", "client_amount_zscore",
        "seconds_since_prev", "client_merchant_changed",
    ]
    X = df_t[cols].copy()
    y = df_t["is_fraud"].astype(int)
    for c in FEATURES_CAT:
        X[c] = X[c].astype("category")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    X_tr, X_es, y_tr, y_es = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=seed)
    pos = y_tr.sum(); spw = (len(y_tr) - pos) / max(pos, 1)
    print(f"[specialist] n_tr={len(y_tr)} n_es={len(y_es)} n_val={len(y_val)} pos_val={int(y_val.sum())} spw={spw:.2f}")

    model = LGBMClassifier(
        n_estimators=3000, learning_rate=0.02, num_leaves=31, max_depth=-1,
        min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_alpha=0.2, reg_lambda=0.2, scale_pos_weight=spw,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    model.fit(X_tr, y_tr, categorical_feature=FEATURES_CAT,
              eval_set=[(X_es, y_es)], eval_metric="auc",
              callbacks=[early_stopping(100)])
    p = model.predict_proba(X_val)[:, 1]
    yv = y_val.values
    auc = roc_auc_score(yv, p); ap = average_precision_score(yv, p)
    print(f"[specialist] val AUC={auc:.4f}  AUC-PR={ap:.4f}  best_iter={model.best_iteration_}")
    def rk(y, p, k):
        n=len(y); tn=max(1,int(n*k/100)); o=np.argsort(p)[::-1][:tn]
        return float(y[o].sum()/max(1,y.sum())), float(y[o].sum()/tn)
    for k in [1,2,5,10]:
        r,pr = rk(yv,p,k); print(f"[specialist] top {k}%: recall={r:.3f} precision={pr:.4f}")
    prec, rec, thr = precision_recall_curve(yv, p)
    f1s = 2*prec*rec/(prec+rec+1e-12)
    bi = int(np.argmax(f1s[:-1])); bt = float(thr[bi])
    print(f"[specialist] F1 best thr={bt:.4f} F1={f1s[bi]:.4f} P={prec[bi]:.4f} R={rec[bi]:.4f}")
    yp = (p>=bt).astype(int)
    print("CM:\n", confusion_matrix(yv, yp))
    print(classification_report(yv, yp, target_names=["Legit","Fraud"], digits=4))
    return model, p, yv, X_val


def train_eval(df_all, df_t_idx, seed=42):
    """Train on ALL data, evaluate only on rest/cine/bar subset."""
    df_all = build_features(df_all)
    df_all = add_client_velocity(df_all)
    cols = FEATURES_NUM + FEATURES_CAT + [
        "client_tx_count", "client_amount_mean_so_far", "client_amount_zscore",
        "seconds_since_prev", "client_merchant_changed",
    ]
    # remove velocity cols already in FEATURES_NUM to avoid duplicates
    cols = list(dict.fromkeys(cols))
    X = df_all[cols].copy()
    y = df_all["is_fraud"].astype(int)
    is_target_mcc = df_all["DE18_merchant_category_code"].isin(MCC_TARGET.keys()).values

    for c in FEATURES_CAT:
        X[c] = X[c].astype("category")

    idx = np.arange(len(X))
    # stratify by MCC-target combo so target MCC positives appear in val
    mcc_in_target = df_all["DE18_merchant_category_code"].isin(MCC_TARGET.keys()).astype(int).values
    strat = y.values * 2 + mcc_in_target
    idx_train, idx_val = train_test_split(idx, test_size=0.25, stratify=strat, random_state=seed)
    idx_tr, idx_es = train_test_split(idx_train, test_size=0.15, stratify=strat[idx_train], random_state=seed)

    X_tr, y_tr = X.iloc[idx_tr], y.iloc[idx_tr]
    X_es, y_es = X.iloc[idx_es], y.iloc[idx_es]
    X_val, y_val = X.iloc[idx_val], y.iloc[idx_val]
    val_target_mask = is_target_mcc[idx_val]

    pos = y_tr.sum(); neg = len(y_tr) - pos
    spw = neg / max(pos, 1)
    print(f"[train] n_tr={len(y_tr)} n_es={len(y_es)} n_val={len(y_val)} (target_subset={val_target_mask.sum()})")
    print(f"[train] fraud_rate_tr={y_tr.mean():.4f} scale_pos_weight={spw:.2f}")

    model = LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.85, subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=0.1,
        scale_pos_weight=spw,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        categorical_feature=FEATURES_CAT,
        eval_set=[(X_es, y_es)],
        eval_metric="auc",
        callbacks=[early_stopping(75)],
    )

    proba_val_all = model.predict_proba(X_val)[:, 1]
    y_val_all = y_val.values
    proba_val = proba_val_all[val_target_mask]
    y_val = y_val_all[val_target_mask]
    print(f"[eval] global val: n={len(y_val_all)} positives={int(y_val_all.sum())}  AUC={roc_auc_score(y_val_all, proba_val_all):.4f}")
    print(f"[eval] target subset val: n={len(y_val)} positives={int(y_val.sum())}")
    if y_val.sum() == 0:
        print("[eval] no positives in target subset val - aborting eval")
        return {"error": "no positives"}

    # Recall @ top-K% — use global threshold, report on subset
    def recall_at_top_k(y, p, k_pct):
        n = len(y); top_n = max(1, int(n * k_pct / 100))
        order = np.argsort(p)[::-1][:top_n]
        return float(y[order].sum() / max(1, y.sum())), float(y[order].sum() / max(top_n, 1))
    for k in [1, 2, 5, 10]:
        rg, prg = recall_at_top_k(y_val_all, proba_val_all, k)
        print(f"[eval] top {k:>2}% global: recall={rg:.3f} precision={prg:.4f}")
    auc_global = roc_auc_score(y_val_all, proba_val_all)
    auc = roc_auc_score(y_val, proba_val)
    ap = average_precision_score(y_val, proba_val)
    print(f"[val] AUC-ROC global={auc_global:.4f}  AUC-ROC target={auc:.4f}  AUC-PR target={ap:.4f}")
    # Use global threshold optimized by F1 over full val
    prec_g, rec_g, thr_g = precision_recall_curve(y_val_all, proba_val_all)
    f1s_g = 2*prec_g*rec_g/(prec_g+rec_g+1e-12)
    global_best_thr = float(thr_g[int(np.argmax(f1s_g[:-1]))])
    print(f"[val] Global best thr={global_best_thr:.4f}")

    # Use global threshold on target subset
    prec_t, rec_t, thr_t = precision_recall_curve(y_val, proba_val)
    f1s_t = 2*prec_t*rec_t/(prec_t+rec_t+1e-12)
    best_i_t = int(np.argmax(f1s_t[:-1]))
    best_thr = global_best_thr
    print(f"\n--- Target subset (rest/cine/bar) @ global thr {best_thr:.4f} ---")
    y_pred_t = (proba_val >= best_thr).astype(int)
    cm = confusion_matrix(y_val, y_pred_t)
    print("CM:\n", cm)
    print(classification_report(y_val, y_pred_t, target_names=["Legit","Fraud"], digits=4))
    print(f"\n--- Global val @ global thr {best_thr:.4f} ---")
    y_pred_g = (proba_val_all >= best_thr).astype(int)
    cm_g = confusion_matrix(y_val_all, y_pred_g)
    print("CM:\n", cm_g)
    print(classification_report(y_val_all, y_pred_g, target_names=["Legit","Fraud"], digits=4))

    # Save curves (global)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fpr_g, tpr_g, _ = roc_curve(y_val_all, proba_val_all)
    axes[0].plot(fpr_g, tpr_g, label=f"ROC AUC global={auc_global:.4f}")
    axes[0].plot([0, 1], [0, 1], "r--")
    axes[0].set_title("ROC - Bolivia Global"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].legend()
    axes[1].plot(rec_g, prec_g, label=f"PR AUC global")
    axes[1].axvline(x=rec_g[int(np.argmax(f1s_g[:-1]))], color="green", ls="--", label=f"thr={best_thr:.3f}")
    axes[1].set_title("PR - Bolivia Global"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].legend()
    plt.tight_layout(); plt.savefig(f"{FIG}/07_roc_pr.png", dpi=110); plt.close()

    # Feature importance
    fi = pd.DataFrame({"feature": X.columns, "importance": model.booster_.feature_importance(importance_type="gain")})
    fi = fi.sort_values("importance", ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(fi["feature"], fi["importance"], color="steelblue")
    ax.set_title("Top-20 importancia (gain)")
    plt.tight_layout(); plt.savefig(f"{FIG}/08_importance.png", dpi=110); plt.close()

    return {
        "auc_roc_global": auc_global, "auc_roc_target": auc,
        "auc_pr_target": ap, "best_thr": best_thr,
        "best_iter": model.best_iteration_,
        "top_features": fi[["feature", "importance"]].iloc[::-1].head(10).to_dict("records"),
    }


def main():
    print("[load] cargando dataset Bolivia BO-VIP...")
    df = load()
    print(f"[load] total filas: {len(df)}, fraud rate={df['is_fraud'].mean():.4f}")
    df_t = filter_mcc(df)
    print(f"[filter] MCC rest/cine/bar: {len(df_t)} filas, fraud rate={df_t['is_fraud'].mean():.4f}")
    by_mcc = df_t.groupby("DE18_merchant_category_code")["is_fraud"].agg(["count", "mean"])
    print("[filter] por MCC:\n", by_mcc)

    viz(df, df_t)
    print("\n=== MODELO LIGHTGBM — Bolivia BO-VIP ===")
    res = train_eval(df, df_t.index)
    print("\n[RESULT]", res)

if __name__ == "__main__":
    main()
