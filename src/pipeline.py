"""End-to-end unsupervised two-model pipeline. No label access anywhere."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .features import StaticFeatures, TemporalFeatures, TemporalScaler
from .transformer import MaskedTabularTransformer, train_masked, leave_one_out_error
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score


def empirical_upper(reference, matrix):
    """Upper-tail anomaly ranks using *training-only* reference distribution."""
    scores=np.full(matrix.shape,np.nan,dtype=np.float32)
    for j in range(matrix.shape[1]):
        ref=np.sort(reference[:,j][np.isfinite(reference[:,j])])
        if len(ref)<5:continue
        values=matrix[:,j]
        good=np.isfinite(values)
        scores[good,j]=(np.searchsorted(ref,values[good],side="right")+1)/(len(ref)+1)
    return scores


def empirical_short_interval(train_df, score_df, columns):
    """High values mean unusually *short* time intervals (learned from train)."""
    result=pd.DataFrame(index=score_df.index)
    for col in columns:
        t=pd.to_numeric(train_df[col],errors="coerce")
        ref=np.sort(t[np.isfinite(t) & (t>=0)].to_numpy(float))
        value=pd.to_numeric(score_df[col],errors="coerce").to_numpy(float)
        arr=np.full(len(score_df),np.nan)
        valid=np.isfinite(value)&(value>=0)
        if len(ref)>=5:
            arr[valid]=1-(np.searchsorted(ref,value[valid],side="right")+1)/(len(ref)+2)
        result[col]=arr
    return result


def mean_top(percentiles,k=3):
    result=np.full(len(percentiles),np.nan,dtype=np.float32)
    for i,row in enumerate(percentiles):
        v=row[np.isfinite(row)]
        if len(v): result[i]=float(np.sort(v)[-k:].mean())
    return result


def risk_category(scores):
    return pd.Series(scores).map(lambda s: "UNSCORED" if pd.isna(s) else "CRITICAL" if s>=.99 else "HIGH" if s>=.95 else "RISKY" if s>=.80 else "NORMAL").to_numpy()


def top_feature_reasons(matrix, cols, k=3):
    out=[]
    for values in matrix:
        idx=np.where(np.isfinite(values))[0]
        best=idx[np.argsort(values[idx])[-k:][::-1]]
        out.append(" | ".join(f"{cols[j]}: {values[j]:.3f}" for j in best) if len(best) else "insufficient history")
    return out


def _indices_by_account(frame, seed=42):
    """Disjoint training/calibration/evaluation accounts (60/20/20)."""
    accounts = frame["account"].unique().copy()
    if len(accounts) < 5:
        raise ValueError("Need at least five accounts")
    rng = np.random.default_rng(seed)
    rng.shuffle(accounts)
    n_train = max(2, int(len(accounts) * .6))
    n_cal = max(1, int(len(accounts) * .2))
    if n_train + n_cal >= len(accounts):
        n_train = len(accounts) - 2
        n_cal = 1
    train = frame["account"].isin(accounts[:n_train]).to_numpy()
    cal = frame["account"].isin(accounts[n_train:n_train+n_cal]).to_numpy()
    evaluation = ~(train | cal)
    return train, cal, evaluation


def finite_row_mean(matrix):
    """Mean of observed feature errors, without all-NaN runtime warnings."""
    count=np.isfinite(matrix).sum(axis=1)
    total=np.nansum(matrix,axis=1)
    return np.divide(total,count,out=np.full(len(count),np.nan),where=count>0)


def calibration_rank(reference, values):
    """Apply an ECDF fitted to separate calibration data, not test data."""
    reference = np.sort(np.asarray(reference, float)[np.isfinite(reference)])
    values = np.asarray(values, float)
    out = np.full(len(values), np.nan)
    ok = np.isfinite(values)
    if len(reference) >= 5:
        out[ok] = (np.searchsorted(reference, values[ok], side="right") + 1) / (len(reference) + 1)
    return np.clip(out, 0, 1)


def evaluate_scores(results, label_frame, output):
    """Labels are joined strictly after model scoring, never used in training."""
    label_frame = label_frame[["tx_id", "is_fraud"]].drop_duplicates("tx_id")
    merged = results.merge(label_frame, on="tx_id", how="left", validate="one_to_one")
    report = {"evaluated_rows": len(merged), "labeled_rows": int(merged.is_fraud.notna().sum()),
              "fraud_count": int((merged.is_fraud == 1).sum()), "methods": {}}
    for name, col in {"static_transformer": "transaction_anomaly_score",
                      "temporal_transformer": "temporal_anomaly_score",
                      "hybrid": "hybrid_priority_score",
                      "isolation_forest": "isolation_forest_score",
                      "robust_zscore": "robust_zscore"}.items():
        valid = merged[col].notna() & merged["is_fraud"].notna()
        y = merged.loc[valid, "is_fraud"].astype(int).to_numpy()
        scores = merged.loc[valid, col].astype(float).to_numpy()
        if not len(y) or not np.any(y) or np.all(y == 1):
            report["methods"][name] = {"status": "undefined: evaluation needs both positive and negative labels", "rows": len(y)}
            continue
        order = np.argsort(-scores, kind="stable")
        metrics = {"pr_auc": float(average_precision_score(y, scores)),
                   "roc_auc": float(roc_auc_score(y, scores)), "rows": int(len(y))}
        for k in [min(10, len(y)), min(50, len(y)), min(100, len(y))]:
            selected = y[order[:k]]
            metrics[f"precision_at_{k}"] = float(selected.mean())
            metrics[f"recall_at_{k}"] = float(selected.sum() / y.sum())
        report["methods"][name] = metrics
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def fit_models(data,epochs=2,width=32,layers=2,seed=42,output_dir="outputs", labels=None, test_data=None):
    """Returns scored, held-out transactions only; training accounts are disjoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    training_data = data.sort_values(["account", "time", "tx_id"]).reset_index(drop=True)
    if "is_fraud" in training_data.columns:
        raise AssertionError("Labels cannot enter the model pipeline")
    if test_data is None:
        # Legacy account-disjoint pilot: useful for smoke tests, NOT out-of-time validation.
        df = training_data
        train_mask, calibration_mask, reference_mask = _indices_by_account(df, seed)
        split_description = "account-disjoint 60/20/20 (pilot; not chronological)"
    else:
        # Train/calibrate on fraudTrain only. Holdout is explicitly fraudTest.
        # Time ordering is checked to prevent historical leakage.
        test_data = test_data.sort_values(["account", "time", "tx_id"]).reset_index(drop=True)
        if "is_fraud" in test_data.columns:
            raise AssertionError("Labels cannot enter test features")
        if training_data.time.max() >= test_data.time.min():
            raise ValueError("Test file starts before training file ends: not a valid out-of-time split")
        account_ids = training_data.account.unique().copy()
        if len(account_ids) < 5:
            raise ValueError("At least five distinct training accounts required")
        rng = np.random.default_rng(seed)
        rng.shuffle(account_ids)
        n_train = max(2, int(len(account_ids) * .8))
        train_accounts = set(account_ids[:n_train])
        training_data = training_data.assign(__source="train")
        test_data = test_data.assign(__source="test")
        df = pd.concat([training_data, test_data], ignore_index=True)
        df = df.sort_values(["account", "time", "tx_id"]).reset_index(drop=True)
        is_training = df["__source"].eq("train")
        train_mask = (is_training & df.account.isin(train_accounts)).to_numpy()
        calibration_mask = (is_training & ~df.account.isin(train_accounts)).to_numpy()
        reference_mask = (~is_training).to_numpy()
        split_description = "fraudTrain account-disjoint train/calibration (80/20), fraudTest chronological holdout"
    train = df.loc[train_mask].reset_index(drop=True)
    calibration = df.loc[calibration_mask].reset_index(drop=True)
    holdout = df.loc[reference_mask].reset_index(drop=True)
    print(f"Train accounts: {train.account.nunique()}, transactions: {len(train)}",flush=True)
    print(f"Calibration accounts: {calibration.account.nunique()}, transactions: {len(calibration)}",flush=True)
    print(f"Holdout accounts: {holdout.account.nunique()}, transactions: {len(holdout)}",flush=True)

    print("[1/2] Static masked Transformer",flush=True)
    static=StaticFeatures().fit(train)
    xtr,otr,ctr=static.transform(train)
    xcal,ocal,ccal=static.transform(calibration)
    xho,oho,cho=static.transform(holdout)
    m1=MaskedTabularTransformer(len(static.numeric_cols),static.cardinalities,width=width,layers=layers)
    h1=train_masked(m1,xtr,otr,ctr,epochs=epochs,seed=seed)
    er_cal=leave_one_out_error(m1,xcal,ocal,ccal)
    er_test=leave_one_out_error(m1,xho,oho,cho)
    # Aggregate reconstruction errors, then calibrate with independent accounts.
    raw_cal=finite_row_mean(er_cal)
    raw_test=finite_row_mean(er_test)
    score1=calibration_rank(raw_cal,raw_test)
    p1=empirical_upper(er_cal,er_test)

    print("[2/2] Temporal masked Transformer + learned lower-tail rarity",flush=True)
    generator=TemporalFeatures(categorical=("merchant", "merchant_region"))
    temporal_full=generator.transform(df)
    temporal_train=temporal_full.loc[train_mask].reset_index(drop=True)
    temporal_cal=temporal_full.loc[calibration_mask].reset_index(drop=True)
    temporal_test=temporal_full.loc[reference_mask].reset_index(drop=True)
    scaler=TemporalScaler(generator.columns).fit(temporal_train)
    xt,ot=scaler.transform(temporal_train)
    xc,oc=scaler.transform(temporal_cal)
    xv,ov=scaler.transform(temporal_test)
    m2=MaskedTabularTransformer(len(generator.columns),width=width,layers=layers)
    h2=train_masked(m2,xt,ot,epochs=epochs,seed=seed+1)
    e2_cal=leave_one_out_error(m2,xc,oc)
    e2_test=leave_one_out_error(m2,xv,ov)
    recon_feature_pct=empirical_upper(e2_cal,e2_test)
    recon_score=calibration_rank(finite_row_mean(e2_cal),finite_row_mean(e2_test))
    rarity=empirical_short_interval(temporal_train,temporal_test,generator.columns)
    rarity_cal=empirical_short_interval(temporal_train,temporal_cal,generator.columns)
    group_rarity=pd.DataFrame(index=temporal_test.index)
    for group,columns in generator.groups.items():
        score=rarity[columns].max(axis=1,skipna=True)
        if group != "repeat":
            # A repeated merchant/region is not a new transition event.
            score=score.where(temporal_test[f"{group}_changed"])
        group_rarity[group+"_rarity"]=score
    group_rarity["reconstruction_percentile"]=recon_score
    temporal_raw=group_rarity.max(axis=1,skipna=True)
    # Calibrate the same max aggregation against independent calibration accounts.
    cal_groups=pd.DataFrame(index=temporal_cal.index)
    for group,columns in generator.groups.items():
        score=rarity_cal[columns].max(axis=1,skipna=True)
        if group != "repeat":
            score=score.where(temporal_cal[f"{group}_changed"])
        cal_groups[group+"_rarity"]=score
    cal_groups["reconstruction_percentile"]=calibration_rank(finite_row_mean(e2_cal), finite_row_mean(e2_cal))
    temporal_score=pd.Series(calibration_rank(cal_groups.max(axis=1).to_numpy(),temporal_raw.to_numpy()),index=temporal_raw.index)
    temporal_score=temporal_score.where(group_rarity.notna().any(axis=1))

    result=temporal_test[["tx_id","account","time","amount","category","merchant","merchant_region"]].copy()
    result["transaction_anomaly_score"]=score1
    result["transaction_risk_level"]=risk_category(score1)
    result["transaction_top_reasons"]=top_feature_reasons(p1,static.numeric_cols)
    result["temporal_anomaly_score"]=temporal_score.to_numpy(dtype=float)
    result["temporal_risk_level"]=risk_category(result["temporal_anomaly_score"])
    result["temporal_top_reasons"]=top_feature_reasons(recon_feature_pct,generator.columns)
    result["minutes_since_previous"]=temporal_test["minutes_since_previous"].to_numpy()
    for group in generator.categorical:
        result["previous_"+group]=temporal_test["previous_"+group].to_numpy()
        result[group+"_changed"]=temporal_test[group+"_changed"].to_numpy()
    for c in group_rarity.columns:
        result[c]=group_rarity[c].to_numpy()
    # Calibrate hybrid maxima using calibration accounts (not evaluation rows).
    cal_static=calibration_rank(raw_cal,raw_cal)
    cal_temporal=calibration_rank(cal_groups.max(axis=1).to_numpy(),cal_groups.max(axis=1).to_numpy())
    hybrid_ref=np.maximum(cal_static,cal_temporal)
    raw_hybrid=result[["transaction_anomaly_score","temporal_anomaly_score"]].max(axis=1)
    result["hybrid_priority_score"]=calibration_rank(hybrid_ref,raw_hybrid.to_numpy())
    # Classical unsupervised baselines for comparison (fit on training accounts only).
    iforest=IsolationForest(n_estimators=100,random_state=seed,n_jobs=-1)
    iforest.fit(xtr)
    result["isolation_forest_score"]=-iforest.score_samples(xho)
    result["robust_zscore"]=np.max(np.abs(xho),axis=1)
    result["hybrid_priority_level"]=risk_category(result["hybrid_priority_score"])
    # NOT an asserted fraud label; only diagnostic proximity check.
    prev=temporal_test.groupby("account")["amount"].shift()
    same_amount=np.isclose(temporal_test["amount"],prev,atol=.01,rtol=0,equal_nan=False)
    unchanged=np.ones(len(temporal_test),dtype=bool)
    for c in generator.categorical:
        unchanged &= ~temporal_test[c+"_changed"].to_numpy(bool)
    result["possible_duplicate"]=(same_amount & (temporal_test["minutes_since_previous"]<=1).fillna(False).to_numpy() & unchanged)
    # Explanations are independently built from whichever rarity groups exist.
    explanations=[]
    for i in range(len(result)):
        parts=[]
        for group in generator.groups:
            v=group_rarity.iloc[i].get(group+"_rarity",np.nan)
            if np.isfinite(v) and v>=.8:
                parts.append(f"{group} unusual-short-interval rarity={v:.3f}")
        if np.isfinite(recon_score[i]) and recon_score[i]>=.95:
            parts.append(f"temporal reconstruction percentile={recon_score[i]:.3f}")
        if bool(result.iloc[i]["possible_duplicate"]):
            parts.append("possible near-duplicate (diagnostic only)")
        explanations.append(" | ".join(parts) if parts else "no strong temporal signal")
    result["temporal_final_reason"]=explanations
    result=result.sort_values("hybrid_priority_score",ascending=False,na_position="last").reset_index(drop=True)

    for name,score in [("static","transaction_anomaly_score"),("temporal","temporal_anomaly_score"),("hybrid","hybrid_priority_score")]:
        subset=result.sort_values(score,ascending=False,na_position="last").head(100)
        subset.to_csv(output/f"top_{name}.csv",index=False)
    torch.save({"state_dict":m1.state_dict(),"features":static.numeric_cols},output/"static_model.pt")
    torch.save({"state_dict":m2.state_dict(),"features":generator.columns},output/"temporal_model.pt")
    summary={"dataset":"unlabeled Kaggle-format transactions","train_accounts":int(train.account.nunique()),"holdout_accounts":int(holdout.account.nunique()),"train_rows":int(len(train)),"holdout_rows":int(len(holdout)),"static_numeric_features":static.numeric_cols,"temporal_features":generator.columns,"temporal_rarity_groups":generator.groups,"static_train_mse":h1,"temporal_train_mse":h2,"label_used_for_training":False,"split":split_description, "score_calibration":"ECDF on separate calibration accounts","risk_meaning":"anomaly prioritization, NOT fraud probability"}
    if labels is not None:
        summary["evaluation"] = evaluate_scores(result,labels,output)
        if summary["evaluation"]["fraud_count"] == 0:
            print("WARNING: No fraud-positive test transactions. Increase --test-accounts or test history.", flush=True)
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("Saved results:",output.resolve(),flush=True)
    return result,summary
