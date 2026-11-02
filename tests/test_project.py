import numpy as np
import pandas as pd
import pytest
from src.dataset import demo_transactions, clean_transactions, load_kaggle
from src.features import TemporalFeatures, StaticFeatures
from src.pipeline import empirical_short_interval, fit_models


def test_label_and_pii_are_excluded():
    df=pd.DataFrame({"trans_date_trans_time":["2020-01-01 10:00:00"], "cc_num":["1234567890123456"],"merchant":["M"],"category":["groceries"],"amt":[4.5],"lat":[40],"long":[30],"merch_lat":[40],"merch_long":[30],"trans_num":["X"],"first":["PERSON"],"street":["ADDRESS"],"is_fraud":[1]})
    x=clean_transactions(df)
    assert "is_fraud" not in x and "first" not in x and "street" not in x and "cc_num" not in x
    assert "1234567890123456" not in x.to_string()


def test_temporal_features_are_in_chronological_order():
    df=demo_transactions(accounts=5,per_account=10)
    temporal=TemporalFeatures().transform(df)
    assert set(["merchant","merchant_region","repeat"])==set(TemporalFeatures().groups)
    assert temporal.groupby("account").head(1)["minutes_since_previous"].isna().all()
    assert temporal["minutes_since_previous"].dropna().ge(0).all()


def test_rare_short_time_has_higher_rarity():
    train=pd.DataFrame({"time_gap":[10.,20.,30.,40.,50.,60.,70.]})
    test=pd.DataFrame({"time_gap":[1.,1000.]})
    scores=empirical_short_interval(train,test,["time_gap"])
    assert scores.time_gap.iloc[0]>scores.time_gap.iloc[1]


def test_kaggle_csv_read_without_label(tmp_path):
    d=demo_transactions(accounts=6,per_account=8)
    # Reverse map into the externally documented Kaggle schema.
    kag=pd.DataFrame({"trans_date_trans_time":d.time,"cc_num":d.account,"merchant":d.merchant,"category":d.category,"amt":d.amount,"lat":d.home_lat,"long":d.home_lon,"merch_lat":d.merchant_lat,"merch_long":d.merchant_lon,"trans_num":d.tx_id,"is_fraud":1})
    path=tmp_path/"fraudTrain.csv"
    kag.to_csv(path,index=False)
    loaded=load_kaggle(path,accounts=4,per_account=8)
    assert len(loaded)==32 and "is_fraud" not in loaded


def test_pipeline_smoke(tmp_path):
    result,meta=fit_models(demo_transactions(8,10),epochs=1,width=16,layers=1,output_dir=tmp_path)
    assert len(result)>0
    assert meta["label_used_for_training"] is False
    assert "temporal_final_reason" in result
    assert (tmp_path/"top_static.csv").exists()
    assert (tmp_path/"top_temporal.csv").exists()
    assert (tmp_path/"top_hybrid.csv").exists()


def test_per_account_zero(tmp_path):
    import pandas as pd
    from src.dataset import load_kaggle
    raw = []
    for n in range(6):
        for j in range(9):
            raw.append(dict(trans_date_trans_time=f"2020-01-{1+n:02d} 12:{j:02d}:00",cc_num=str(123400+n),merchant="test",category="misc",amt=10,lat=40,long=-70,merch_lat=40,merch_long=-70,trans_num=f"{n}-{j}"))
    path=tmp_path/"fraudTrain.csv"
    pd.DataFrame(raw).to_csv(path,index=False)
    assert len(load_kaggle(path,accounts=6,per_account=0)) == 54


def test_calibration_and_splits():
    from src.pipeline import calibration_rank, _indices_by_account
    from src.dataset import demo_transactions
    df=demo_transactions(12,10)
    a,b,c=_indices_by_account(df)
    assert not (a&b).any() and not (a&c).any() and not (b&c).any()
    assert (a|b|c).all()
    scores=calibration_rank([0,1,2,3,4,5,6,7,8,9],[1,9])
    assert scores[0]<scores[1]


def test_evaluate_labels_separate(tmp_path):
    import pandas as pd
    from src.pipeline import evaluate_scores
    df=pd.DataFrame({"tx_id":["a","b","c","d"],"transaction_anomaly_score":[.9,.1,.8,.2],"temporal_anomaly_score":[.8,.3,.7,.4],"hybrid_priority_score":[.9,.2,.8,.3],"isolation_forest_score":[4,1,3,2],"robust_zscore":[5,1,4,2]})
    labels=pd.DataFrame({"tx_id":["a","b","c","d"],"is_fraud":[1,0,1,0]})
    metrics=evaluate_scores(df,labels,tmp_path)
    assert metrics["methods"]["hybrid"]["pr_auc"]==1.0
