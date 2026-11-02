"""VS Code entrypoint: python run.py --demo or --data data/raw/fraudTrain.csv"""
import argparse
import pandas as pd
from src.dataset import load_kaggle, demo_transactions
from src.pipeline import fit_models


def main():
    parser=argparse.ArgumentParser(description="Two independent unsupervised anomaly models")
    parser.add_argument("--data",default="data/raw/fraudTrain.csv",help="Kaggle fraudTrain.csv")
    parser.add_argument("--test-data",default=None,help="Separate chronological holdout (fraudTest.csv)")
    parser.add_argument("--test-accounts",type=int,default=300,help="Accounts sampled from fraudTest (0=all)")
    parser.add_argument("--test-per-account",type=int,default=100,help="Transactions per test account (0=all)")
    parser.add_argument("--demo",action="store_true",help="Use fully synthetic smoke-test data")
    parser.add_argument("--accounts",type=int,default=25,help="Sample this many accounts")
    parser.add_argument("--per-account",type=int,default=100,help="Maximum chronological events per account (0=all)")
    parser.add_argument("--epochs",type=int,default=2)
    parser.add_argument("--width",type=int,default=32)
    parser.add_argument("--layers",type=int,default=2)
    parser.add_argument("--output",default="outputs")
    parser.add_argument("--eval-labels",default=None,help="Optional CSV containing trans_num,is_fraud; used ONLY after scoring")
    args=parser.parse_args()
    if args.epochs < 1: parser.error("--epochs must be >= 1")
    if args.accounts < 5: parser.error("--accounts must be >= 5")
    if args.demo:
        print("Running independently generated SYNTHETIC DATA smoke test (not Kaggle)")
        data=demo_transactions(accounts=max(args.accounts,6),per_account=args.per_account if args.per_account else 30)
    else:
        print("Loading Kaggle file (explicitly excluding is_fraud, names, addresses)...")
        data=load_kaggle(args.data,args.accounts,args.per_account)
    print(f"Read {len(data):,} transactions from {data.account.nunique()} accounts")
    test_data = None
    if args.test_data:
        if args.demo:
            parser.error("--demo and --test-data cannot be combined")
        print("Loading separate out-of-time test data, without labels...")
        # load_kaggle 0 accounts means use all available accounts; bounded by RAM.
        test_data = load_kaggle(args.test_data, args.test_accounts, args.test_per_account, seed=123)
        print(f"Test: {len(test_data):,} transactions from {test_data.account.nunique()} accounts")
    labels = None
    if args.test_data and args.eval_labels is None:
        args.eval_labels = args.test_data
    if args.eval_labels:
        # Evaluation labels stay outside the model's training and features.
        labels = pd.read_csv(args.eval_labels, usecols=["trans_num", "is_fraud"])
        labels = labels.rename(columns={"trans_num": "tx_id"})
    result, summary=fit_models(data,epochs=args.epochs,width=args.width,layers=args.layers,output_dir=args.output,labels=labels,test_data=test_data)
    print("\nHighest prioritized anomalies (not confirmed fraud):")
    print(result[["tx_id","transaction_anomaly_score","temporal_anomaly_score","hybrid_priority_score","hybrid_priority_level"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
