import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn import preprocessing


def run(fold):
    # load the full training data with folds
    df = pd.read_csv("input/adult_folds.csv")
    
    # list of numerical columns
    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]
    
    # map targets to 0s and 1s
    target_mapping = {"<=50K": 0, ">50K": 1}
    df["income"] = df["income"].map(target_mapping)

    # drop rows where mapping failed and enforce int dtype
    df = df.dropna(subset=["income"])
    df["income"] = df["income"].astype(int)
    
    # all columns are features except kfold & income columns
    features = [f for f in df.columns if f not in ("kfold", "income")]
    
    # fill all NaN values with NONE for categorical columns
    for col in features:
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # label encode categorical features
    for col in features:
        if col not in num_cols:
            lbl = preprocessing.LabelEncoder()
            df.loc[:, col] = lbl.fit_transform(df[col])
        
    # split into train and validation
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    x_train = df_train[features].values
    x_valid = df_valid[features].values
    
    # initialize xgboost model
    model = xgb.XGBClassifier(
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    
    model.fit(x_train, df_train["income"].values)
    
    # predict probabilities
    valid_preds = model.predict_proba(x_valid)[:, 1]
    
    # AUC score
    auc = metrics.roc_auc_score(df_valid["income"].values, valid_preds)
    print(f"Fold = {fold}, AUC = {auc:.4f}")


if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)
