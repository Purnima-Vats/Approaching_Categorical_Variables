import itertools
import pandas as pd
import xgboost as xgb

from sklearn import metrics 
from sklearn import preprocessing 

def feature_engineering(df, cat_cols):
    """
    This function is used for feature engineering
    :param df: the pandas dataframe with train/test data
    :param cat_cols: list of categorical columns
    :return: dataframe with new features
    """
    # this will create all 2-combinations of categorical columns
    combi = list(itertools.combinations(cat_cols, 2))
    for c1, c2 in combi:
        df.loc[:, c1 + "_" + c2] = df[c1].astype(str) + "_" + df[c2].astype(str)
    return df

def run(fold):
    # load the full training data with folds 
    df = pd.read_csv("input/adult_folds.csv")

    # list of numerical cols 
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
    
    # list of categorical columns for feature engineering
    cat_cols = [
        c for c in df.columns if c not in num_cols
        and c not in ("kfold", "income")
    ]
    
    # add new features
    df = feature_engineering(df, cat_cols)
    
    # all columns are features except kfold & income columns
    features = [
        f for f in df.columns if f not in ("kfold", "income")
    ]
    
    # fill all NaN values with NONE (only for categorical cols)
    for col in features:
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # label encode categorical features
    for col in features:
        if col not in num_cols:
            lbl = preprocessing.LabelEncoder()
            df.loc[:, col] = lbl.fit_transform(df[col])
    
    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    if df_valid.shape[0] == 0:
        print(f"⚠️ Fold {fold} has no validation samples. Skipping.")
        return
    
    # get training data
    x_train = df_train[features].values
    x_valid = df_valid[features].values
    
    # initialize xgboost model
    model = xgb.XGBClassifier(
        n_jobs=-1,
        objective="binary:logistic",
        base_score=0.5,
        eval_metric="auc",
        use_label_encoder=False,
        n_estimators=200,
        max_depth=7
    )
    
    # fit model on training data
    model.fit(x_train, df_train.income.values)
    
    # predict on validation data
    valid_preds = model.predict_proba(x_valid)[:, 1]
    
    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    print(f"Fold = {fold}, AUC = {auc:.4f}")

if __name__ == "__main__":
    n_folds = pd.read_csv("input/adult_folds.csv")["kfold"].nunique()
    for fold_ in range(n_folds):
        run(fold_)
