import pandas as pd
from sklearn import linear_model
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
    
    # drop numerical columns
    df = df.drop(num_cols, axis=1)
    
    # map targets to 0s and 1s
    target_mapping = {"<=50K": 0, ">50K": 1}
    df["income"] = df["income"].map(target_mapping)

    # drop rows with unmapped values (if any) and enforce int dtype
    df = df.dropna(subset=["income"])
    df["income"] = df["income"].astype(int)
    
    # all columns are features except income and kfold
    features = [f for f in df.columns if f not in ("kfold", "income")]
    
    # fill all NaN values with NONE, convert to string
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # split into train/valid sets
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # OneHotEncoder
    ohe = preprocessing.OneHotEncoder(handle_unknown="ignore")
    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)
    ohe.fit(full_data[features])
    
    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])
    
    # Logistic Regression model
    model = linear_model.LogisticRegression(max_iter=1000)
    model.fit(x_train, df_train["income"].values)
    
    # predict probabilities for validation
    valid_preds = model.predict_proba(x_valid)[:, 1]
    y_true = df_valid["income"].values
    
    # sanity check
    # print(f"\nFold = {fold}")
    # print(df_valid["income"].value_counts())
    # print("y_true dtype:", y_true.dtype)
    # print("valid_preds shape:", valid_preds.shape)
    
    # AUC score
    auc = metrics.roc_auc_score(y_true, valid_preds)
    print(f"Fold = {fold}, AUC = {auc:.4f}")


if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)
