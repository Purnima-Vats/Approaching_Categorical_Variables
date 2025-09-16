import copy
import pandas as pd 

from sklearn import metrics 
from sklearn import preprocessing 
import xgboost as xgb

def mean_target_encoding(data):
    # make a copy of dataframe
    df = copy.deepcopy(data)
    
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
    df.loc[:, "income"] = df.income.map(target_mapping).astype(int)

    # all columns are features except income and kfold columns
    features = [
        f for f in df.columns if f not in ("kfold", "income") and f not in num_cols
    ]
    
    # fill all NaN values with NONE
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # label encode categorical features
    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col])
        df.loc[:, col] = lbl.transform(df[col])

    # a list to store 5 validation dataframes
    encoded_dfs = []
    
    # go over all folds
    for fold in range(5):
        # fetch training and validation data
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        
        # for all feature columns (categorical ones)
        for column in features:
            # create dict of category:mean target
            mapping_dict = dict(
                df_train.groupby(column)["income"].mean()
            )
            # add mean target encoded column
            df_valid.loc[:, column + "_enc"] = df_valid[column].map(mapping_dict)
            
        # append to our list of encoded validation dataframes
        encoded_dfs.append(df_valid)
    
    # create full data frame again
    encoded_df = pd.concat(encoded_dfs, axis=0)

    # keep only numeric cols + encoded cols + target + kfold
    use_cols = num_cols + [col for col in encoded_df.columns if col.endswith("_enc")] + ["income", "kfold"]
    encoded_df = encoded_df[use_cols]

    return encoded_df

def run(df, fold):
    # get training and validation data
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # all columns are features except income and kfold
    features = [f for f in df.columns if f not in ("kfold", "income")]
    
    # scale training and validation data
    x_train = df_train[features].values
    x_valid = df_valid[features].values
    
    # initialize xgboost model
    model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=7,
        eval_metric="auc"
    )
    
    # fit model on training data
    model.fit(x_train, df_train.income.values)
    
    # predict on validation data
    valid_preds = model.predict_proba(x_valid)[:, 1]
    
    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.income.values.astype(int), valid_preds)
    
    print(f"Fold = {fold}, AUC = {auc:.4f}")

if __name__ == "__main__":
    # read data
    df = pd.read_csv("input/adult_folds.csv")

    # create mean target encoded categories
    df = mean_target_encoding(df)
    
    # run training and validation for 5 folds
    for fold_ in range(5):
        run(df, fold_)

