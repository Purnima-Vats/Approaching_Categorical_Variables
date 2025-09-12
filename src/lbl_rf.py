import pandas as pd

from sklearn import ensemble 
from sklearn import metrics 
from sklearn import preprocessing 

def run(fold):
    # load the full training data with folds
    df = pd.read_csv("input/cat_train_folds.csv")
    
    # all columns are features except id, target and kfold columns 
    features = [
        f for f in df.columns if f not in {"id", "target", "kfold"}
    ]
    
    # fill all NaN values with NONE
    # Note that I am converting all columns to "strings"
    # It doesn't matter because all are categories 
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
    
    # Now it's time to label encode the features
    for col in features:
        # Initialise label encoder for each feature column
        lbl = preprocessing.LabelEncoder()
        
        # fit label encoder on all data
        lbl.fit(df[col])
        
        # transform all the data
        df.loc[:, col] = lbl.transform(df[col])
    
    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True) 
    
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True) 
    
    # get training data
    x_train = df_train[features]
    
    # get validation data
    x_valid = df_valid[features]
    
    # initialise random forest model 
    model = ensemble.RandomForestClassifier(n_jobs=-1)
    
    # fit model on training data
    model.fit(x_train, df_train.target.values) 
    
    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:, 1]
    
    # get roc auc score 
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    
    # print auc
    print(f"Fold = {fold}, AUC = {auc}")
    

if __name__ == "__main__": 
    for fold_ in range(5):
        run(fold_) 