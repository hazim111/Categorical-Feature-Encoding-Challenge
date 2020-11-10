#src/train.py
import os
import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree
from sklearn import preprocessing
import argparse
import joblib

import config

import model_dispatcher

def run(fold,model):
    df = pd.read_csv(config.training_data_with_folds)
    df_test = pd.read_csv(config.test_data_loc)

    #for c in df.columns:
        #lbl = preprocessing.LabelEncoder()
        #lbl.fit(df[c].values.tolist())
        #df.loc[:, c] = lbl.transform(df[c].values.tolist())

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    y_train = train_df.target.values
    y_valid = valid_df.target.values

    train_df = train_df.drop(["id", "target", "kfold"], axis=1)
    valid_df = valid_df.drop(["id", "target", "kfold"], axis=1)

    valid_df = valid_df[train_df.columns]

    label_encoders = {}
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        #train_df.loc[:, c] = train_df.loc[:, c].astype(str).fillna("NONE")
        #valid_df.loc[:, c] = valid_df.loc[:, c].astype(str).fillna("NONE")
        #df_test.loc[:, c] = df_test.loc[:, c].astype(str).fillna("NONE")
        lbl.fit(train_df[c].values.tolist() + 
                valid_df[c].values.tolist() + df_test[c].values.tolist() )
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders[c]= lbl


    clf = model_dispatcher.MODELS[model]

    clf.fit(train_df,y_train)

    preds = clf.predict_proba(valid_df)[:,1]

    accuracy = metrics.roc_auc_score(y_valid,preds)

    print(f"FOLD={fold}, Accuracy = {accuracy}")
    
    joblib.dump(clf, os.path.join(config.models_location,f"{model}_{fold}.bin"))
    joblib.dump(label_encoders, os.path.join(config.models_location,f"{model}_{fold}_label_encoder.bin"))
    joblib.dump(train_df.columns, os.path.join(config.models_location, f"{model}_{fold}_columns.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold",type=int)


    parser.add_argument("--model",type=str)

    args = parser.parse_args()

    run(fold= args.fold, model=args.model)
