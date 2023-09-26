import argparse
import pandas as pd
from sklearn.metrics import classification_report

def report(df, label_columns, model="chatgpt"):
    y_true = df[label_columns]
    if model == "chatgpt":
        irr_pred = df['result'].str.contains('irrelevant', case=False).replace({True: 1, False: 0})
        fea_pred = df['result'].str.contains('feature request', case=False).replace({True: 1, False: 0})
        pro_pred = df['result'].str.contains('problem report', case=False).replace({True: 1, False: 0})
    else:
        fea_pred = df['result'].str.contains('Feature Request: (?!(No|Non|None|Irrelevant|N\/A)).*\\n', case=False).replace({True: 1, False: 0})
        pro_pred = df['result'].str.contains('Problem Report: (?!(No|Non|None|Irrelevant|N\/A)).*\\n', case=False).replace({True: 1, False: 0})
        irr_pred = df['result'].str.contains('Irrelevant: (Yes|Oui|Irrelevant)', case=False).replace({True: 1, False: 0})
    y_pred = pd.concat([irr_pred, fea_pred, pro_pred], axis=1)

    df['result'].str.extract('Feature Request:(.{12,}|yes)\n')
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=label_columns, 
        digits=6,
        zero_division=0
    )
    print(report)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--csv_path', action='store')
    parser.add_argument('--model', action='store')
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv_path)
    label_columns = ["irrelevant", "feature_request", "problem_report"]
        
    en_df = df[df['ori_lang'] == 'en']
    print('English')
    report(en_df, label_columns, model=args.model)
    
    print('French')
    fr_df = df[df['ori_lang'] == 'fr']
    report(fr_df, label_columns, model=args.model)
