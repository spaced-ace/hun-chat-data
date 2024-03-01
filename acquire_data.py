import os
import datasets
import csv
import pandas as pd

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('data/oasst1'):
        data = datasets.load_dataset('OpenAssistant/oasst1')
        data.save_to_disk('data/oasst1')
    else:
        data = datasets.load_from_disk('data/oasst1')
    df = pd.concat([data['train'].to_pandas(), data['validation'].to_pandas()])
    df = df.query('lang == "en"')
    df.to_csv('data/oasst1-en.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(df.info())
