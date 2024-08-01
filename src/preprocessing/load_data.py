import pandas as pd

def load_datasets():
    
    data=pd.read_csv('faces/labels.csv')
    return data