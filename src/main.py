import sys
from src.preprocessing.load_data import load_datasets
from src.preprocessing.preprocess import preprocess_data
from src.EDA.EDA import eda_report
from src.models.built_models import string
import pandas as pd

def main():
    '''This main function progresses through various stages to process data, 
    evaluate variables, and create a robust model for predicting churned users. 
    For more detailed information, please refer to the README.md file. '''

    data = load_datasets() #Loading stage
    preprocessed_data = preprocess_data(data) #Preprocessing stage
    eda_report(data,preprocessed_data[0],preprocessed_data[1]) # Analysis stage
    string()
main()