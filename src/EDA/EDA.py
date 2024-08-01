import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import os
import numpy as np
import pandas as pd

def eda_report(data,features,target):
    
    describe_result=data.describe()
    
    eda_path = './files/modeling_output/figures/'

    if not os.path.exists(eda_path):
        os.makedirs(eda_path)

    # Exporting the file
    with open(eda_path+'describe.txt', 'w') as f:
        f.write(describe_result.to_string())

    # Exporting general info
    with open(eda_path+'info.txt','w') as f:
        sys.stdout = f
        data.info()
        sys.stdout = sys.__stdout__
        
    data['real_age'].plot(kind='hist')
    plt.title('Distribución de la edad')
    plt.show()
    plt.savefig(eda_path+'fig1.png')
    
    data['real_age'].plot(kind='box')
    plt.title('Distribución de la edad')
    plt.show()
    plt.savefig(eda_path+'fig2.png')
    
    # muestra 16 imágenes
    fig = plt.figure(figsize=(10,10))
    for i in range(15):
        fig.add_subplot(4, 4, i+1)
        plt.imshow(features[i])
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    fig.savefig(eda_path+'fig3.png')
    
    print(target[:15])
    target_1=pd.Series(target[:15])
    target_1.to_csv(eda_path+'target.txt', sep='\t', index=False)