import pandas as pd
from sklearn_datasets import load_iris


def load_iris_data():
   
    iris=load_iris(as_Frame=True)
    df =iris.frame
    return 	df


if __name__=="__main__":
    df=load_iris_data()
    print(df.head())

