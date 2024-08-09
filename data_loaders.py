import pandas as pd

class DataLoader:
    """
    data loading
    """    
    def __init__(
        self, 
        data_dir:str="./data", 
    ):
        self.data_dir = data_dir
        self.train = pd.read_csv(f"{data_dir}/train.csv").drop(columns="ID")
        self.test = pd.read_csv(f"{data_dir}/test.csv").drop(columns="ID")