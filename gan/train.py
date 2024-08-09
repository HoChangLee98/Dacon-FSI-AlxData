import argparse
import yaml
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append("../")

from tqdm import tqdm 

from data_loaders import DataLoader
from model import GanProcessor

def main(args):
    """
    gan 모델을 학습시키고 저장하는 함수
    """
    with open(f"../configs/{args.model_name}/version{args.version}.yaml") as f:
        config = yaml.full_load(f)
    
    data_loader = DataLoader(data_dir="../data")
    
    fraud_types = data_loader.train['Fraud_Type'].unique()
    for fraud_type in tqdm(fraud_types):
        subset = data_loader.train[data_loader.train["Fraud_Type"] == fraud_type]
        subset = subset.sample(n=config["n_sample"], random_state=config["random_state"])

        gan = GanProcessor(
            subset = subset, 
            subset_name = fraud_type, 
            **config["model_params"]
        )
        gan._save(config["epochs"])
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="ctgan", help="model name of yaml file")
    parser.add_argument("--version", "-v", type=int, default=0, help="version number of yaml file")
    args = parser.parse_args()
        
    main(args)