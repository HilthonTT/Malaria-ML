import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "..")
sys.path.append(src_dir)

import tensorflow_datasets as tfds
from visualization.visualization import  open_model_results
from data.data_initializer import get_datasets
from models.model import load_model

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def main():
    dataset, dataset_info = tfds.load("malaria", with_info=True, as_supervised=True, split=['train'])
    _, _, test_dataset = get_datasets(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO) 
    
    model = load_model()
    
    open_model_results(model, test_dataset)
    

if __name__ == "__main__":
    main()