from surprise import Dataset
from surprise.dataset import DatasetAutoFolds
from pathlib import Path
from surprise import Reader
import pandas as pd
from surprise.prediction_algorithms.algo_base import AlgoBase
from surprise.prediction_algorithms.knns import KNNBasic
from surprise import SVD
from surprise.trainset import Trainset
from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise import accuracy
from surprise.model_selection import train_test_split
import time

def get_trained_model(algo: str, model_kwargs: dict, train_set: Trainset) -> AlgoBase:    
    if algo == 'KNN':
        model = KNNBasic(k=model_kwargs['k'], min_k=model_kwargs['min_k'], sim_options = model_kwargs['sim_options'])
    elif algo == 'NMF':
        model = NMF(n_factors=model_kwargs['n_factors'], n_epochs=model_kwargs['n_epochs'], verbose=model_kwargs['verbose'])
    elif algo == 'SVD':
        model = SVD()
    else:                
        raise Exception('Only support: SVD, KNN, NMF')
    time_start = time.time()
    model.fit(train_set)
    time_end = time.time()
    exec_time = (time_end-time_start)
    return model, exec_time




def load_ratings_from_surprise() -> DatasetAutoFolds:
    ratings = Dataset.load_builtin('ml-100k')
    return ratings

def load_ratings_from_file(ratings_filepath : Path) -> DatasetAutoFolds:
    reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    ratings = Dataset.load_from_file(ratings_filepath, reader)
    return ratings
    
def get_ratings(load_from_surprise : bool = True, ratings_file_path:str = 'ratings.csv') -> DatasetAutoFolds:
    if load_from_surprise:
        ratings = load_ratings_from_surprise()
    else:
        file_path = Path(ratings_file_path)
        ratings = load_ratings_from_file(file_path)
    return ratings





def evaluate_model(model: AlgoBase, test_set: [(int, int, float)], exec_time: float) -> dict:
    predictions = model.test(test_set)
    metrics_dict = {'fit time': exec_time}
    metrics_dict['RMSE'] = accuracy.rmse(predictions, verbose=False)
    metrics_dict['MAE'] = accuracy.mae(predictions, verbose=False)
    return metrics_dict

def train_and_evalute_model_pipeline(algo: str, model_kwargs: dict = {},
                                     load_from_surprise: bool = True) -> (AlgoBase, dict):
    data = get_ratings(load_from_surprise)
    train_set, test_set = train_test_split(data, 0.2, random_state=42)
    model, exec_time = get_trained_model(algo, model_kwargs, train_set)
    metrics_dict = evaluate_model(model, test_set, exec_time)
    return model, metrics_dict

def benchmark(params: dict) -> (pd.DataFrame, dict):    
    table = []
    for key in params.keys():        
        item = params[key]
        model, metrics_dict = train_and_evalute_model_pipeline(algo=item['algo'], 
                                                               model_kwargs=item['model_kwargs'], 
                                                               load_from_surprise=item['from_surprise'])
        new_line = [key, metrics_dict['RMSE'], metrics_dict['MAE'], metrics_dict['fit time']]        
        tabulate([new_line], tablefmt="pipe")
        table.append(new_line)
    header = ['Model Name',
          'RMSE',
          'MAE',
          'Fit Time'
          ]
    df = pd.DataFrame(table)
    df.columns = header        
    return df