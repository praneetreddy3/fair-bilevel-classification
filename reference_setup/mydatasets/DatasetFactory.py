import fnmatch
import os
import pandas as pd
####################
from mydatasets.Law import Law
from mydatasets.Dutch import Dutch
from mydatasets.Adult import Adult
from mydatasets.Credit import Credit

def get_raw_datasets(isFL=False):
    return [
        Law(isFL), Dutch(isFL), Adult(isFL), Credit(isFL)
    ]

def get_raw_datasets_names(isFL=False):
    return [dataset.name for dataset in get_raw_datasets(isFL)]


def get_raw_dataset(dataset_name, isFL=False):
    for dataset in get_raw_datasets(isFL):
        if dataset.name == dataset_name:
            return dataset
    raise ValueError(format)

def get_datasets(dataset, dataset_name, dataset_dir):
    # Load dataset
    dir_path = os.path.join('datasets', dataset_name, dataset_dir)
    number_of_clients = len(fnmatch.filter(os.listdir(dir_path), '*_train*.csv'))

    # read clients training dataset
    if dataset.mode == "ML":
        file_name_train = dir_path + f'\\{dataset_name}_{dataset.mode}_train.csv'
        file_name_test = dir_path + f'\\{dataset_name}_{dataset.mode}_test.csv'
        # file_name_val = dir_path + '\\validation_data.csv'
    else:
        file_name_train = dir_path + f'\\{dataset_name}_{dataset.mode}_train_{{}}.csv'
        file_name_test = dir_path + f'\\{dataset_name}_{dataset.mode}_test_{{}}.csv'
        # file_name_val = dir_path + '\\validation_data.csv'

    train_datasets = []
    test_datasets = []
    for i in range(1, number_of_clients + 1):
        with open(file_name_train.format(i), 'r') as file:
            train_datasets.append(pd.read_csv(file))
        with open(file_name_test.format(i), 'r') as file:
            test_datasets.append(pd.read_csv(file))

    return train_datasets, test_datasets
