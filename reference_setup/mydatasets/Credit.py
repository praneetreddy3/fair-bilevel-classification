from mydatasets.Dataset import Dataset
from mydatasets.Feature import Feature

class Credit(Dataset):  # for FL with multiple clients
    def __init__(self, isFL=False):
        name = "credit"
        rawdata_path = "./rawdata/{}.csv".format(name)
        mode = 'FL' if isFL else 'ML'
        datasets_path = "./datasets/{}/".format(name)
        results_path = "./results/{}/".format(name)
        sensitive_attributes = [Feature("SEX", [1], [2], "Male", "Female")]
        target = Feature("default", 1, 0, "default", "no-default")
        cat_columns = []  # all features already numeric in rawdata/credit.csv
        all_columns = [
            "LIMIT_BAL", "EDUCATION", "MARRIAGE", "AGE",
            "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
            "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
            "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
            "SEX", "default"
        ]
        number_of_clients = 12 if isFL else 1
        num_clients_per_round = 4 if isFL else 1
        num_epochs = 10
        learning_rate = 0.01
        super().__init__(name, rawdata_path, mode, datasets_path, results_path, sensitive_attributes, target, cat_columns, all_columns, number_of_clients,
                         num_clients_per_round, num_epochs, learning_rate)

    def custom_preprocess(self, df):
        return df
