from mydatasets.Dataset import Dataset
from mydatasets.Feature import Feature

class Adult(Dataset):  # for FL with multiple clients
    def __init__(self, isFL=False):
        name = "adult"
        rawdata_path = "./rawdata/{}.csv".format(name)
        mode = 'FL' if isFL else 'ML'
        datasets_path = "./datasets/{}/".format(name)
        results_path = "./results/{}/".format(name)
        sensitive_attributes = [Feature("sex", ["Male"], ["Female"], "Male", "Female")]
        target = Feature("income", ">50K", "<=50K", "above-50K", "at-or-below-50K")
        cat_columns = []  # categorical features already label-encoded in rawdata/adult.csv
        all_columns = [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship",
            "capital-gain", "capital-loss", "hours-per-week", "native-country",
            "sex", "income"
        ]
        number_of_clients = 12 if isFL else 1
        num_clients_per_round = 4 if isFL else 1
        num_epochs = 10
        learning_rate = 0.01
        super().__init__(name, rawdata_path, mode, datasets_path, results_path, sensitive_attributes, target, cat_columns, all_columns, number_of_clients,
                         num_clients_per_round, num_epochs, learning_rate)

    def custom_preprocess(self, df):
        return df
