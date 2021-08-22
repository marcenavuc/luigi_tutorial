import luigi
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression


class DataSetLoadTask(luigi.Task):
    DATA_FILENAME = "workflow/data.csv"
    TARGET_FILENAME = "workflow/target.npy"

    def run(self):
        dataset = load_wine()
        data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        data.to_csv(self.output()['data'].path, index=False)
        np.save(self.output()['target'].path, dataset.target)

    def output(self):
        return {
            "data": luigi.LocalTarget(self.DATA_FILENAME),
            "target": luigi.LocalTarget(self.TARGET_FILENAME)
        }


class FeatureExtractionTask(luigi.Task):
    RESULT_FILE = "workflow/prepared_data.npy"

    def requires(self):
        return DataSetLoadTask()

    def load_data(self):
        self.data = pd.read_csv(self.input()['data'].path)

    def run(self):
        self.load_data()
        scaler = StandardScaler()
        result = scaler.fit_transform(self.data.values)
        np.save(self.output().path, result)

    def output(self):
        return luigi.LocalTarget(self.RESULT_FILE)


class SplitDataTask(luigi.Task):
    TEST_SIZE = luigi.Parameter(default=0.2, description="test_size")
    RANDOM_STATE = luigi.Parameter(default=1, description="random state")

    def requires(self):
        return {
            "data": DataSetLoadTask(),
            "feature": FeatureExtractionTask()
        }

    def run(self):
        data = np.load(self.input()['feature'].path)
        target = np.load(self.input()['data']['target'].path)
        X_train, X_test, y_train, y_test = \
            train_test_split(data, target,
                             test_size=self.TEST_SIZE,
                             random_state=self.RANDOM_STATE,
                             stratify=target)
        np.save(self.output()['X_train'].path, X_train)
        np.save(self.output()['X_test'].path, X_test)
        np.save(self.output()['y_train'].path, y_train)
        np.save(self.output()['y_test'].path, y_test)

    def output(self):
        return {
            "X_train": luigi.LocalTarget("workflow/data_train.npy"),
            "X_test": luigi.LocalTarget("workflow/data_test.npy"),
            "y_train": luigi.LocalTarget("workflow/target_train.npy"),
            "y_test": luigi.LocalTarget("workflow/target_test.npy")
        }


class TeachModelTask(luigi.Task):
    RANDOM_STATE = luigi.Parameter(default=1, description="random state")
    MODEL_NAME = "workflow/test_model.pickle"

    def requires(self):
        return SplitDataTask()

    def run(self):
        X_train = np.load(self.input()['X_train'].path)
        y_train = np.load(self.input()['y_train'].path)
        model = LogisticRegression()
        result = cross_val_score(model, X_train, y_train, cv=3)
        model.fit(X_train, y_train)
        dump(model, self.output().path)

    def output(self):
        return luigi.LocalTarget(self.MODEL_NAME)
