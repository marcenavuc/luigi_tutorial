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
        data.to_csv(self.DATA_FILENAME, index=False)
        np.save(self.TARGET_FILENAME, dataset.target)

    def output(self):
        luigi.LocalTarget(self.DATA_FILENAME)
        luigi.LocalTarget(self.TARGET_FILENAME)


class FeatureExtractionTask(luigi.Task):
    RESULT_FILE = "workflow/prepared_data.npy"

    def requires(self):
        DataSetLoadTask()

    def load_data(self):
        self.data = pd.read_csv(DataSetLoadTask.DATA_FILENAME)

    def run(self):
        self.load_data()
        scaler = StandardScaler()
        result = scaler.fit_transform(self.data.values)
        np.save(self.RESULT_FILE, result)

    def output(self):
        luigi.LocalTarget(self.RESULT_FILE)


class SplitDataTask(luigi.Task):
    TEST_SIZE = luigi.Parameter(default=0.2, description="test_size")
    RANDOM_STATE = luigi.Parameter(default=1, description="random state")

    def requires(self):
        DataSetLoadTask()
        FeatureExtractionTask()

    def run(self):
        data = np.load(FeatureExtractionTask.RESULT_FILE)
        target = np.load(DataSetLoadTask.TARGET_FILENAME)
        X_train, X_test, y_train, y_test = \
            train_test_split(data, target,
                             test_size=self.TEST_SIZE,
                             random_state=self.RANDOM_STATE,
                             stratify=target)
        np.save("workflow/data_train.npy", X_train)
        np.save("workflow/data_test.npy", X_test)
        np.save("workflow/target_train.npy", y_train)
        np.save("workflow/target_test.npy", y_test)

    def output(self):
        luigi.LocalTarget("workflow/data_train.npy")
        luigi.LocalTarget("workflow/data_test.npy")
        luigi.LocalTarget("workflow/target_train.npy")
        luigi.LocalTarget("workflow/target_test.npy")


class TeachModelTask(luigi.Task):
    RANDOM_STATE = luigi.Parameter(default=1, description="random state")
    MODEL_NAME = "workflow/test_model.pickle"

    def requires(self):
        SplitDataTask()

    def run(self):
        X_train = np.load("workflow/data_train.npy")
        y_train = np.load("workflow/target_train.npy")
        model = LogisticRegression()
        result = cross_val_score(model, X_train, y_train, cv=3)
        model.fit(X_train, y_train)
        dump(model, self.MODEL_NAME)

    def output(self):
        luigi.LocalTarget(self.MODEL_NAME)
