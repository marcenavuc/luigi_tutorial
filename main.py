import luigi
from tutorial.core import TeachModelTask, DataSetLoadTask, \
    FeatureExtractionTask, SplitDataTask

if __name__ == "__main__":
    luigi.build([
        DataSetLoadTask(),
        FeatureExtractionTask(),
        SplitDataTask(),
        TeachModelTask(),
    ], local_scheduler=True)
