from config import *
import numpy as np
import pandas as pd


class DataFrameClassification:
    """[summary]

    Args:
        fpath: File path or DataFrame.
        max_rows : For testing, a max number of rows to take
        file_type : csv, excel anything supported by pandas
        train_pct : split percentage
        normalize : Well. It normalizes. Either True or False

    Returns:
        [type]: [description]
    Loader for a table type of dataset. Anything you would use pandas for
    """

    def __init__(
        self,
        fpath,
        max_rows,
        label_col=["target"],
        file_type="csv",
        train_pct=0.8,
        normalize=True,
    ):
        self.fpath = fpath
        self.train_pct = train_pct
        self.max_rows = max_rows
        self.normalize = normalize
        self.file_type = file_type
        self.label_col = label_col

    def read_data(self):
        print("Loading data")
        if type(self.fpath) == "<class 'pandas.core.frame.DataFrame'>":
            df = self.fpath
        df = getattr(pd, f"read_{self.file_type}")(self.fpath)
        df = df.head(self.max_rows)
        print("Cols : ", df.columns)
        X, y = df.drop(self.label_col, axis=1), df[self.label_col]
        print(f"X shape : {X.shape} , y shape : {y.shape}")
        if self.normalize == True:
            X = (X - X.min()) / (X.max() - X.min())
        trainX = X.sample(frac=self.train_pct)
        trainy = y.sample(frac=self.train_pct)
        testX = X.drop(trainX.index)
        testy = y.drop(trainy.index)
        print("Done loading data")
        return (
            trainX.to_numpy(),
            trainy.to_numpy(dtype="float64"),
            testX.to_numpy(),
            testy.to_numpy(dtype="float64"),
        )
