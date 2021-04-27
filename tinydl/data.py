from config import *
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from tinydl.augmentation import *


class DataFrameClassification:
    """

    Args:
        fpath: File path or DataFrame.
        max_rows : For testing, a max number of rows to take
        file_type : csv, excel anything supported by pandas
        train_pct : split percentage
        normalize : Well. It normalizes. Either True or False

    Returns:
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


class ImageFolderClassification:
    """
    Directory type:
    - class1
        - img1
        - img2
        - ....

    - class2
        - img1
        - img2
        - ....
    """

    def __init__(
        self,
        fpath,
        max_entries=None,
        train_pct=0.8,
        image_shape=(64, 64),
        aug=[Normalize],
    ):
        self.fpath = Path(fpath)
        self.train_pct = train_pct
        self.max_rows = max_entries
        self.label_dict = {}
        self.image_shape = image_shape
        self.aug = aug

    def labelFromMap(self, x):
        return self.label_dict[x.parent.name]

    def loadAndAugment(self, X):
        X_images = []
        for im in X:
            X_images.append(cv2.resize(cv2.imread(str(im)), self.image_shape))
        X_images = np.array(X_images)

        X_images = augment(X_images, self.image_shape, self.aug)
        return X_images

    def read_data(self):
        print("Loading data")
        classes = [x for x in self.fpath.glob("*")]
        self.label_dict = {classes[x].name: x for x in range(len(classes))}
        print(f"\nLabels : {self.label_dict}")
        all_files = pd.DataFrame(
            [x for x in self.fpath.glob("*/*.png")], columns=["path"]
        )
        all_files["label"] = all_files["path"].apply(self.labelFromMap)

        X, y = all_files["path"], all_files["label"]
        trainX = X.sample(frac=self.train_pct)
        trainy = y.sample(frac=self.train_pct)
        testX = self.loadAndAugment(X.drop(trainX.index))
        trainX = self.loadAndAugment(trainX)
        testy = y.drop(trainy.index)

        print("Done loading data")
        return (
            trainX,
            trainy.to_numpy(dtype="float64"),
            testX,
            testy.to_numpy(dtype="float64"),
        )
