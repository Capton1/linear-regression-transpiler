import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd


def train_example_model():
    house = fetch_california_housing()

    dataset = pd.DataFrame(
        data= np.c_[house['data'], house['target']],
        columns= house['feature_names'] + ['target']
    )

    train_x, test_x, train_y, test_y = train_test_split(dataset[['MedInc', 'HouseAge']].values,
                                                        dataset['target'].values,
                                                        test_size=0.2,
                                                        random_state=42)
    lr = LinearRegression()
    lr.fit(train_x, train_y)
    joblib.dump(lr, './regression.joblib')