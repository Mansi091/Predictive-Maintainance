import os
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, classification_report


np.random.seed(42)


def train_model():
    