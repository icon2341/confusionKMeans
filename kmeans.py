import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import sklearn
from joblib.numpy_pickle_utils import xrange
from kneed import KneeLocator
from plotly.validators.box.marker import SymbolValidator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly as pgo
import chart_studio
import plotly.io as pio
import glob
from random import seed
import plotly.graph_objects as go
from random import random
from old.code import base85 as encode




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    directory = "data"
    filePaths = []

    #itterate through each file in directory
    for filename in os.listdir(directory):
        #save to list and
        filePaths.append(os.path.join(directory, filename))








