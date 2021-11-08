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
import plotly.io as pio
import glob
from random import seed
import plotly.graph_objects as go
from random import random
from old.code import base85 as encode
import csv




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    lem = [9.439999999999999503e+00, 1.021000000000000085e+01, 0.000000000000000000e+00, 0.000000000000000000e+00, 8.000000000000000000e+00, 3.278688524590162245e+00, 1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 9.113038329999999503e-04, -6.468343949999999509e+00, -9.006716039999998813e+00, 8.604701499999999337e-01, -7.238464375499999548e+00, 4.347116150000000623e+00, -6.426943650000000119e+00, -1.147227394999999994e+01, 4.076204144000000085e+00, 3.122128874999999582e+00, -6.651401149999998097e+00, -1.374413149150000457e+01, -4.615285499999999708e+00, 4.606250049999999902e-02, 3.422568039999999145e-01, 0.000000000000000000e+00, 6.329905899999999964e-05, 5.278247085000000727e-01, 9.216903000000008228e-02, 7.120593249999999930e-02, -1.007723220000000808e-01, -3.181237600000000887e-01, -3.307145155000000836e-01, 5.124800499999995068e-02, 4.425095650000000491e-01, -7.800322539999999805e-01, 1.704407849999999836e-01, 2.412281904999999393e-01, -4.308382930000000388e-01, -1.977604304999999642e-03, 1.614588779999999793e-02, 4.437141999999999697e+00, 2.429999999999999660e-01, 0.000000000000000000e+00, 0.000000000000000000e+00, 7.844999999999999751e-01, 4.384999999999998899e-01, 1.419999999999999929e+00, 2.255000000000000338e-01, 1.471499999999999808e+00, 7.749999999999999112e-01, 1.729999999999999871e-01, 5.494999999999999885e-01, 0.000000000000000000e+00, 1.151499999999999968e+00, 0.000000000000000000e+00, 1.063499999999999890e+00, 7.599999999999999811e-02, 3.079999999999999405e-01, 1.000000000000000000e+00, 1.000000000000000000e+00, 0.000000000000000000e+00, 9.499999999999999556e-01, 1.000000000000000000e+00, 2.999999999999999889e-01, 5.500000000000000444e-01, 9.499999999999999556e-01, 5.500000000000000444e-01, 1.000000000000000000e+00, 1.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00, 0.000000000000000000e+00, 7.500000000000000000e-01, 5.500000000000000444e-01, 5.000000000000000000e-01, 0.000000000000000000e+00, 0.000000000000000000e+00]
    print(len(lem))
    directory = "data"
    filePaths =[]

    #itterate through each file in directory
    for filename in os.listdir(directory):
        #save to list and
        filePaths.append(directory + "\\" + filename)

    dfList = [pd.read_csv(filePaths[0])]

    index = 0
    for filename in filePaths:
        if index == 0:
            index += 1
            continue
        df = pd.read_csv(filename, skiprows=[0])
        dfList.append(df)


    df = pd.concat(dfList,axis=0, ignore_index=True)

    with open("data\\file.txt", 'a') as f:
        dfAsString = df.to_string()
        f.write(dfAsString)

    print(df)

    df = df.dropna()





    # Standardize the data to have a mean of ~0 and a variance of 1
    X_std = StandardScaler().fit_transform(df)

    # Create a PCA instance: pca
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X_std)
    # Plot the explained variances
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_ratio_, color='black')
    plt.xlabel('PCA features')
    plt.ylabel('variance %')
    plt.xticks(features)

    # Save components to a DataFrame
    PCA_components = pd.DataFrame(principalComponents)


    #Determine number of clusters
    ks = range(1, 10)
    inertias = []
    for k in ks:
        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=k)

        # Fit model to samples
        model.fit(PCA_components.iloc[:, :3])

        # Append the inertia to the list of inertias
        inertias.append(model.inertia_)

    plt.plot(ks, inertias, '-o', color='black')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(ks)

    plt.show()

    k_clusters = 4










