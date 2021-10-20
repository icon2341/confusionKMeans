import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import sklearn
from kneed import KneeLocator
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


# take the data in the csv and convert it into an array


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # open the csv data file (for the first user for now)
    file = open("data/139t2.csv")

    fLabel = []
    features = []
    skipFirst = True
    for line in file:
        if (skipFirst):
            elements = line.strip().split(',')
            for element in elements:
                fLabel.append(element)
            skipFirst = False
            continue
        elements = line.strip().split(',')
        features.append(elements)

    #fLabel.pop(fLabel.index(' curr_sentence_length'))
    #print(features)

    df = pd.read_csv("data/139t2.csv")
    #scale the data such that mean of zero and std of 1
    #dataReduced = pca.fit_transform(scale(df))

    fig = px.scatter_matrix(df,dimensions=fLabel, color=" curr_sentence_length")

    fig.update_traces(diagonal_visible = False)


    #now apply PCA

    pca = PCA()
    components = pca.fit_transform(df[fLabel])


    labels = {
        str(i): f"PC {i + 1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(4),
        color=df[" curr_sentence_length"]
    )

    fig.update_traces(diagonal_visible=False)
    fig.show()


    kmeans = KMeans(
        init="random",  # random initial centroid position
        n_clusters=3,  # number of centroids/clusters of the data
        n_init=10,  # number of initializations to perform
        max_iter=300,  # number of itterations to perform to move each centroid
        random_state=42  # the seed
    )
    # initialization with the lowest SSE (error)
    kmeans.fit(features)

    # lowest sse value of the 10 initial runs
    lowestSSE = kmeans.inertia_

    # Final locations of the centroid
    finalLoc = kmeans.cluster_centers_
    #print(finalLoc)
    # the number of iterations required to converged
    numberItter = kmeans.n_iter_

    #i now have classified data,










