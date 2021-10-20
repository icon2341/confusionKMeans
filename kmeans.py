import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import sklearn
from joblib.numpy_pickle_utils import xrange
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
import plotly as pgo
import chart_studio
import plotly.io as pio
import glob

# take the data in the csv and convert it into an array


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #get the data
    df = pd.read_csv("data/139t2.csv")

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

    plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

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

    kmeans = KMeans(
        init="random",  # random initial centroid position
        n_clusters=4,  # number of centroids/clusters of the data
        n_init=10,  # number of initializations to perform
        max_iter=300,  # number of itterations to perform to move each centroid
        random_state=42  # the seed
    )
    # initialization with the lowest SSE (error)
    kmeans.fit(PCA_components)

    # lowest sse value of the 10 initial runs
    lowestSSE = kmeans.inertia_



    # Final locations of the centroid
    finalLoc = kmeans.cluster_centers_
    print(finalLoc)
    # print(finalLoc)
    # the number of iterations required to converged
    numberItter = kmeans.n_iter_

    # i now have classified data,

    #out = px.scatter(x=PCA_components[0], y=PCA_components[1], mode='markers', marker=dict(color=kmeans.labels_))
    data1 = pgo.graph_objs.Scatter(x=PCA_components[0],
                           y=PCA_components[1],
                           mode='markers',
                           marker=dict(color=kmeans.labels_),
                           name="PCA1 vs PCA2"
                           )

    data2 = pgo.graph_objs.Scatter()



    out = [data1]

    pgo.offline.iplot(out)

