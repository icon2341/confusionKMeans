import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import sklearn
from joblib.numpy_pickle_utils import xrange
from kneed import KneeLocator
from plotly.offline import iplot
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

import kmeans
from old.code import base85 as encode

pd.options.plotting.backend = "plotly"
import csv

# Feature Names
TXT_HEADERS = ['start_time', 'end_time', 'is_question', 'is_pause', 'curr_sentence_length', 'speech_rate',
               'is_edit_word', 'is_reparandum', 'is_interregnum', 'is_repair']
AUD_HEADERS = ['pcm_RMSenergy_sma', 'pcm_fftMag_mfcc_sma[1]', 'pcm_fftMag_mfcc_sma[2]', 'pcm_fftMag_mfcc_sma[3]',
               'pcm_fftMag_mfcc_sma[4]', 'pcm_fftMag_mfcc_sma[5]', 'pcm_fftMag_mfcc_sma[6]', 'pcm_fftMag_mfcc_sma[7]',
               'pcm_fftMag_mfcc_sma[8]', 'pcm_fftMag_mfcc_sma[9]', 'pcm_fftMag_mfcc_sma[10]', 'pcm_fftMag_mfcc_sma[11]',
               'pcm_fftMag_mfcc_sma[12]', 'pcm_zcr_sma', 'voiceProb_sma', 'F0_sma', 'pcm_RMSenergy_sma_de',
               'pcm_fftMag_mfcc_sma_de[1]', 'pcm_fftMag_mfcc_sma_de[2]', 'pcm_fftMag_mfcc_sma_de[3]',
               'pcm_fftMag_mfcc_sma_de[4]', 'pcm_fftMag_mfcc_sma_de[5]', 'pcm_fftMag_mfcc_sma_de[6]',
               'pcm_fftMag_mfcc_sma_de[7]', 'pcm_fftMag_mfcc_sma_de[8]', 'pcm_fftMag_mfcc_sma_de[9]',
               'pcm_fftMag_mfcc_sma_de[10]', 'pcm_fftMag_mfcc_sma_de[11]', 'pcm_fftMag_mfcc_sma_de[12]',
               'pcm_zcr_sma_de', 'voiceProb_sma_de', 'F0_sma_de']
VID_HEADERS = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r',
               'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r', 'AU01_c', 'AU02_c', 'AU04_c',
               'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c',
               'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']
ALL_HEADERS = TXT_HEADERS + AUD_HEADERS + VID_HEADERS

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(len(ALL_HEADERS))

    # dataset containing all featuers for all frames accross users and all modalities
    dataSet = pandas.read_csv("../data/output.csv", names=ALL_HEADERS, index_col=False)

    # check if any nulls
    # print(dataSet.isnull().sum())

    # standardize dataset so that data works better with K-means
    scaledDataSet = pd.DataFrame(StandardScaler().fit_transform(dataSet))

    # determine number of clusters using elbow method

    ks = range(1, 10)
    inertias = []
    for k in ks:
        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=k)

        # Fit model to samples
        model.fit(scaledDataSet)

        # Append the inertia to the list of inertias
        inertias.append(model.inertia_)

    plt.plot(ks, inertias, '-o', color='black')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(ks)

    plt.show()

    # Using sklearn
    km = sklearn.cluster.KMeans(n_clusters=3)
    km.fit(scaledDataSet)

    # Find which cluster each data-point belongs to
    clusters = km.predict(scaledDataSet)
    # Format results as a DataFrame
    # Add the cluster vector to our scaled DataFrame
    scaledDataSet["Cluster"] = clusters

    # # PCA varience graphed
    #
    # pca = PCA().fit(scaledDataSet)
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.plot(3, np.cumsum(pca.explained_variance_ratio_)[3], marker='o', markersize=6, color="black",
    #          label='3 PCA components')
    # print(np.cumsum(pca.explained_variance_ratio_)[3])
    # plt.xlabel('number of components')
    # plt.ylabel('cumulative explained variance')
    # plt.title('cumulative explained variance vs number of PCA components')
    #
    # plt.show()

    # get cluster centers
    # print(km.cluster_centers_)

    # #sampled subset of the entire scaledDataSet
    #
    #
    # #using PCA to display data
    # features = ALL_HEADERS
    #
    # pca = PCA()
    # components = pca.fit_transform(subSet)
    # labels = {
    #     str(i): f"PC {i + 1} ({var:.1f}%)"
    #     for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    # }
    #
    # fig = px.scatter_matrix(
    #     components,
    #     labels=labels,
    #     dimensions=range(4),
    #     color=subSet["Cluster"]
    # )
    # fig.update_traces(diagonal_visible=False)
    # fig.show()

    subSet = scaledDataSet  # .sample(500)

    # PCA with one principal component
    pca_1d = PCA(n_components=1)

    # PCA with two principal components
    pca_2d = PCA(n_components=2)

    # PCA with three principal components
    pca_3d = PCA(n_components=3)

    # This DataFrame holds that single principal component mentioned above
    PCs_1d = pd.DataFrame(pca_1d.fit_transform(subSet.drop(["Cluster"], axis=1)))

    # This DataFrame contains the two principal components that will be used
    # for the 2-D visualization mentioned above
    PCs_2d = pd.DataFrame(pca_2d.fit_transform(subSet.drop(["Cluster"], axis=1)))

    # And this DataFrame contains three principal components that will aid us
    # in visualizing our clusters in 3-D
    PCs_3d = pd.DataFrame(pca_3d.fit_transform(subSet.drop(["Cluster"], axis=1)))

    PCs_1d.columns = ["PC1_1d"]

    # "PC1_2d" means: 'The first principal component of the components created for 2-D visualization, by PCA.'
    # And "PC2_2d" means: 'The second principal component of the components created for 2-D visualization, by PCA.'
    PCs_2d.columns = ["PC1_2d", "PC2_2d"]

    PCs_3d.columns = ["PC1_3d", "PC2_3d", "PC3_3d"]

    subSet = pd.concat([subSet, PCs_1d, PCs_2d, PCs_3d], axis=1, join='inner')


    # Note that all of the DataFrames below are sub-DataFrames of 'plotX'.
    # This is because we intend to plot the values contained within each of these DataFrames.

    # divide the plots by cluster

    # divide the plots by cluster

    cluster0 = subSet[subSet["Cluster"] == 0]
    cluster1 = subSet[subSet["Cluster"] == 1]
    cluster2 = subSet[subSet["Cluster"] == 2]
    cluster3 = subSet[subSet["Cluster"] == 3]
    cluster4 = subSet[subSet["Cluster"] == 4]
    cluster5 = subSet[subSet["Cluster"] == 5]
    cluster6 = subSet[subSet["Cluster"] == 6]
    cluster7 = subSet[subSet["Cluster"] == 7]

    cluster02d = pd.concat([cluster0["PC1_2d"], cluster0["PC2_2d"]], axis=1, join='inner')
    cluster12d = pd.concat([cluster1["PC1_2d"], cluster1["PC2_2d"]], axis=1, join='inner')
    cluster22d = pd.concat([cluster2["PC1_2d"], cluster2["PC2_2d"]], axis=1, join='inner')
    cluster32d = pd.concat([cluster3["PC1_2d"], cluster3["PC2_2d"]], axis=1, join='inner')
    cluster42d = pd.concat([cluster4["PC1_2d"], cluster4["PC2_2d"]], axis=1, join='inner')
    cluster52d = pd.concat([cluster5["PC1_2d"], cluster5["PC2_2d"]], axis=1, join='inner')
    cluster62d = pd.concat([cluster6["PC1_2d"], cluster6["PC2_2d"]], axis=1, join='inner')
    cluster72d = pd.concat([cluster7["PC1_2d"], cluster7["PC2_2d"]], axis=1, join='inner')

    cluster0Centroid = cluster02d.mean(0)
    cluster1Centroid = cluster12d.mean(0)
    cluster2Centroid = cluster22d.mean(0)
    cluster3Centroid = cluster32d.mean(0)
    cluster4Centroid = cluster42d.mean(0)
    cluster5Centroid = cluster52d.mean(0)
    cluster6Centroid = cluster62d.mean(0)
    cluster7Centroid = cluster72d.mean(0)

    centroids = pd.concat([cluster0Centroid, cluster1Centroid, cluster2Centroid], axis=0)
    print(centroids["PC1_2d"])

    # trace1 is for 'Cluster 0'
    trace1 = go.Scatter(
        x=cluster0["PC1_2d"],
        y=cluster0["PC2_2d"],
        mode="markers",
        name="Cluster 0",
        marker=dict(color='rgba(255, 128, 255, 0.8)'),
        text=None)

    # trace2 is for 'Cluster 1'
    trace2 = go.Scatter(
        x=cluster1["PC1_2d"],
        y=cluster1["PC2_2d"],
        mode="markers",
        name="Cluster 1",
        marker=dict(color='rgba(255, 128, 2, 0.8)'),
        text=None)

    # trace3 is for 'Cluster 2'
    trace3 = go.Scatter(
        x=cluster2["PC1_2d"],
        y=cluster2["PC2_2d"],
        mode="markers",
        name="Cluster 2",
        marker=dict(color='rgba(0, 255, 200, 0.8)'),
        text=None)

    centroidsTrace = go.Scatter(
        x=centroids["PC1_2d"],
        y=centroids["PC2_2d"],
        mode="markers",
        name="Cluster Centroids",
        marker=dict(symbol=2, color='black'),
        text=None
    )
    data = [trace1, trace2, trace3, centroidsTrace]

    title = "Kmeans Clustering of All Modalities PC 1 and 2"

    layout = dict(title=title,
                  xaxis=dict(title='PC1', ticklen=5, zeroline=False),
                  yaxis=dict(title='PC2', ticklen=5, zeroline=False)
                  )

    fig = dict(data=data, layout=layout)

    iplot(fig)

    print(len(scaledDataSet.index))
    print(" instances clustered.")
    print("showing")


