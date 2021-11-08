import math

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
    # get the data
    df = pd.read_csv("data/556t3.csv")

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

    # #render PCA
    # plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
    # plt.xlabel('PCA 1')
    # plt.ylabel('PCA 2')

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

    # plt.show()

    k_clusters = 4

    kmeans = KMeans(
        init="random",  # random initial centroid position
        n_clusters=k_clusters,  # number of centroids/clusters of the data
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

    raw_symbols = SymbolValidator().values
    namestems = []
    namevariants = []
    symbols = []
    for i in range(0, len(raw_symbols), 3):
        name = raw_symbols[i + 2]
        symbols.append(raw_symbols[i])
        namestems.append(name.replace("-open", "").replace("-dot", ""))
        namevariants.append(name[len(namestems[-1]):])


    # HOPKINS STATISTIC SCORING
    # generate random points of same size

    numberOfPoints = len(PCA_components)

    # calculate sum of distances from each point to nearest neighbor
    pointX = []
    pointY = []

    seed(21)
    for i in range(len(PCA_components)):
        pointX.append(random())
        pointY.append(random())

    # calculate sum of nearest neighbors for artificial
    artificialSum = 0

    for i in range(numberOfPoints):
        X1 = pointX[i]
        Y1 = pointY[i]
        minDistance = 1000000000

        for j in range(numberOfPoints):
            if i == j:
                continue
            X2 = pointX[j]
            Y2 = pointY[j]
            # calculate distance
            distance = math.sqrt(abs(Y2 - Y1) + abs(X2 - X1))

            if distance < minDistance:
                minDistance = distance

        artificialSum += minDistance

        # calculate sum of nearest neighbors for artificial
    realSum = 0

    for i in range(numberOfPoints):
        X1 = PCA_components[0][i]
        Y1 = PCA_components[1][i]
        minDistance = 1000000000

        for j in range(numberOfPoints):
            if i == j:
                continue
            X2 = PCA_components[0][j]
            Y2 = PCA_components[1][j]
            # calculate distance
            distance = math.sqrt(abs(Y2 - Y1) + abs(X2 - X1))

            if distance < minDistance:
                minDistance = distance

        realSum += minDistance

    hopkinsStatistic = realSum / (artificialSum + realSum)

    print("HOPKINS: " + str(hopkinsStatistic))

    #number of frames is taken by calculating the final stamp in seconds and dividing by frame rate, or 0.04
    NUM_FRAMES = 10510

    #cluster labels
    CONF = {"556t3": "yMR#f16L(X2n)+W3FbW-4ycGw5J{:r7cJC:7LAWe8w0<.9m#B.betS=dF>QBe9[*sf-9OAgdTk9hiG>#hYT([kF/Vql5U^z"}
    labels = encode.time16_to_frames(CONF["556t3"], NUM_FRAMES)
    #print(lables)

    #cluster purity implementation
    #print(kmeans.labels_)

    cluster_map = pd.DataFrame()
    cluster_map['START_TIMES'] = df['start_time'].tolist()
    cluster_map['PCAX'] = PCA_components[0]
    cluster_map['PCAY'] = PCA_components[1]
    cluster_map['Clusters'] = kmeans.labels_

    #k many clusters, each with odrx 0-3 values of confusion
    clusterTallys = []
    for i in range(k_clusters):
        clusterTallys.append([0, 0, 0, 0])


    print(cluster_map)

    dataLabels = []

    #map the points to the lables
    for i in range(len(cluster_map.index)):
        row = cluster_map.iloc[i]
        #calculate frame index based off of start time
        value = int(row[0]/0.04)
        labelValue = int(labels[value])
        clusterValue = int(row[-1])
        clusterTallys[clusterValue-1][labelValue] += 1
        dataLabels.append(labelValue)



    #now calculate cluster purity
    clusterMaxSum = 0
    for i in range(k_clusters):
        clusterMaxSum += max(clusterTallys[i])

    # divide by total number of points

    purityScore = clusterMaxSum/len(cluster_map.index)
    print(purityScore)


    # seperate the PCA into various Lables for plotting in particular shapes
    # EACH INDEX IS A LIST OF PCA X OR Y, THE INDEX CORRELATES TO THE DATA LABEL OF HOW CONFUSED THAT PERSON IS
    PCA_X = [[], [], [], []]
    PCA_Y = [[], [], [], []]
    colors = [[], [], [], []]

    for i in range(len(PCA_components[0])):
        # go through each PCA component pair, then go through the corresponding label to determine what group to add the PCA to
        PCA_X[dataLabels[i]].append(PCA_components[0][i])
        PCA_Y[dataLabels[i]].append(PCA_components[1][i])
        colors[dataLabels[i]].append(kmeans.labels_[i])





    fig = go.Figure()
    # data1 = fig.add_trace(go.Scatter(x=PCA_components[0],
    #                                name="Confusion Data",
    #                                y=PCA_components[1],
    #                                mode='markers',
    #                                marker=dict(color=kmeans.labels_),
    #                                ))
    #add each PCA label to scatter

    #not confused
    data1 = fig.add_trace(go.Scatter(mode = 'markers', x = PCA_X[0], y=PCA_Y[0],
                             marker_symbol="0",
                             marker=dict(size= 8,color=colors[0]),
                             name = "Not At All Confused",
                             line= dict(width= 20, color = 'black'),
                             ))

    #somewhat confused
    data1 = fig.add_trace(go.Scatter(mode='markers', x=PCA_X[1], y=PCA_Y[1],
                                     marker_symbol="1",
                                     marker=dict(size= 8, color=colors[1]),
                                     name="Slightly Confused",
                                     line= dict(width= 10, color = 'black')
                                     ))

    #very confused
    data1 = fig.add_trace(go.Scatter(mode='markers', x=PCA_X[2], y=PCA_Y[2],
                                     marker_symbol="13",
                                     marker=dict(size= 8,color=colors[2]),
                                     name="Very Confused",
                                     line= dict(width= 3, color = 'black')
                                     ))

    #exrtremely confused
    data1 = fig.add_trace(go.Scatter(mode='markers', x=PCA_X[3], y=PCA_Y[3],
                                     marker_symbol="17",
                                     marker=dict(size= 8,color=colors[3]),
                                     name="Extremely Confused",
                                     line= dict(width= 3, color = 'black')
                                     ))





    # Centroids
    data2 = fig.add_trace(go.Scatter(mode="markers", x=finalLoc[:, 0], y=finalLoc[:, 1], marker_symbol="x",
                             # marker_line_color="black", marker_color="blue",
                             marker=dict(color=[1,2,3,4]),
                             marker_line_width=2, marker_size=15,
                             name="Cluster Centroids",
                             hovertemplate="name: %{y}%{x}<br>number: %{marker.symbol}<extra></extra>"))





    fig.update_layout(
        title="Clustering of Dimension Reduced Confusion Data",
        xaxis_title="PCA 1",
        yaxis_title="PCA 2",
        legend_title="Symbols",
        font=dict(
            family="Arial",
            size=18,
        )
    )

    fig.show()




