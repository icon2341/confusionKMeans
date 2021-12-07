import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import sklearn
from joblib.numpy_pickle_utils import xrange
from kneed import KneeLocator
from numpy import number
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



# Confusion Labels
CONF_LABELS = {"903t1": "j+8KD6la:fa3?39",
"668t1": "j.uQ81e2)Y2>evw5IK*05][W{7gX.t",
"149t1": "v3<lf0m.u}2]%sE3Cv5B63g$A",
"811t1": "v3&7M4dK^67:f-v",
"556t1": "yLFa^1L<s94C}T#4UbdR6E1l#9A:L2a$>fy",
"291t1": "yLLp+2vN{(3/}*.4OWHd5&7fE6>5058*rxbaUL5ybA-3b",
"391t1": "y+lZ+0cFKx2[WRk4t+!06[4078QP!Db(4Au",
"670t1": "y+kl00]TB#4a#E]",
"913t1": "FPfUW0e*&&0W!5?3qQT4",
"425t1": "O}{M0V.B}49n}Q4kwCu",
"305t1": "G[aeB1&QXI2Lwwp349gd4nEvN4Czr{5BHco6?e%39Q$nfa$0u<",
"622t1": "G[a=j2aVbu4+hgM8qckY",
"194t1": "HfkH=1fz:p5IjUA",
"833t1": "Hfh)>0bI^H1-aT44.@)e7TKv]bn}^!buK{C",
"903t2": "j-9^*62+KB6[3{n97.CGdA9??fV=n7j4RSYk=sU.",
"668t2": "j.*}m1doH/4fmcD6AG4#8c[c-8Tek/cdsmjdu2)Ah/Z}>i:a:{kH@9U",
"279t2": "j<{971?s}C34SJA4{oK[6)yMW9wF&0dWI^}h90=O",
"843t2": "j<(#{0{G>v2eZV)2t2i-3?T#W4{xGT5HXlN63.iy6#Z@k8F6<m8{YH#9vS2%9*X0&a)Qu%cs57Xer@pb",
"149t2": "v4q1Q0zSVf0+8nx1.4Xj33D:o4hGGU6snND7!#1V9wex(9^P4)b*4GOej7{kgtT}BhSl*2",
"811t2": "v4o$f2ak.O30vS03/ILx4(B(f5FL&&6r&g&caUX>gRbHgh*du>",
"731t2": "yzTs%0r&OZ2GWym7T<)D",
"177t2": "yzX@-15PgV27l=O5jmBXb-%}e",
"556t2": "yL{sc0qV.e0@h#}1-BT}32f<U4Utl15rNT%7dn#28wsabacE=<bDJehb@l$@e1!PvgQ68&",
"291t2": "yM0Yo0@SA$3uDXC6QxVPbloj{f0)p+gDmy{jtn]&k6f0I",
"139t2": "yX6n}1gY1%a&bSZ",
"998t2": "yX9::0nW^l2F7gr3LA(A80S0z8ShL(bKcZ{ey(z5g7=oV",
"391t2": "y+XrD3+WqS8*9krax{4)b?jp0dUPCkfKlCWg[uXEjK<Ai",
"670t2": "y+V5n6rSff8myE-fYt7vjEl3F",
"913t2": "FPxCe0Gvr64UB-Y",
"425t2": "FPf7}19>ac4Ej:77GRGK8.SpVaU+fkcIF}BdU/PNg.hSJi$RC)",
"305t2": "G[JK00o11)1-jE52pfTl3aHri3Kvdw3{UB%6b?JC6V.g08X%xB9Gqg$b10LPbATv{dn.-Fhnq5@jOnWO",
"622t2": "G[La<0rSUJ4lUlo6djL98}=HBdOzO]hoN(e",
"194t2": "HfTX:0Hj5T5G8sf7@mJ$dS)Fi",
"833t2": "HfP#V0Ypej3bm0+5l6n26C}Lz7nz!79RyLfaVQ/Eb/}x<dz)R{fnqemf!29sjcY^C",
"903t3": "j-W$#24m&>5N=rF6A]iG7v-h]895TX8]S[^aO^tMcOW9RfI:M$jF$z&kSnVy",
"668t3": "j-?Td5Tr#p8D!gZeibEzg/l9DhsS^UlnnDu",
"279t3": "j>Qdv21Y4[2#Is}7e2IH95wUUcFH0=f>SpaiC*&Akd%he",
"843t3": "j>N4f1dYQZ362sP4NRdL8!Qwlblxc>b[.ItjRx]ak6xe)lzsCW",
"731t3": "yAn[30Hi}B5UQ<AfxAfl",
"177t3": "yAu=H0Hi$C7PwN1bv8=}",
"556t3": "yMR#f16L(X2n)+W3FbW-4ycGw5J{:r7cJC:7LAWe8w0<.9m#B.betS=dF>QBe9[*sf-9OAgdTk9hiG>#hYT([kF/Vql5U^z",
"291t3": "yM.<u2H%eJ3zOsQ6&=9Gc@zJRgvWp6h]dJTi*&jJ",
"139t3": "yX[6-0cFpE3*!u57y.z8ddYC2",
"998t3": "yXY&L0uzwR2yM{Z3AT/p7d/o(9%A?}a@@2cdzWGsd-<s?ff7!7gw9B.g.z&lj-7f0",
"391t3": "y=ysx1c0No2#.i&5cS.d8Ejwr9Tr^cfNU+Jkfi^8",
"670t3": "y=vCC1yqpV6=4!<9&HjKcosFo",
"913t3": "FQcHh0I6TJ6R=)Ycq2U5emFOV",
"425t3": "FP>2^0NRau3ayK+66f}n9S)F*aa>=mbx1>zc?O}VedTpveIO&Se-OF=",
"305t3": "G]j>I0pZgq2/Nuw7{s>UaZ36gdub&keSP(i",
"622t3": "G]jTd0iVyB0>eT(4>Ow*6LOQ3ab7/0bc&Gce$%#3f^O::h}m)d",
"194t3": "HgrK82%/mDb^/D]",
"833t3": "Hgn&50!*+{2ZW9oa3hjc",
"149t3": "^KD3/2gTqF7ekVK8o%oTc(aHhhqg4*kh2#p",
"811t3": "^JYT/0X<[R3rW}.5}7zJ6E1qi8A5Kubuuv-e^0)Bhm>]^jnw<R",}

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

def getFrames():
    # open each csv and get the final timestamp to get maximum times
    #initialize frames
    fileFrames = dict()
    for key in CONF_LABELS.keys():
        fileFrames[key] = 0
    print(fileFrames)

    keys =  fileFrames.keys()
    for key in keys:
        csv = open("data/" + str(key) + ".csv", "r")
        csvLines = csv.readlines()
        lastLine = csvLines[-1].strip().split(",")
        endTime = float(lastLine[1].strip())
        frames = int(endTime/.04)
        fileFrames[key] = frames
        csv.close()

    return fileFrames










# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(len(ALL_HEADERS))

    # dataset containing all featuers for all frames accross users and all modalities
    dataSet = pandas.read_csv("data/output.csv", names=ALL_HEADERS, index_col=False)

    #get label frame counts for each csv
    fileFrames = getFrames()

    # assign resulting label frames to the csvs
    labelFrames = dict()
    for key in CONF_LABELS.keys():
        labelFrames[key] = encode.time16_to_frames(CONF_LABELS[key], fileFrames[key])
    #map labels to rows
    labelMap = dict()
    # go through each file
    for file in os.listdir("data"):
        #skip output
        if(file == "output.csv"):
            continue

        filename = os.fsdecode(file)
        # open the file
        l = open("data/" + filename,"r")
        labelMap[filename[:5]] = []

        #read all the lines
        lines = l.readlines()
        for line in lines:
            #find start time
            entries = line.split(",")
            #assign use start time to get frame index.
            startTime = float(entries[0])
            startIndex = int(startTime/0.04)
            #get label at index and append to labelMap
            labelMap[str(file[:5])].append(labelFrames[str(file[:5])][startIndex])
        l.close()








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
    plt.ylabel('inertia (SSE)')
    plt.title('Inertia v.s Number of Clusters Across All Modalities')
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

    # PCA 3

    print("showing")

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

    # rename the columns of these models
    PCs_1d.columns = ["PC1_1d"]

    # "PC1_2d" means: 'The first principal component of the components created for 2-D visualization, by PCA.'
    # And "PC2_2d" means: 'The second principal component of the components created for 2-D visualization, by PCA.'
    PCs_2d.columns = ["PC1_2d", "PC2_2d"]

    PCs_3d.columns = ["PC1_3d", "PC2_3d", "PC3_3d"]

    subSet = pd.concat([subSet, PCs_1d, PCs_2d, PCs_3d], axis=1, join='inner')

    # divide the plots by cluster

    cluster0 = subSet[subSet["Cluster"] == 0]
    cluster1 = subSet[subSet["Cluster"] == 1]
    cluster2 = subSet[subSet["Cluster"] == 2]

    cluster03d = pd.concat([cluster0["PC1_3d"], cluster0["PC2_3d"], cluster0["PC3_3d"]], axis=1, join='inner')
    cluster13d = pd.concat([cluster1["PC1_3d"], cluster1["PC2_3d"], cluster1["PC3_3d"]], axis=1, join='inner')
    cluster23d = pd.concat([cluster2["PC1_3d"], cluster2["PC2_3d"], cluster2["PC3_3d"]], axis=1, join='inner')

    cluster0Centroid = cluster03d.mean(0)
    cluster1Centroid = cluster13d.mean(0)
    cluster2Centroid = cluster23d.mean(0)

    centroids = pd.concat([cluster0Centroid, cluster1Centroid, cluster2Centroid], axis=0)
    print(centroids["PC1_3d"])

    # Instructions for building the 3-D plot

    # trace1 is for 'Cluster 0'
    trace1 = go.Scatter3d(
        x=cluster0["PC1_3d"],
        y=cluster0["PC2_3d"],
        z=cluster0["PC3_3d"],
        mode="markers",
        name="Cluster 0",
        marker=dict(color='rgba(255, 128, 255, 0.8)'),
        text=None,
        opacity=.5)

    # trace2 is for 'Cluster 1'
    trace2 = go.Scatter3d(
        x=cluster1["PC1_3d"],
        y=cluster1["PC2_3d"],
        z=cluster1["PC3_3d"],
        mode="markers",
        name="Cluster 1",
        marker=dict(color='rgba(255, 128, 2, 0.8)'),
        text=None,
        opacity=.5)

    # trace3 is for 'Cluster 2'
    trace3 = go.Scatter3d(
        x=cluster2["PC1_3d"],
        y=cluster2["PC2_3d"],
        z=cluster2["PC3_3d"],
        mode="markers",
        name="Cluster 2",
        marker=dict(color='rgba(0, 255, 200, 0.8)'),
        text=None,
        opacity=.5)

    centroidTrace = go.Scatter3d(
        x=centroids["PC1_3d"],
        y=centroids["PC2_3d"],
        z=centroids["PC3_3d"],
        mode="markers",
        name="Cluster Centroid",
        marker=dict(color='black'),
        text=None,
        opacity=.7)

    data = [trace1, trace2, trace3, centroidTrace]

    title = "KMeans Clustering of Confusion Data for All Modalities"

    layout = dict(title=title,
                  xaxis=dict(title='PC1', ticklen=5, zeroline=False),
                  yaxis=dict(title='PC2', ticklen=5, zeroline=False)
                  )

    fig = dict(data=data, layout=layout)

    print(len(scaledDataSet.index))
    print(" instances clustered.")
    print("showing")

    iplot(fig)

    np.set_printoptions(threshold=np.inf)

    # print(km.labels_)

    #get bar chart of label distributions







