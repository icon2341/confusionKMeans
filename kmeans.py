import matplotlib.pyplot as plt
import pandas
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# take the data in the csv and convert it into an array


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # open the csv data file (for the first user for now)
    file = open("data/139t2.csv")

    features = []
    skipFirst = True
    for line in file:
        if (skipFirst):
            skipFirst = False
            continue
        elements = line.strip().split(',')
        features.append(elements)

    #print(features)

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

    #add data to frame
    data_x_long = [i[47] for i in features]
    data_y_long = [i[2] for i in features]

    data_x = data_x_long[1::4]
    data_y = data_y_long[1::4]

    df = pandas.DataFrame({'data_x': data_x, 'data_y': data_y})

    print(df.to_string)
    print(kmeans.cluster_centers_)


    plt.scatter(df['data_x'], df['data_y'], c=kmeans.labels_.astype(float)[1::4], s=50, alpha=0.5)

    plt.title("Clustering of the presence of questions vs the furrowed brow")

    plt.ylabel("is_question")
    plt.xlabel("AU7 Present")

    plt.show()
    # cen_x = [i[0] for i in finalLoc]
    # cen_y = [i[1] for i in finalLoc]
    # df['cen_x'] = df.map({0: cen_x[0], 1: cen_x[1]})
    # df['cen_y'] = df.map({0: cen_y[0], 1: cen_y[1]})
    # # define and map colors
    # colors = ['#DF2020', '#81DF20', '#2095DF']
    # df['c'] = df.cluster.map({0: colors[0], 1: colors[1], 2: colors[2]})



