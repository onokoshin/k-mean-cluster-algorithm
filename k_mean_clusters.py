'''
KOSHIN ONO
CSPC 5610
AI_HW4

Description:
k-means clustering is a method of vector quantization, originally from signal
processing, that is popular for cluster analysis in data mining. k-means clustering
aims to partition n observations into k clusters in which each observation belongs
to the cluster with the nearest mean, serving as a prototype of the cluster.
'''

import math
import random
import sys

# global variables
COM_MAX = sys.float_info.min
COM_MIN = sys.float_info.max
LEN_MAX = sys.float_info.min
LEN_MIN = sys.float_info.max
WID_MAX = sys.float_info.min
WID_MIN = sys.float_info.max


def get_data(arr):
    global COM_MAX
    global COM_MIN
    global LEN_MAX
    global LEN_MIN
    global WID_MAX
    global WID_MIN

    path = 'seeds_dataset.txt'
    data = open(path, 'r')

    for line in data:
        i = 1
        features = []
        for x in line.split():
            if i == 3:
                y: float = float(x)
                if COM_MIN > y:
                    COM_MIN = y
                if COM_MAX < y:
                    COM_MAX = y
                features.append(y)

            if i == 4:
                y: float = float(x)
                if LEN_MIN > y:
                    LEN_MIN = y
                if LEN_MAX < y:
                    LEN_MAX = y
                features.append(y)

            if i == 5:
                y: float = float(x)
                if WID_MIN > y:
                    WID_MIN = y
                if WID_MAX < y:
                    WID_MAX = y
                features.append(y)

            i += 1

        arr.append(features)


# initializes center values based on min and max value
def initialize_center():
    global COM_MAX
    global COM_MIN
    global LEN_MAX
    global LEN_MIN
    global WID_MAX
    global WID_MIN

    seed = []

    compact_center = random.uniform(COM_MIN, COM_MAX)
    length_center = random.uniform(LEN_MIN, LEN_MAX)
    width_center = random.uniform(WID_MIN, WID_MAX)

    seed.append(compact_center)
    seed.append(length_center)
    seed.append(width_center)

    return seed


# euclidean distance measurement
def euclidean(center_pt, one_pt):
    dist = math.sqrt(math.pow(center_pt[0] - one_pt[0], 2)
                     + math.pow(center_pt[1] - one_pt[1], 2)
                     + math.pow(center_pt[2] - one_pt[2], 2))

    return dist


# in case there is an empty cluster, fill it with another point
def mod_empty_cluster(first_cluster, second_cluster, third_cluster):
    if len(first_cluster) == 0:
        if len(second_cluster) != 0:
            first_cluster.append(second_cluster[0])
        elif len(third_cluster) != 0:
            first_cluster.append(third_cluster[0])
    elif len(second_cluster) == 0:
        if len(first_cluster) != 0:
            second_cluster.append(first_cluster[0])
        elif len(third_cluster) != 0:
            second_cluster.append(third_cluster[0])
    elif len(third_cluster) == 0:
        if len(first_cluster) != 0:
            third_cluster.append(first_cluster[0])
        elif len(second_cluster) != 0:
            third_cluster.append(second_cluster[0])


'''
Create 3 clusters based on arbitrary center_pts and distribute SeedDatas 
to appropriate cluster
'''
def create_clusters(center_pts, SeedDatas):

    first_cluster = []
    second_cluster = []
    third_cluster = []

    # each SeedData contains compactness, length, and width
    for x in SeedDatas:
        one_pt = x
        dist_1 = euclidean(center_pts[0], one_pt)
        dist_2 = euclidean(center_pts[1], one_pt)
        dist_3 = euclidean(center_pts[2], one_pt)

        # whichever is the shortest get appended to an appropriate cluster
        # dist_1 is the distance between a pt with center point 1 and so on
        if dist_1 < dist_2 and dist_1 < dist_3:
            first_cluster.append(one_pt)
        elif dist_2 < dist_1 and dist_2 < dist_3:
            second_cluster.append(one_pt)
        elif dist_3 < dist_1 and dist_3 < dist_2:
            third_cluster.append(one_pt)

    # call mod_empty_cluster in case there is an empty cluster
    mod_empty_cluster(first_cluster, second_cluster, third_cluster)

    # create clusters - a list that contains three clusters
    clusters = [first_cluster, second_cluster, third_cluster]

    return clusters


# obtains a new center based on each point in cluster
def get_new_center(cluster):
    com_total: float = 0.0
    len_total: float = 0.0
    wid_total: float = 0.0

    for x in cluster:
        com_total += x[0]
        len_total += x[1]
        wid_total += x[2]

    com_middle = com_total / len(cluster)
    len_middle = len_total / len(cluster)
    wid_middle = wid_total / len(cluster)

    result = [com_middle, len_middle, wid_middle]

    return result


# compares and checks center points
def is_equal_cluster(prev_ct_pts, new_ct_pts):
    if prev_ct_pts[0] == new_ct_pts[0] and prev_ct_pts[1] == new_ct_pts[1] and prev_ct_pts[2] == new_ct_pts[2]:
        return True
    else:
        return False


# main function to do everything :)
def main():
    global COM_MAX
    global COM_MIN
    global LEN_MAX
    global LEN_MIN
    global WID_MAX
    global WID_MIN

    # obtain dataset from text document
    SeedDatas = []
    get_data(SeedDatas)

    # initialize k-means -- it gets 3 arbitrary center points within max and min values
    center_pt_1 = initialize_center()
    center_pt_2 = initialize_center()
    center_pt_3 = initialize_center()

    center_pts = [center_pt_1, center_pt_2, center_pt_3]

    # create k-cluster and keep generating centers until unable to generate more
    ending = False

    # the loop ends when 3 old center points are the same as new center points
    while not ending:

        clusters = create_clusters(center_pts, SeedDatas)

        first_cluster = clusters[0]
        second_cluster = clusters[1]
        third_cluster = clusters[2]

        # obtain better center point using new cluster
        new_ct_pt1 = get_new_center(first_cluster)
        new_ct_pt2 = get_new_center(second_cluster)
        new_ct_pt3 = get_new_center(third_cluster)

        # A list to store three new center points
        new_ct_pts = [new_ct_pt1, new_ct_pt2, new_ct_pt3]

        # check whether the new center-points are same or different from old ones
        ending = is_equal_cluster(center_pts, new_ct_pts)

        # if the old center points and new ones are different, update them
        if not ending:
            center_pts[0] = new_ct_pts[0]
            center_pts[1] = new_ct_pts[1]
            center_pts[2] = new_ct_pts[2]

    # display the results
    print('Cluster 1:',
          'Number of data points:', len(first_cluster), '|',
          'Compactness:', center_pts[0][0], '|',
          'Length:', center_pts[0][1], '|',
          'Width:', center_pts[0][2]
          )
    print('Cluster 2:',
          'Number of data points:', len(second_cluster), '|',
          'Compactness:', center_pts[1][0], '|',
          'Length:', center_pts[1][1], '|',
          'Width:', center_pts[1][2]
          )
    print('Cluster 3:',
          'Number of data points:', len(third_cluster), '|',
          'Compactness:', center_pts[2][0], '|',
          'Length:', center_pts[2][1], '|',
          'Width:', center_pts[2][2]
          )


if __name__ == "__main__":
    main()
