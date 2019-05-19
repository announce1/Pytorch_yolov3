import cv2
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
current_palette = list(sns.xkcd_rgb.values())


########################
## 描述：计算一个Ground Truth bounding box到所有聚类中心的距离(IOU)
## box:一个Ground Truth bounding box，包含该box的宽和高
## cluster:聚类中心(ndarray),k个聚类中心给成的列表
########################
def iou(box, clusters):
    x = np.minimum(clusters[:, 0], box[0]) # 将聚类中心的宽与box的宽比较，取得宽的最小值
    y = np.minimum(clusters[:, 1], box[1]) # 将聚类中心的高与box的高比较，取得宽的最小值
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


########################
## 描述： 计算kmeans聚类中心
## boxes:所有Ground Truth组成的ndarray
## k:    k个聚类中心
## dist：簇的中心的计算方法
## seed：随机数种子
########################
def kmeans(boxes, k, dist=np.median,seed=1):
    rows = boxes.shape[0]  # Ground Truth的数量
    distances     = np.empty((rows, k)) ## N row x k cluster
    last_clusters = np.zeros((rows,))

    np.random.seed(seed)
    
    # a1 = np.random.choice(a=5, size=3, replace=False, p=None)
    # 参数意思分别 是从a 中以概率P，随机选择size个, p没有指定的时候相当于是一致的分布
    # replacement 代表的意思是抽样之后还放不放回去，False不放回抽样，True放回抽样
    
    # initialize the cluster centers to be k items
    clusters = boxes[np.random.choice(rows, k, replace=False)] # 从Ground Truth(boxes)中随机取出k个作为聚类中心的初值

    while True:
        # Step 1: allocate each item to the closest cluster centers
        for icluster in range(k): # for循环比较耗时，对k进行迭代循环，减少迭代次数，减少耗时
            distances[:,icluster] = 1 - iou(clusters[icluster], boxes) # 循环计算每个cluster到所有Ground Truth Bounding box的IOU

        nearest_clusters = np.argmin(distances, axis=1) # 在维度1上对distance计算最小值(最近)

        if (last_clusters == nearest_clusters).all():
            break

        # Step 2: calculate the cluster centers as mean (or median) of all the cases in the clusters.
        for cluster in range(k):
            # boxes[nearest_clusters == cluster]找出每个聚类对应的Ground Truth bounding box
            # 计算每个簇中Ground Truth bounding box的均值，作为新的簇
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters

    return clusters, nearest_clusters, distances


def parse_anno(annotation_path):
    anno = open(annotation_path, 'r')
    result = []
    for line in anno:
        s = line.strip().split(' ')
        image = cv2.imread(s[0])
        image_h, image_w = image.shape[:2]
        s = s[1:]
        box_cnt = len(s) // 5
        for i in range(box_cnt):
            x_min, y_min, x_max, y_max = float(s[i*5+0]), float(s[i*5+1]), float(s[i*5+2]), float(s[i*5+3])
            width  = (x_max - x_min) / image_w
            height = (y_max - y_min) / image_h
            result.append([width, height])
    result = np.asarray(result)
    return result


def plot_cluster_result(clusters,nearest_clusters,WithinClusterSumDist,wh,k):
    for icluster in np.unique(nearest_clusters):
        pick = nearest_clusters==icluster
        c = current_palette[icluster]
        plt.rc('font', size=8)
        plt.plot(wh[pick,0],wh[pick,1],"p",
                 color=c,
                 alpha=0.5,label="cluster = {}, N = {:6.0f}".format(icluster,np.sum(pick)))
        plt.text(clusters[icluster,0],
                 clusters[icluster,1],
                 "c{}".format(icluster),
                 fontsize=20,color="red")
        plt.title("Clusters=%d" %k)
        plt.xlabel("width")
        plt.ylabel("height")
    plt.legend(title="Mean IoU = {:5.4f}".format(WithinClusterSumDist))
    plt.tight_layout()
    plt.savefig("./kmeans.jpg")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_txt", type=str, default="./raccoon_dataset/train.txt")
    parser.add_argument("--anchors_txt", type=str, default="./data/raccoon_anchors.txt")
    parser.add_argument("--cluster_num", type=int, default=9)
    args = parser.parse_args()
    anno_result = parse_anno(args.dataset_txt)
    clusters, nearest_clusters, distances = kmeans(anno_result, args.cluster_num)

    # sorted by area
    area = clusters[:, 0] * clusters[:, 1]
    indice = np.argsort(area)
    clusters = clusters[indice]
    with open(args.anchors_txt, "w") as f:
        for i in range(args.cluster_num):
            width, height = clusters[i]
            f.writelines(str(width) + " " + str(height) + " ")

    WithinClusterMeanDist = np.mean(distances[np.arange(distances.shape[0]),nearest_clusters])
    plot_cluster_result(clusters, nearest_clusters, 1-WithinClusterMeanDist, anno_result, args.cluster_num)
