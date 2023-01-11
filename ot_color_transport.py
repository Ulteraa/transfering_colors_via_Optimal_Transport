import ot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.pyplot import legend
from sklearn.cluster import MiniBatchKMeans
from PIL import Image

def load_data():
    # ['cafe_pos', 'cafe_prod', 'bakery_prod', 'bakery_pos', 'Imap']
    with np.load('manhattan.npz') as data:
        X = data['bakery_pos']
        a = data['bakery_prod']
        Y = data['cafe_pos']
        b = data['cafe_prod']
    return X, a, Y, b

def convert_back_image(img, dim, fame_):
    image_ = img.reshape(dim[0], dim[1], dim[2])
    image_*=255
    PIL_image = Image.fromarray(np.uint8(image_)).convert('RGB')
    PIL_image.show()
    fname='Restults/2/'+str(fame_)+'.png'
    PIL_image.save(fname)

def clustering(X):
    nsa = 1000
    kmeans_ = MiniBatchKMeans(n_clusters=nsa, init_size=nsa).fit(X)
    # print(kmeans_.labels_)
    return kmeans_.cluster_centers_, kmeans_.labels_
def image_to_mat():
    path_1='images/4.jpg'

    img_1 = np.asarray(Image.open(path_1))/255

    path_2 = 'images/10.jpg'

    img_2 = np.asarray(Image.open(path_2))/255

    w_1, h_1, c_1 = img_1.shape
    w_2, h_2, c_2 = img_2.shape
    return [img_1.reshape(-1, 3), [w_1, h_1, c_1]], [img_2.reshape(-1, 3), [w_2, h_2, c_2]]
def color_transfer():
    img_1, img_2 = image_to_mat()
    X, lable_x = clustering(img_1[0])
    Y, lable_y = clustering(img_2[0])
    ones = np.ones((X.shape[0], X.shape[0]))
    #compute the cost matrix
    XX = np.diag(np.diag(X.dot(X.T))).dot(ones)
    YY = ones.dot(np.diag(np.diag(Y.dot(Y.T))))
    XY = X.dot(Y.T)
    C = XX+YY-2*XY
    C = C
    regs = [0.01, 0.1, 0.5, 0.8]
    OT_plan = []
    OT_plan.append(ot.emd(np.ones(X.shape[0]), np.ones(X.shape[0]), C))
    for reg in regs:
        OT_plan.append(ot.sinkhorn(np.ones(X.shape[0]), np.ones(X.shape[0]), C, reg=reg))

    sample_transformed = OT_plan[4].dot(Y)
    img_trans = sample_transformed[lable_x]
    convert_back_image(img_trans, img_1[1], fame_=12)

def Sinkhorn(a, b, cost, epsilon, max_iter=200):
    K = np.exp(-cost/epsilon)
    v = np.ones(b.shape[0])
    for i in range(max_iter):
        print(K.dot(v))
        u = a/K.dot(v)
        v = b/K.T.dot(u)
    return np.diag(u).dot(K).dot(np.diag(v))

def im_show(X, a, Y, b, Flag=False):
    plt.figure(figsize=(10, 10))
    plt.scatter(X[:, 0], X[:, 1], s=10*a, c='r', edgecolors='k', label='bakeries')
    plt.scatter(Y[:, 0], Y[:, 1], s=10*b, c='b', edgecolors='k', label='caffe')
    if Flag:
        optimal_plan1, dis1, optimal_plan2, dis2 = compute_ot_no_rg(X, a, Y, b)
        row, col= optimal_plan2.shape
        for i in range(row):
            for j in range(col):
                plt.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]], c='k', lw=0.15*optimal_plan2[i, j], alpha=0.7)
    legend1 = plt.legend(loc="upper left")
    plt.axis('off')
    plt.title('Bakeries and Caffe', fontsize=10)
    plt.show()
def compute_ot_no_rg(X, a, Y, b):
    row, col = len(X),  len(Y)
    cost1 = np.zeros((row, col))
    cost2 = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            cost1[i, j] = np.linalg.norm(X[i]-Y[j])
            cost2[i, j] = cost1[i, j]**2
    optimal_plan1 = ot.emd(a, b, cost1)
    print(f'this is the OT plan: {optimal_plan1}')
    wass_1 = optimal_plan1*cost1
    dis1=np.sum(wass_1)
    print(f'the first wasserstein distance is: {dis1}')

    optimal_plan2 = ot.emd(a, b, cost2)
    wass_2 = optimal_plan2*cost2
    print(f'this is the OT plan: {optimal_plan2}')
    dis2=np.sqrt(np.sum(wass_2))
    print(f'the second wasserstein distance is: {dis2}')
    return optimal_plan1, dis1, optimal_plan2, dis2
def plot_cluster(cluster_1):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(cluster_1[:, 0]/max(cluster_1[:, 0]), cluster_1[:, 1]/max(cluster_1[:, 1]), cluster_1[:, 2]/max(cluster_1[:, 2]), c=cluster_1/255, s=10, marker='o', alpha=0.6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    ax.set_xlabel('R', fontsize=10)
    ax.set_ylabel('G', fontsize=10)
    ax.set_zlabel('B', fontsize=10)

    # ax.axis('off')
    ax.set_title('cluster', fontsize=10)
    plt.show()
if __name__=='__main__':
    # img_1, img_2 = image_to_mat()
    # cluster_1 = clustering(img_1[0])
    # cluster_2 = clustering(img_2[0])
    # plot_cluster(cluster_2)
    color_transfer()


    # print(img_2[1])
    # convert_back_image(img_2[0], img_2[1])
    # print('Finish')


    # X, a, Y, b = load_data()
    # compute_ot_no_rg(X, a, Y, b)
    # im_show(X, a, Y, b, Flag=False)
    #-----------------------------------------
    # row, col = len(X), len(Y)
    # cost1 = np.zeros((row, col))
    # cost2 = np.zeros((row, col))
    # for i in range(row):
    #     for j in range(col):
    #         cost1[i, j] = np.linalg.norm(X[i] - Y[j])
    #         cost2[i, j] = cost1[i, j] ** 2
    #
    # Sinkhorn(a, b, cost=cost1/cost1.max(), epsilon=0.1, max_iter=200)

