from scipy.io import loadmat as load
import matplotlib.pyplot as plt
import numpy as np

def reformat(samples, labels):
    samples = np.transpose(samples, (3, 0, 1, 2))
    labels = np.array([x[0] for x in labels])
    one_hot_labels = []
    for num in labels:
        one_hot = [0.0] * 10
        if num == 10:
            one_hot[0] = 1.0
        else:
            one_hot[num] = 1.0
        one_hot_labels.append(one_hot)
    
    return samples, labels
 
train = load('C:\\users\\lenovo/train_32x32.mat')
test = load('C:\\users\\lenovo/test_32x32.mat')
 
train_samples = train['X']
train_labels = train['y']
test_samples = test['X']
test_labels = test['y']
 
traindata1, trainlabels = reformat(train_samples, train_labels)
testdata1, testlabels = reformat(test_samples, test_labels)
for i in range(trainlabels.shape[0]):
    if trainlabels[i]==10:
        trainlabels[i]=0
for j in range(testlabels.shape[0]):
    if testlabels[j]==10:
        testlabels[j]=0
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    grayimg = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return grayimg
traindata=np.zeros([traindata1.shape[0],32,32])
for e in range(traindata.shape[0]):
    traindata[e]=rgb2gray(traindata1[e])
traindata=np.reshape(traindata,[traindata.shape[0],32,32,1])
print ("Train Data Shape:", traindata.shape)
print ("Train Data Shape:", trainlabels.shape)
def classify(a):
    n=[]
    for i in range(73257):
        if trainlabels[i]==a:
            n.append(traindata[i])
    n=n[:4000]
    return n
dataset=np.zeros([10,4000,32,32,1])
for i in range(10):
    dataset[i]=classify(i)
print(dataset.shape)


from    omniglot import Omniglot
import  torchvision.transforms as transforms
from    PIL import Image
import  os.path
import  numpy as np


class SvhnNShot:

    def __init__(self, batchsz, n_way, k_shot, k_query, imgsz):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """

        self.resize = imgsz

        # [10, 4000, 32, 32, 1]
        # TODO: can not shuffle here, we must keep training and test set distinct!
        self.x_train, self.x_test = dataset, dataset

        # self.normalization()

        self.batchsz = batchsz
        self.n_cls = 10  
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        assert (k_shot + k_query) <=20

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        print("DB: train", self.x_train.shape, "test", self.x_test.shape)

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),  # current epoch data cached
                               "test": self.load_data_cache(self.datasets["test"])}

    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        # print("before norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

    # print("after norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)

    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 4000, 32, 32, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way # 5 way 1 shot
        querysz = self.k_query * self.n_way
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)

                for j, cur_class in enumerate(selected_cls):

                    selected_img = np.random.choice(20, self.k_shot + self.k_query, False)

                    # meta-training and meta-test
                    x_spt.append(data_pack[cur_class][selected_img[:self.k_shot]])
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot:]])
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, 1, self.resize, self.resize)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, 1, self.resize, self.resize)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

                # append [sptsz, 1, 32, 32] => [b, setsz, 1, 32, 32]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)


            # [b, setsz, 1, 32, 32]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, 1, self.resize, self.resize)
            y_spts = np.array(y_spts).astype(np.int).reshape(self.batchsz, setsz)
            # [b, qrysz, 1, 32, 32]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, 1, self.resize, self.resize)
            y_qrys = np.array(y_qrys).astype(np.int).reshape(self.batchsz, querysz)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch





if __name__ == '__main__':

    import  time
    import  torch
    import  visdom

    # plt.ion()
    viz = visdom.Visdom(env='omniglot_view')

    db = SvhnNShot( batchsz=20, n_way=5, k_shot=5, k_query=15, imgsz=64)

    for i in range(1000):
        x_spt, y_spt, x_qry, y_qry = db.next('train')


        # [b, setsz, h, w, c] => [b, setsz, c, w, h] => [b, setsz, 3c, w, h]
        x_spt = torch.from_numpy(x_spt)
        x_qry = torch.from_numpy(x_qry)
        y_spt = torch.from_numpy(y_spt)
        y_qry = torch.from_numpy(y_qry)
        batchsz, setsz, c, h, w = x_spt.size()


        viz.images(x_spt[0], nrow=5, win='x_spt', opts=dict(title='x_spt'))
        viz.images(x_qry[0], nrow=15, win='x_qry', opts=dict(title='x_qry'))
        viz.text(str(y_spt[0]), win='y_spt', opts=dict(title='y_spt'))
        viz.text(str(y_qry[0]), win='y_qry', opts=dict(title='y_qry'))


        time.sleep(10)

