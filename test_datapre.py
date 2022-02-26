import os
import pandas
import numpy
import torch
import dgl



def data_process(raw_dir, processed_dir, reverse_edge,interval):
    dataset = Dataset(raw_dir, processed_dir, reverse_edge,interval)
    g , src_dst_time_path =dataset.process()
    num_classes = dataset.num_classes
    id_features = torch.Tensor(numpy.load(os.path.join(processed_dir, 'id_features.npy')))
    # id_time_features：Id , time , feature
    id_label = torch.IntTensor(numpy.load(os.path.join(processed_dir, 'id_label.npy')))
    # id_label: Id , class
    src_dst_time = torch.IntTensor(numpy.load(os.path.join(processed_dir, 'src_dst_time.npy')))
    print(id_features)
    print(id_label)
    print(src_dst_time)
    return g,src_dst_time_path,num_classes

class Dataset:

    def __init__(self, raw_dir, processed_dir,  reverse_edge=True,interval=1):
        self.raw_dir = raw_dir
        self.processd_dir = processed_dir
        self.reverse_edge = reverse_edge
        self.interval=interval

    def process(self):
        if not os.path.exists('./data/test_dataset/processed/dataset_{}.bin.bin'.format(self.reverse_edge)):
            self.process_raw_data()
            id_features = torch.Tensor(numpy.load(os.path.join(self.processd_dir, 'id_features.npy')))
            # id_time_features：Id , time , feature
            id_label = torch.IntTensor(numpy.load(os.path.join(self.processd_dir, 'id_label.npy')))
            # id_label: Id , class
            src_dst_time = torch.IntTensor(numpy.load(os.path.join(self.processd_dir, 'src_dst_time.npy')))
            # src_dst_time: srcId , dstId , edge_time
            src = src_dst_time[:, 1]
            dst = src_dst_time[:, 2]
            # id_label[:, 0] is used to add self loop
            if self.reverse_edge:
                g = dgl.graph(data=(torch.cat((src, dst)), torch.cat((dst, src))),
                              num_nodes=id_label.shape[0])
                # 对于无向的图，用户需要为每条边都创建两个方向的边,DGL支持使用 32 位或 64 位的整数作为节点ID和边ID,
                g.edata['timestamp'] = torch.cat((src_dst_time[:, 3], src_dst_time[:, 3]))
                # DGLGraph 对象的节点和边可具有多个用户定义的、可命名的特征，以储存图的节点和边的属性。 通过 ndata 和 edata 接口可访问这些特征。
            else:
                g = dgl.graph(data=(src, dst),
                              num_nodes=id_label.shape[0])
                g.edata['timestamp'] = src_dst_time[:, 3]

            time_features = id_features[:, 1:]
            label = id_label[:, 1]
            g.ndata['label'] = label
            g.ndata['feat'] = time_features
            # 用ndata添加节点特征
            # used to construct time-based sub-graph.
            print("Gragh processing is complete.")
            dgl.save_graphs('./data/test_dataset/processed/dataset_{}.bin'.format(self.reverse_edge), [g])
        else:
            print("Data is exist directly loaded.")
            gs, _ = dgl.load_graphs('./data/test_dataset/processed/dataset_{}.bin'.format(self.reverse_edge))
            g = gs[0]
        return g , os.path.join(self.processd_dir, 'src_dst_time.npy')

    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return 3

    def process_raw_data(self):
        r"""
        Description
        -----------
        Preprocess Elliptic dataset like the EvolveGCN official instruction:
        github.com/IBM/EvolveGCN/blob/master/elliptic_construction.md
        The main purpose is to convert original idx to contiguous idx start at 0.
        """
        processed_dir = self.processd_dir
        raw_dir = self.raw_dir
        print("Start Process Data ...")
        PATH = os.path.join(raw_dir, 'testdataset.csv')
        id_label_path = os.path.join(processed_dir, 'id_label.npy')
        id_features_path = os.path.join(processed_dir, 'id_features.npy')
        src_dst_time_path = os.path.join(processed_dir, 'src_dst_time.npy')
        # 创立了四个路径用于储存处理后的数据，这一步只是生成路径
        if os.path.exists(id_label_path) and \
                os.path.exists(id_features_path) and os.path.exists(src_dst_time_path):
            print("The preprocessed data already exists, skip the preprocess stage!")
            return
        print("starting process raw data in {}".format(raw_dir))
        #确定了需要得到的文件结果，开始处理文件

        u_list, i_list, ts_list, label_list = [], [], [], []
        feat_l = []
        idx_list = []

        with open(PATH) as f:
            s = next(f)
            for idx, line in enumerate(f):
                e = line.strip().split(',')
                u = int(e[0])  # user_id
                i = int(e[1])  # item_id
                ts = int(e[2])  # timestamp
                label = float(e[3])  # int(e[3]) state_label
                feat = [float(x) for x in e[4:]]  # features
                feat.insert(0,idx)
                u_list.append(u)
                i_list.append(i)
                ts_list.append(ts)
                label_list.append(label)
                idx_list.append(idx)
                feat_l.append(feat)
        srt_dst_time=pandas.DataFrame({'idx': idx_list,'u': u_list,'i': i_list,'ts': ts_list})
        id_label=pandas.DataFrame({'idx':idx_list,'label': label_list})
        id_features=numpy.array(feat_l)
        #因为是二部图所以需要讲帖子节点的序号改变

        assert (srt_dst_time.u.max() - srt_dst_time.u.min() + 1 == len(srt_dst_time.u.unique()))
        assert (srt_dst_time.i.max() - srt_dst_time.i.min() + 1 == len(srt_dst_time.i.unique()))
        # 最大编号和最小编号中没有缺失的节点
        upper_u = srt_dst_time.u.max() + 1
        new_i = srt_dst_time.i + upper_u
        srt_dst_time.i = new_i
        zero_shape = id_features.shape[1]
        for i in range(srt_dst_time.i.min(), srt_dst_time.i.max() + 1):
            id_label = id_label.append({'idx': i, 'label': 2}, ignore_index=True)
            # print(zero_shape)
            feat=numpy.zeros((1,zero_shape))
            feat[0,0]=i
            # print(type(id_features),type(feat))
            # print(id_features.shape,feat.shape)
            id_features=numpy.append(id_features,feat,axis=0)

        #对数据重新切片，切片后的数据集从0开始
        interval=self.interval
        if interval > 1 :
            print("Re-slicing by the number of time slices")
            max_time_stamp=srt_dst_time.ts.max()
            min_time_stamp=srt_dst_time.ts.min()
            time_slice = int((max_time_stamp - min_time_stamp + 1) / interval)
            assert (time_slice != 0), "Please check if interval is correctly set!"
            srt_dst_time.ts = (srt_dst_time.ts-min_time_stamp) / time_slice

        id_label = id_label.to_numpy(dtype=int)#更新后的class
        # id_features = id_features.to_numpy(dtype=float)#更新后的features
        src_dst_time = srt_dst_time.to_numpy(dtype=int)#更新后的edge_list

        numpy.save(id_label_path, id_label)
        numpy.save(id_features_path, id_features)
        numpy.save(src_dst_time_path, src_dst_time)
        print("Process raw data done, data has saved into {}".format(processed_dir))
        # 将转换好的数据储存起来


raw_dir='./data/test_dataset/testset/'
processed_dir='./data/test_dataset/processed/'
g, edge_mask_by_time, num_classes = data_process(raw_dir=raw_dir, processed_dir=processed_dir,
                                                 reverse_edge=True, interval=3)
print(g.edges())
print(g)
print(g.ndata['feat'])
