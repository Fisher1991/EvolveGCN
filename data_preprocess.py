import os
import sys
import pandas
import numpy
import torch
import dgl

def datasave(Loss_train, cl_f1s, processed_dir):
    Loss_train = numpy.array(Loss_train)
    cl_f1s=numpy.array(cl_f1s)
    numpy.save(os.path.join(processed_dir, 'Loss_train.npy'), Loss_train)
    numpy.save(os.path.join(processed_dir, 'cl_f1s.npy'), cl_f1s)

def data_process(raw_dir, processed_dir, reverse_edge, dataname,interval):
    if dataname=='wikipedia':
        wikipedia_dataset = WikipediaDataset(raw_dir=raw_dir, processed_dir=processed_dir,
                                             reverse_edge=reverse_edge,interval=interval)
        g , src_dst_time_path  = wikipedia_dataset.process()
        print("g.ndata:",g.ndata['label'][0:10])
        num_class = wikipedia_dataset.num_classes

    elif dataname=='reddit':
        Reddit_dataset=RedditDataset(raw_dir=raw_dir, processed_dir=processed_dir,
                                     reverse_edge=reverse_edge,interval=interval)
        g , src_dst_time_path = Reddit_dataset.process()
        num_class = Reddit_dataset.num_classes

    elif dataname=='elliptic':
        elliptic_dataset = EllipticDataset(raw_dir=raw_dir, processed_dir=processed_dir,
                                           self_loop=True, reverse_edge=reverse_edge)
        g , src_dst_time_path= elliptic_dataset.process()
        num_class = elliptic_dataset.num_classes

    else:
        print("Dataset does not exist!")
        sys.exit(0)
    # print(g)

    #########
    src_dst_time = torch.IntTensor(numpy.load(os.path.join(src_dst_time_path)))
    edge_mask_by_time = []
    start_time = int(torch.min(src_dst_time[:, -1]))
    print("start_time:",start_time)
    end_time = int(torch.max(src_dst_time[:, -1]))
    # 设定开始的时间切片，结束的时间切片
    # 按边上的时间戳对图进行切片
    print("row:",src_dst_time.shape[0])
    for i in range(start_time, end_time + 1):
        # for j in range(src_dst_time.shape[0]):
        mask=src_dst_time[:, -1] == i
        edge_mask=list(numpy.where(numpy.array(mask)==True)[0])
        edge_mask_by_time.append(edge_mask)
        # if i==10:
        #     break
        # 剥离出不同时间切片的id_features序列
    print("prepare slicing finished")
    # print(edge_mask_by_time)
    return g , edge_mask_by_time ,num_class,end_time - start_time + 1


class EllipticDataset:
    def __init__(self, raw_dir, processed_dir, self_loop=False, reverse_edge=True):
        self.raw_dir = raw_dir
        self.processd_dir = processed_dir
        self.self_loop = self_loop
        self.reverse_edge = reverse_edge

    def process(self):
        if not os.path.exists('./data/elliptic/processed/elliptic_{}.bin'.format(self.reverse_edge)):
            self.process_raw_data()
            id_time_features = torch.Tensor(numpy.load(os.path.join(self.processd_dir, 'id_time_features.npy')))
            #id_time_features：Id , time , feature
            id_label = torch.IntTensor(numpy.load(os.path.join(self.processd_dir, 'id_label.npy')))
            #id_label: Id , class
            src_dst_time = torch.IntTensor(numpy.load(os.path.join(self.processd_dir, 'src_dst_time.npy')))
            #src_dst_time: srcId , dstId , edge_time
            src = src_dst_time[:, 0]
            dst = src_dst_time[:, 1]
            # id_label[:, 0] is used to add self loop
            if self.self_loop:
                if self.reverse_edge:
                    g = dgl.graph(data=(torch.cat((src, dst, id_label[:, 0])), torch.cat((dst, src, id_label[:, 0]))),
                                  num_nodes=id_label.shape[0])
                    #对于无向的图，用户需要为每条边都创建两个方向的边,DGL支持使用 32 位或 64 位的整数作为节点ID和边ID,id_label相当于添加自循环
                    g.edata['timestamp'] = torch.cat((src_dst_time[:, 2], src_dst_time[:, 2], id_time_features[:, 1].int()))
                    #DGLGraph 对象的节点和边可具有多个用户定义的、可命名的特征，以储存图的节点和边的属性。 通过 ndata 和 edata 接口可访问这些特征。
                else:
                    g = dgl.graph(data=(torch.cat((src, id_label[:, 0])), torch.cat((dst, id_label[:, 0]))),
                                  num_nodes=id_label.shape[0])
                    g.edata['timestamp'] = torch.cat((src_dst_time[:, 2], id_time_features[:, 1].int()))
            else:
                if self.reverse_edge:
                    g = dgl.graph(data=(torch.cat((src, dst)), torch.cat((dst, src))),
                                  num_nodes=id_label.shape[0])
                    g.edata['timestamp'] = torch.cat((src_dst_time[:, 2], src_dst_time[:, 2]))
                else:
                    g = dgl.graph(data=(src, dst),
                                  num_nodes=id_label.shape[0])
                    g.edata['timestamp'] = src_dst_time[:, 2]

            time_features = id_time_features[:, 1:]
            label = id_label[:, 1]
            g.ndata['label'] = label
            g.ndata['feat'] = time_features
            # print(time_features[0])
            #用ndata添加节点特征
            # used to construct time-based sub-graph.
            print("Gragh processing is complete.")
            dgl.save_graphs('./data/elliptic/processed/elliptic_{}.bin'.format(self.reverse_edge), [g])

        else:
            print("Data is exist directly loaded.")
            gs, _ = dgl.load_graphs('./data/elliptic/processed/elliptic_{}.bin'.format(self.reverse_edge))
            g = gs[0]
        return g, os.path.join(self.processd_dir, 'src_dst_time.npy')

    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return 2

    def process_raw_data(self):
        r"""

        Description
        -----------
        Preprocess Elliptic dataset like the EvolveGCN official instruction:
        github.com/IBM/EvolveGCN/blob/master/elliptic_construction.md
        The main purpose is to convert original idx to contiguous idx start at 0.
        """
        processed_dir=self.processd_dir
        raw_dir=self.raw_dir

        oid_nid_path = os.path.join(processed_dir, 'oid_nid.npy')
        id_label_path = os.path.join(processed_dir, 'id_label.npy')
        id_time_features_path = os.path.join(processed_dir, 'id_time_features.npy')
        src_dst_time_path = os.path.join(processed_dir, 'src_dst_time.npy')
        #创立了四个路径用于储存处理后的数据，这一步只是生成路径
        if os.path.exists(oid_nid_path) and os.path.exists(id_label_path) and \
                os.path.exists(id_time_features_path) and os.path.exists(src_dst_time_path):
            print("The preprocessed data already exists, skip the preprocess stage!")
            return
        print("starting process raw data in {}".format(raw_dir))
        id_label = pandas.read_csv(os.path.join(raw_dir, 'elliptic_txs_classes.csv'))
        src_dst = pandas.read_csv(os.path.join(raw_dir, 'elliptic_txs_edgelist.csv'))
        # elliptic_txs_features.csv has no header, and it has the same order idx with elliptic_txs_classes.csv
        id_time_features = pandas.read_csv(os.path.join(raw_dir, 'elliptic_txs_features.csv'), header=None)

        # get oldId_newId
        oid_nid = id_label.loc[:, ['txId']]#读取pandas中某一列
        oid_nid = oid_nid.rename(columns={'txId': 'originalId'})#将名称从‘txId’改成‘originalId’
        oid_nid.insert(1, 'newId', range(len(oid_nid)))
        #在第1列插入‘newId’,数据是按序列生成的
        # map classes unknown,1,2 to -1,1,0 and construct id_label. type 1 means illicit.
        id_label = pandas.concat(
            [oid_nid['newId'], id_label['class'].map({'unknown': -1.0, '1': 1.0, '2': 0.0})], axis=1)
        #横着拼axis=1,将class设置为三种结果 ‘-1’ ， ‘1’ ，‘0’
        # replace originalId to newId.
        # Attention: the timestamp in features start at 1.
        id_time_features[0] = oid_nid['newId']

        # construct originalId2newId dict
        #设置了新节点编号和旧节点编号的键值对
        oid_nid_dict = oid_nid.set_index(['originalId'])['newId'].to_dict()
        # construct newId2timestamp dict
        # 设置了新节点编号和时间切片的键值对
        nid_time_dict = id_time_features.set_index([0])[1].to_dict()

        # Map id in edgelist to newId, and add a timestamp to each edge.
        # Attention: From the EvolveGCN official instruction, the timestamp with edgelist start at 0, rather than 1.
        # see: github.com/IBM/EvolveGCN/blob/master/elliptic_construction.md
        # Here we dose not follow the official instruction, which means timestamp with edgelist also start at 1.
        # In EvolveGCN example, the edge timestamp will not be used.
        #
        # Note: in the dataset, src and dst node has the same timestamp, so it's easy to set edge's timestamp.
        new_src = src_dst['txId1'].map(oid_nid_dict).rename('newSrc')
        new_dst = src_dst['txId2'].map(oid_nid_dict).rename('newDst')
        #对edge的源节点和终节点完成重新编号
        edge_time = new_src.map(nid_time_dict).rename('timestamp')
        src_dst_time = pandas.concat([new_src, new_dst, edge_time], axis=1)

        # save oid_nid, id_label, id_time_features, src_dst_time to disk. we can convert them to numpy.
        # oid_nid: type int.  id_label: type int.  id_time_features: type float.  src_dst_time: type int.
        oid_nid = oid_nid.to_numpy(dtype=int)#第一列是新节点编号，第二列是原节点编号
        id_label = id_label.to_numpy(dtype=int)#更新后的class
        id_time_features = id_time_features.to_numpy(dtype=float)#更新后的features
        src_dst_time = src_dst_time.to_numpy(dtype=int)#更新后的edge_list

        numpy.save(oid_nid_path, oid_nid)
        numpy.save(id_label_path, id_label)
        numpy.save(id_time_features_path, id_time_features)
        numpy.save(src_dst_time_path, src_dst_time)
        print("Process Elliptic raw data done, data has saved into {}".format(processed_dir))
        #将转换好的数据储存起来


class RedditDataset:

    def __init__(self, raw_dir, processed_dir,  reverse_edge=True,interval=1):
        self.raw_dir = raw_dir
        self.processd_dir = processed_dir
        self.reverse_edge = reverse_edge
        self.interval=interval

    def process(self):
        # if not os.path.exists('./data/reddit/processed/reddit_{}.bin'.format(self.reverse_edge)):
            self.process_raw_data()
            id_features = torch.Tensor(numpy.load(os.path.join(self.processd_dir, 'id_features.npy')))
            # id_time_features：Id , time , feature
            id_label = torch.IntTensor(numpy.load(os.path.join(self.processd_dir, 'id_label.npy')))
            # id_label: Id , class
            src_dst_time = numpy.load(os.path.join(self.processd_dir, 'src_dst_time.npy'))
            # src_dst_time: srcId , dstId , edge_time
            interval = self.interval
            if interval > 1:
                print("Re-slicing by the number of time slices")
                max_time_stamp = src_dst_time[:, -1].max()
                min_time_stamp = src_dst_time[:, -1].min()
                time_slice = int((max_time_stamp - min_time_stamp + 1) / interval)
                assert (time_slice != 0), "Please check if interval is correctly set!"
                src_dst_time.ts = (src_dst_time.ts - min_time_stamp) / time_slice
            src_dst_time=torch.IntTensor(src_dst_time)
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
        #     dgl.save_graphs('./data/reddit/processed/reddit_{}.bin'.format(self.reverse_edge), [g])
        # else:
        #     print("Data is exist directly loaded.")
        #     gs, _ = dgl.load_graphs('./data/reddit/processed/reddit_{}.bin'.format(self.reverse_edge))
        #     g = gs[0]
            return g , os.path.join(self.processd_dir, 'src_dst_time.npy')

    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return 2

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
        PATH = os.path.join(raw_dir, 'reddit.csv')
        id_label_path = os.path.join(processed_dir, 'id_label.npy')
        id_features_path = os.path.join(processed_dir, 'id_features.npy')
        src_dst_time_path = os.path.join(processed_dir, 'src_dst_time.npy')
        # 创立了四个路径用于储存处理后的数据，这一步只是生成路径
        if os.path.exists(id_label_path) and \
                os.path.exists(id_features_path) and os.path.exists(src_dst_time_path):
            print("The preprocessed data already exists, skip the preprocess stage!")
            return
        print("starting process raw data in {}".format(raw_dir))
        # 确定了需要得到的文件结果，开始处理文件

        u_list, i_list, ts_list, label_list = [], [], [], []
        feat_l = []
        idx_list = []

        with open(PATH) as f:
            s = next(f)
            for idx, line in enumerate(f):
                e = line.strip().split(',')
                u = int(e[0])  # user_id
                i = int(e[1])  # item_id
                ts = int(float(e[2]))  # timestamp
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
            id_label = id_label.append({'idx': i, 'label': -1}, ignore_index=True)
            # print(zero_shape)
            feat=numpy.zeros((1,zero_shape))

            div_term=numpy.exp(numpy.arange(0,zero_shape,2)* -(numpy.log(10000.0)/zero_shape))
            feat[0, 0::2] = numpy.sin(i * div_term)
            feat[0, 1::2] = numpy.cos(i * div_term[0:int(zero_shape / 2)])
            feat[0, 0] = i
            # print(type(id_features),type(feat))
            # print(id_features.shape,feat.shape)
            id_features=numpy.append(id_features,feat,axis=0)

        #对数据重新切片，切片后的数据集从0开始

        id_label = id_label.to_numpy(dtype=int)#更新后的class
        # id_features = id_features.to_numpy(dtype=float)#更新后的features
        src_dst_time = srt_dst_time.to_numpy(dtype=int)#更新后的edge_list

        numpy.save(id_label_path, id_label)
        numpy.save(id_features_path, id_features)
        numpy.save(src_dst_time_path, src_dst_time)
        print("Process raw data done, data has saved into {}".format(processed_dir))
        # 将转换好的数据储存起来


class WikipediaDataset:
    def __init__(self, raw_dir, processed_dir,  reverse_edge=True,interval=1):
        self.raw_dir = raw_dir
        self.processd_dir = processed_dir
        self.reverse_edge = reverse_edge
        self.interval=interval

    def process(self):
        # if not os.path.exists('./data/wikipedia/processed/wikipedia_{}.bin'.format(self.reverse_edge)):
            self.process_raw_data()
            id_features = torch.Tensor(numpy.load(os.path.join(self.processd_dir, 'id_features.npy')))
            # id_time_features：Id , time , feature
            id_label = torch.IntTensor(numpy.load(os.path.join(self.processd_dir, 'id_label.npy')))
            # id_label: Id , class
            src_dst_time = numpy.load(os.path.join(self.processd_dir, 'src_dst_time.npy'))
            # src_dst_time: srcId , dstId , edge_time
            interval = self.interval
            if interval > 1:
                print("Re-slicing by the number of time slices")
                max_time_stamp = src_dst_time[:, -1].max()
                min_time_stamp = src_dst_time[:, -1].min()
                time_slice = int((max_time_stamp - min_time_stamp + 1) / interval)
                assert (time_slice != 0), "Please check if interval is correctly set!"
                src_dst_time.ts = (src_dst_time.ts - min_time_stamp) / time_slice
            src_dst_time=torch.IntTensor(src_dst_time)

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
        #     dgl.save_graphs('./data/wikipedia/processed/wikipedia_{}.bin'.format(self.reverse_edge), [g])
        # else:
        #     print("Data is exist directly loaded.")
        #     gs, _ = dgl.load_graphs('./data/wikipedia/processed/wikipedia_{}.bin'.format(self.reverse_edge))
        #     g = gs[0]
            return g , os.path.join(self.processd_dir, 'src_dst_time.npy')

    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return 2

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
        PATH = os.path.join(raw_dir, 'wikipedia.csv')
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
                ts = int(float(e[2]))  # timestamp
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
            id_label = id_label.append({'idx': i, 'label': -1}, ignore_index=True)
            # print(zero_shape)
            feat=numpy.zeros((1,zero_shape))

            div_term=numpy.exp(numpy.arange(0,zero_shape,2)* -(numpy.log(10000.0)/zero_shape))
            feat[0, 0::2] = numpy.sin(i * div_term)
            feat[0, 1::2] = numpy.cos(i * div_term[0:int(zero_shape / 2)])
            feat[0, 0] = i
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


