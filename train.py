# -*- coding: UTF-8 -*-
import argparse
import numpy
import time
import dgl
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import EvolveGCNO, EvolveGCNH
from utils import Measure
from data_preprocess import data_process,datasave

def plt_loss(loss_train):
    plt.figure(figsize=(20, 10), dpi=100)
    x=[i+1 for i in range(len(loss_train))]
    y=loss_train
    plt.plot(x, y, 'r-o', label="Loss")
    # plt.yticks(range(0, int(max(loss_train)),int(max(loss_train)/10 )))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("epoch", fontdict={'size': 16})
    plt.ylabel("Loss", fontdict={'size': 16})
    plt.title("LOSS_TRAIN", fontdict={'size': 20})
    plt.show()

def train(args, device):
    raw_dir='./data/{}/{}_dataset/'.format(args.dataset, args.dataset)
    processed_dir='./data/{}/processed/'.format(args.dataset)

    g , edge_mask_by_time , num_classes ,interval = data_process(raw_dir=raw_dir,processed_dir=processed_dir,
                                                 reverse_edge=args.reverse_edge,
                                                     dataname=args.dataset,interval=args.interval)
    cached_subgraph = []
    cached_labeled_edge_mask = []
    #使用dgl.node_subgraph会对选取后的节点重新编号
    print("G:",g)
    print("start slicing......")
    # print(edge_mask_by_time[0])
    for i in range(len(edge_mask_by_time)):
        # we add self loop edge when we construct full graph, not here
        edge_subgraph = dgl.add_self_loop(dgl.edge_subgraph(g, edge_mask_by_time[i]))
        # print(edge_subgraph.ndata['feat'][0])
        #根据切片提取子图
        cached_subgraph.append(edge_subgraph.to(device))
        valid_node_mask = edge_subgraph.ndata['label'] >= 0
        #去除节点分类为unknown的节点
        # if i==3:
        #     print(edge_subgraph)
        #     print("edge_subgraph.ndata['label'][0:10]",edge_subgraph.ndata['label'][0:10])
        #     print(valid_node_mask.shape)
        #     print(valid_node_mask[0:10])
        cached_labeled_edge_mask.append(valid_node_mask)
    #cashed_subgraph中是单纯的提取后的切片合集
    #cashed_labeled_node_mask筛选后的类别标签大于0的部分,结果是true or false
    print("g.ndata['feat']:",g.ndata['feat'].shape[1])
    print("start initializing the model.....")
    if args.model == 'EvolveGCN-O':
        model = EvolveGCNO(in_feats=int(g.ndata['feat'].shape[1]),
                           n_hidden=args.n_hidden,
                           num_layers=args.n_layers,
                           n_classes=num_classes)#################
    elif args.model == 'EvolveGCN-H':
        model = EvolveGCNH(in_feats=int(g.ndata['feat'].shape[1]),
                           num_layers=args.n_layers,n_classes=num_classes)
    else:
        return NotImplementedError('Unsupported model {}'.format(args.model))

    print("Initialize multiple GPUs.....")
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     device_sequence = [int(w) for w in args.device_ids.split(',')]
    #     device_ids = [i for i in range(device_sequence[0], device_sequence[1] + 1)]
    #     model = torch.nn.DataParallel(model, device_ids, output_device=args.gpu)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    timestamp=interval
    # split train, valid, test(0-30,31-35,36-48)
    # train/valid/test split follow the paper.
    scale = [float(w) for w in args.scale.split(',')]
    train_max_index = int(scale[0] * timestamp)
    valid_max_index = train_max_index + int (scale[1] * timestamp)
    test_max_index = int(timestamp) - 1#这个切片是在g_list中的从0开始所以最后一片是数量-1
    print(train_max_index,valid_max_index,test_max_index)

    time_window_size = args.n_hist_steps
    loss_class_weight = [float(w) for w in args.loss_class_weight.split(',')]
    loss_class_weight = torch.Tensor(loss_class_weight).to(device)

    train_measure = Measure(num_classes=num_classes, target_class=args.eval_class_id)
    valid_measure = Measure(num_classes=num_classes, target_class=args.eval_class_id)
    test_measure = Measure(num_classes=num_classes, target_class=args.eval_class_id)
    cl_measure=Measure(num_classes=num_classes, target_class=args.eval_class_id)
    test_res_f1 = 0
    Loss_train=[]
    # cl_train=[]
    # cl_valid=[]
    # cl_test=[]

    print("start train......")
    i=args.eval_class_id
    for epoch in range(args.num_epochs):
        model.train()
        for i in range(time_window_size, train_max_index + 1):#i是为了定位时间切片
            g_list = cached_subgraph[i - time_window_size:i + 1]#列表切片多切一个[i-window,i+1]只到i不到i+1 num:i-(i-window)+1
            print("len:",len(g_list))
            #把一次训练的部分提取出来
            predictions = model(g_list)
            print("predictions :",predictions.shape)
            print(g_list[-1])
            # get predictions which has label
            print("cached_labeled_edge_mask[i].shape:",cached_labeled_edge_mask[i].shape)

            predictions = predictions[cached_labeled_edge_mask[i]]
            #预测的结果中label>=0的结果
            print("筛选后的label.shape:",predictions.shape)
            # 这是核心问题，label,cached_labeled_edge_mask[i]

            labels = cached_subgraph[i].ndata['label'][cached_labeled_edge_mask[i]].long()
            print("cached_subgraph.label:",labels.shape)

            #原始的结果中label>=0的结果
            loss = F.cross_entropy(predictions, labels,weight=loss_class_weight)#巨大的改动点原始的函数中无weight参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()#训练模型

            train_measure.append_measures(predictions, labels)#将预测结果实时计算
        Loss_train.append(loss)
        if epoch%20==0 and epoch!=0:
            plt_loss(Loss_train)
        # get each epoch measures during training.
        #到这里对于整个数据集的一次训练结束
        cl_precision, cl_recall, cl_f1 = train_measure.get_total_measure()
        train_measure.update_best_f1(cl_f1, epoch)
        #保存一次训练中最好的一个F1
        # reset measures for next epoch
        train_measure.reset_info()
        #清空计算结果
        print("Train Epoch {} | class {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
              .format(epoch, args.eval_class_id, cl_precision, cl_recall, cl_f1))
        # cl_train.append([epoch,cl_f1])
        # eval每一次训练完都在vaild数据集上看效果
        model.eval()
        for i in range(train_max_index + 1, valid_max_index + 1):
            g_list = cached_subgraph[i - time_window_size:i + 1]
            predictions = model(g_list)
            # get node predictions which has label
            predictions = predictions[cached_labeled_edge_mask[i]]
            labels = cached_subgraph[i].ndata['label'][cached_labeled_edge_mask[i]].long()

            valid_measure.append_measures(predictions, labels)

        # get each epoch measure during eval.
        cl_precision, cl_recall, cl_f1 = valid_measure.get_total_measure()
        valid_measure.update_best_f1(cl_f1, epoch)
        # reset measures for next epoch
        valid_measure.reset_info()
        # cl_valid.append([epoch,cl_f1])
        print("Eval Epoch {} | class {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
              .format(epoch, args.eval_class_id, cl_precision, cl_recall, cl_f1))

        # early stop
        if epoch - valid_measure.target_best_f1_epoch >= args.patience:
            print("Best eval Epoch {}, Cur Epoch {}".format(valid_measure.target_best_f1_epoch, epoch))
            break
        #防止模型过拟合
        # if cur valid f1 score is best, do test
        if epoch == valid_measure.target_best_f1_epoch:#每次训练出都是更好的模型
            print("###################Epoch {} Test###################".format(epoch))
            for i in range(valid_max_index + 1, test_max_index + 1):
                g_list = cached_subgraph[i - time_window_size:i + 1]
                predictions = model(g_list)
                # get predictions which has label
                # print("i",i)
                # print("length",len(cached_labeled_edge_mask))
                predictions = predictions[cached_labeled_edge_mask[i]]
                labels = cached_subgraph[i].ndata['label'][cached_labeled_edge_mask[i]].long()

                test_measure.append_measures(predictions, labels)

            # we get each subgraph measure when testing to match fig 4 in EvolveGCN paper.
            cl_precisions, cl_recalls, cl_f1s = test_measure.get_each_timestamp_measure()
            for index, (sub_p, sub_r, sub_f1) in enumerate(zip(cl_precisions, cl_recalls, cl_f1s)):
                print("  Test | Time {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
                      .format(valid_max_index + index + 1, sub_p, sub_r, sub_f1))##改动点index+2

            # get each epoch measure during test.
            cl_precision, cl_recall, cl_f1 = test_measure.get_total_measure()
            test_measure.update_best_f1(cl_f1, epoch)
            # reset measures for next test
            test_measure.reset_info()

            test_res_f1 = cl_f1
            # cl_test.append([epoch,cl_f1])
            print("  Test | Epoch {} | class {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
                  .format(epoch, args.eval_class_id, cl_precision, cl_recall, cl_f1))

#至此训练完成

    print("Best test f1 is {}, in Epoch {}"
          .format(test_measure.target_best_f1, test_measure.target_best_f1_epoch))
    if test_measure.target_best_f1_epoch != valid_measure.target_best_f1_epoch:
        print("The Epoch get best Valid measure not get the best Test measure, "
              "please checkout the test result in Epoch {}, which f1 is {}"
              .format(valid_measure.target_best_f1_epoch, test_res_f1))

    for i in range(time_window_size, test_max_index + 1):
        g_list = cached_subgraph[i - time_window_size:i + 1]
        predictions = model(g_list)
        # get predictions which has label
        # print("i",i)
        # print("length",len(cached_labeled_edge_mask))
        predictions = predictions[cached_labeled_edge_mask[i]]
        labels = cached_subgraph[i].ndata['label'][cached_labeled_edge_mask[i]].long()
        cl_measure.append_measures(predictions, labels)
    cl_precisions, cl_recalls, cl_f1s = cl_measure.get_each_timestamp_measure()
    for index, (sub_p, sub_r, sub_f1) in enumerate(zip(cl_precisions, cl_recalls, cl_f1s)):
        print("  Test | Time {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
              .format( index + 1, sub_p, sub_r, sub_f1))  ##改动点index+2

    datasave(Loss_train, cl_f1s, processed_dir)
    cl_precision, cl_recall, cl_f1 = cl_measure.get_total_measure()
    print("  Test | Epoch {} | class {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
          .format("final", args.eval_class_id, cl_precision, cl_recall, cl_f1))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("EvolveGCN")
    argparser.add_argument('--model', type=str, default='EvolveGCN-O',
                           help='We can choose EvolveGCN-O or EvolveGCN-H,'
                                'but the EvolveGCN-H performance on Elliptic dataset is not good.')
    argparser.add_argument("--dataset", type=str, default="elliptic",
                        help="dataset selection wikipedia/reddit/elliptic")
    # argparser.add_argument('--raw-dir', type=str,
    #                        default='./data/Elliptic/elliptic_bitcoin_dataset/',
    #                        help="Dir after unzip downloaded dataset, which contains 3 csv files.")
    # argparser.add_argument('--processed-dir', type=str,
    #                        default='./data/Elliptic/processed/',
    #                        help="Dir to store processed raw data.")
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training.")
    argparser.add_argument('--device_ids', type=str, default='0,1',
                           help="device_ids for use Gpus")
    argparser.add_argument('--interval', type=int, default='110',
                           help="The number of time slices you want to set")
    argparser.add_argument('--reverse_edge', type=bool, default='True',
                           help="True for undirected graph False for directed graph")

    argparser.add_argument('--scale', type=str, default='0.7,0.1',
                           help="The proportion of input train and valid")

    argparser.add_argument('--num-epochs', type=int, default=1000)
    argparser.add_argument('--n-hidden', type=int, default=256)
    argparser.add_argument('--n-layers', type=int, default=2)
    argparser.add_argument('--n-hist-steps', type=int, default=5,
                           help="If it is set to 5, it means in the first batch,"
                                "we use historical data of 0-4 to predict the data of time 5.")
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--loss-class-weight', type=str, default='0.25,0.75',
                           help='Weight for loss function. Follow the official code,'
                                'we need to change it to 0.25, 0.75 when use EvolveGCN-H')
    argparser.add_argument('--eval-class-id', type=int, default=1,
                           help="Class type to eval. On Elliptic, type 1(illicit) is the main interest.")
    argparser.add_argument('--patience', type=int, default=150,
                           help="Patience for early stopping.")

    args = argparser.parse_args()

    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
        print("we can use", torch.cuda.device_count(), "GPUs!")
    else:
        device = torch.device('cpu')

    print("use :",device)
    start_time = time.perf_counter()
    train(args, device)
    print("train time is: {}".format(time.perf_counter() - start_time))
