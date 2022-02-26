def calculate_measure(tp, fn, fp):
    # avoid nan
    if tp == 0:
        return 0, 0, 0

    p = tp * 1.0 / (tp + fp)
    r = tp * 1.0 / (tp + fn)
    if (p + r) > 0:
        f1 = 2.0 * (p * r) / (p + r)
    else:
        f1 = 0
    return p, r, f1


class Measure(object):
    def __init__(self, num_classes, target_class):
        """

        Args:
            num_classes: number of classes.
            target_class: target class we focus on, used to print info and do early stopping.
        """
        self.num_classes = num_classes
        self.target_class = target_class
        self.true_positives = {}
        self.false_positives = {}
        self.false_negatives = {}
        self.target_best_f1 = 0.0
        self.target_best_f1_epoch = 0
        self.reset_info()

    def reset_info(self):
        """
            reset info after each epoch.
        """
        self.true_positives = {cur_class: [] for cur_class in range(self.num_classes)}
        self.false_positives = {cur_class: [] for cur_class in range(self.num_classes)}
        self.false_negatives = {cur_class: [] for cur_class in range(self.num_classes)}

    def append_measures(self, predictions, labels):
        predicted_classes = predictions.argmax(dim=1)#返回每行最大值的索引prediction是一个向量结果
        for cl in range(self.num_classes):
            cl_indices = (labels == cl)
            pos = (predicted_classes == cl)
            hits = (predicted_classes[cl_indices] == labels[cl_indices])

            tp = hits.sum()#对应起来相等的结果数量
            fn = hits.size(0) - tp#没预测到
            fp = pos.sum() - tp#预测错误

            self.true_positives[cl].append(tp.cpu())
            self.false_negatives[cl].append(fn.cpu())
            self.false_positives[cl].append(fp.cpu())

    def get_each_timestamp_measure(self):
        precisions = []
        recalls = []
        f1s = []
        for i in range(len(self.true_positives[self.target_class])):
            tp = self.true_positives[self.target_class][i]
            fn = self.false_negatives[self.target_class][i]
            fp = self.false_positives[self.target_class][i]

            p, r, f1 = calculate_measure(tp, fn, fp)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
        #对每一个batch计算一次数值
        return precisions, recalls, f1s

    def get_total_measure(self):
        tp = sum(self.true_positives[self.target_class])
        fn = sum(self.false_negatives[self.target_class])
        fp = sum(self.false_positives[self.target_class])

        p, r, f1 = calculate_measure(tp, fn, fp)
        #对整体做计算
        return p, r, f1

    def update_best_f1(self, cur_f1, cur_epoch):
        if cur_f1 > self.target_best_f1 and cur_f1!=1: #######重点改动
            self.target_best_f1 = cur_f1
            self.target_best_f1_epoch = cur_epoch
