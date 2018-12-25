# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
import sklearn.metrics
from terminaltables import SingleTable

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn")

class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        fmiou = np.nanmean(iu[1:])

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
                "FG mIoU : \t": fmiou
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

def compute_metrics(y_true, y_score, num_classes=3):

    y_true = y_true.flatten()
    labels = list(range(num_classes))

    y_score = y_score.reshape(num_classes, -1)
    y_pred = np.argmax(y_score, axis=0)

    acc          = sklearn.metrics.accuracy_score(y_true, y_pred)
    balanced_acc = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
    precision    = sklearn.metrics.precision_score(y_true, y_pred, labels=labels, average=None)
    recall       = sklearn.metrics.recall_score(y_true, y_pred, labels=labels, average=None)
    dice         = sklearn.metrics.f1_score(y_true, y_pred, labels=labels, average=None)
    cfm          = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=labels)
    jaccard      = sklearn.metrics.jaccard_similarity_score(y_true, y_pred)
    f1           = sklearn.metrics.f1_score(y_true, y_pred, labels=labels, average=None)
    # ce           = sklearn.metrics.log_loss(y_true, y_score[y_pred], labels=labels)

    normalized_cfm = cfm.astype(np.float32) / cfm.sum(axis=1)[:, np.newaxis]

    iou = np.diag(cfm) / (cfm.sum(axis=1) + cfm.sum(axis=0) - np.diag(cfm))
    miou = np.nanmean(iou)
    fiou = np.nanmean(iou[1:])

    metrics = {
        "acc": acc,
        "balanced_acc": balanced_acc,
        "precision": precision,
        "recall": recall,
        "dice_score": dice,
        "jaccard": jaccard,
        "iou": iou,
        "miou": miou,
        "foreground_miou": fiou,
        "f1": f1,
        # "cross_entropy": ce,
        "cfm": cfm,
        "normalized_cfm": normalized_cfm,
        "pr_curve": [None] * num_classes,
        "roc_curve": [None] * num_classes,
        "tp": np.zeros(3),
        "fp": np.zeros(3),
        "tn": np.zeros(3),
        "fn": np.zeros(3),
        "tpr": np.zeros(3),
        "fpr": np.zeros(3),
        "tnr": np.zeros(3),
        "fnr": np.zeros(3),
    }

    for c in range(num_classes):
        y_true_c = y_true == c
        y_score_c = y_score[c]
        metrics["pr_curve"][c]  = sklearn.metrics.precision_recall_curve(y_true_c, y_score_c)
        metrics["roc_curve"][c] = sklearn.metrics.roc_curve(y_true_c, y_score_c)
    
    for c in range(num_classes):
        tp = cfm[c, c]
        fp = np.sum(cfm[:, c]) - tp
        fn = np.sum(cfm[c, :]) - tp
        tn = np.sum(cfm) - tp - fp - fn
        metrics["tp"][c], metrics["fp"][c], metrics["fn"][c], metrics["tn"][c] = tp, fp, fn, tn
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        fnr = fn / (tp + fn)
        tnr = tn / (fp + tn)
        metrics["tpr"][c], metrics["fpr"][c], metrics["fnr"][c], metrics["tnr"][c] = tpr, fpr, fnr, tnr

    return metrics

def mean_metric_table(m, metrics):
    table = []
    for metric_name in metrics:
        table.append([metric_name, str(m[metric_name])])
    table = SingleTable(table, title="Metrics")
    table.inner_heading_row_border = False
    table.inner_column_border = True
    table.inner_row_border = False
    table.outer_border = True
    return table.table

def mean_square_metric_table(m, metrics):
    if len(metrics) != 4: raise Exception("Invalid number of metrics for table.")
    s1 = "\033[1m" + metrics[0] + "\033[0m\n" + str(m[metrics[0]])
    s2 = "\033[1m" + metrics[1] + "\033[0m\n" + str(m[metrics[1]])
    s3 = "\033[1m" + metrics[2] + "\033[0m\n" + str(m[metrics[2]])
    s4 = "\033[1m" + metrics[3] + "\033[0m\n" + str(m[metrics[3]])
    table = [[s1, s2], [s3, s4]]
    table = SingleTable(table)
    table.inner_heading_row_border = False
    table.inner_column_border = True
    table.inner_row_border = True
    table.outer_border = True
    return table.table

def perclass_metric_table(m, metrics, classes):
    table = []
    table.append(["metric"] + classes)
    for metric_name in metrics:
        table.append([metric_name] + [str(s) for s in m[metric_name]])
    table = SingleTable(table, title="Metrics")
    table.inner_heading_row_border = True
    table.inner_column_border = True
    table.inner_row_border = False
    table.outer_border = True
    return table.table

def perclass_square_metric_table(m, metrics, classes):
    if len(metrics) != 4: raise Exception("Invalid number of metrics for table.")
    s = ""
    for c, class_name in enumerate(classes):
        s1 = "\033[1m" + metrics[0] + "\033[0m\n" + str(m[metrics[0]][c])
        s2 = "\033[1m" + metrics[1] + "\033[0m\n" + str(m[metrics[1]][c])
        s3 = "\033[1m" + metrics[2] + "\033[0m\n" + str(m[metrics[2]][c])
        s4 = "\033[1m" + metrics[3] + "\033[0m\n" + str(m[metrics[3]][c])
        table = [[s1, s2], [s3, s4]]
        table = SingleTable(table, title=class_name)
        table.inner_heading_row_border = False
        table.inner_column_border = True
        table.inner_row_border = True
        table.outer_border = True
        s += table.table
        if c != len(classes)-1: s += "\n"
    return s

def cfm_table(m, normalized=True):
    if normalized: cfm = m["normalized_cfm"]
    else: cfm = m["cfm"]
    table = list(cfm)
    table = SingleTable(table, title="CFM")
    table.inner_heading_row_border = False
    table.inner_column_border = False
    table.inner_row_border = False
    table.outer_border = True
    return table.table

def make_plot(x, y, path, title=None, ylabel=None, xlabel=None):
    plt.clf()
    plt.plot(x, y)
    if title is not None: plt.title(title)
    if ylabel is not None: plt.ylabel(ylabel)
    if xlabel is not None: plt.xlabel(xlabel)
    plt.savefig(path)
    plt.close()

class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
