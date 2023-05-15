import math

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sn  #画图模块



class Performance:
    """
    定义一个类，用来分类器的性能度量
    """
    def __init__(self, true_labels, pre_labels):
        """
        :param labels:数组类型，真实的标签
        :param scores:数组类型，分类器的得分
        """
        self.true_labels = true_labels
        self.pre_labels = pre_labels
        self.db = self.get_db()
        self.TP, self.FP, self.FN, self.TN = self.get_confusion_matrix()


    def get_db(self):
        db = []
        for i in range(len(self.true_labels)):
            db.append([self.true_labels[i], self.pre_labels[i]])
        db = sorted(db, key=lambda x: x[1], reverse=True)
        return db


    def get_confusion_matrix(self):
        tp, fp, fn, tn = 0., 0., 0., 0.
        for i in range(len(self.true_labels)):
            if self.true_labels[i] == 1 and self.pre_labels[i] == 1:
                tp += 1
            elif self.true_labels[i] == 0 and self.pre_labels[i] == 1:
                fp += 1
            elif self.true_labels[i] == 1 and self.pre_labels[i] == 0:
                fn += 1
            else:
                tn += 1
        return [tp, fp, fn, tn]


    def accuracy(self):
        return (self.TP + self.TN) / (self.TP + self.FN + self.FP + self.TN)


    def presision(self):
        return self.TP / (self.TP + self.FP)


    def recall(self):
        return self.TP / (self.TP + self.FN)

    def specificity(self):
        return self.TN / (self.TN + self.FP)


    def f1_score(self):
        return f1_score(self.true_labels, self.pre_labels, average='binary')


    def roc_plot(self):
        fpr, tpr, threshold = roc_curve(self.true_labels, self.pre_labels)  ###计算真正率和假正率
        roc_auc = auc(fpr, tpr)  ###计算auc的值
        lw = 2
        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (AUC = %0.4f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Roc Curve')
        plt.legend(loc="lower right")
        plt.show()



    def plot_matrix(self):
        cm = confusion_matrix(self.true_labels, self.pre_labels)#混淆矩阵
        ax = sn.heatmap(cm,annot=True,fmt='g',xticklabels=['HC Predicted','PD Predicted'],yticklabels=['HC Actual','PD Actual'])
        # ax.set_title('Confusion Matrix') #标题
        # ax.set_xlabel('predict') #x轴
        # ax.set_ylabel('true') #y轴
        plt.show()

    def mcc(self):
        return (self.TP*self.TN - self.FP*self.FN)/ math.sqrt((self.TP+self.FP) * (self.TP+self.FN) * (self.TN+self.FP) * (self.TN+self.FN))





