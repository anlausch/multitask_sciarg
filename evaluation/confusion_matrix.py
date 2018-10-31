import numpy as np
import math
import codecs


def merge_confusion_matrices(conf_mats):
    res_mat = ConfusionMatrix(conf_mats[0].labels)
    for cm in conf_mats:
        res_mat.matrix = np.add(res_mat.matrix, cm.matrix)
    res_mat.compute_all_scores()
    return res_mat


class ConfusionMatrix(object):
    """
    Confusion matrix for evaluating classification tasks.
    """

    def __init__(self, labels=None, predictions=None, gold=None, one_hot_encoding=False, class_indices=False, token_level=False, sequence_lengths=None):
        # rows are true labels, columns predictions
        self.matrix = np.zeros(shape=(len(labels), len(labels)))
        self.labels = labels

        if len(predictions) != len(gold):
            raise ValueError("Predictions and gold labels do not have the same count.")
        if token_level == True:
            for i in range(len(predictions)):
                # todo: here I need to consider the sequence-lengths
                if sequence_lengths is not None:
                    for j in range(sequence_lengths[i]):
                        index_pred = np.argmax(predictions[i][j]) if one_hot_encoding else (predictions[i][j] if class_indices else labels.index(predictions[i][j]))
                        index_gold = np.argmax(gold[i][j]) if one_hot_encoding else (gold[i][j] if class_indices else labels.index(gold[i][j]))
                        self.matrix[index_gold][index_pred] += 1
                else:
                    for j in range(len(predictions[i])):
                        index_pred = np.argmax(predictions[i][j]) if one_hot_encoding else (predictions[i][j] if class_indices else labels.index(predictions[i][j]))
                        index_gold = np.argmax(gold[i][j]) if one_hot_encoding else (gold[i][j] if class_indices else labels.index(gold[i][j]))
                        self.matrix[index_gold][index_pred] += 1
        else:
            for i in range(len(predictions)):
                index_pred = np.argmax(predictions[i]) if one_hot_encoding else (predictions[i] if class_indices else labels.index(predictions[i]))
                index_gold = np.argmax(gold[i]) if one_hot_encoding else (gold[i] if class_indices else labels.index(gold[i]))
                self.matrix[index_gold][index_pred] += 1
        if len(predictions) > 0:
            self.compute_all_scores()

    def compute_all_scores(self, exclude_class=None):
        self.class_performances = {}
        for i in range(len(self.labels)):
            tp = np.float32(self.matrix[i][i])
            fp_plus_tp = np.float32(np.sum(self.matrix, axis=0)[i])
            fn_plus_tp = np.float32(np.sum(self.matrix, axis=1)[i])
            p = 0.0 if math.isnan(tp / fp_plus_tp) else tp / fp_plus_tp
            r = 0.0 if math.isnan(tp / fn_plus_tp) else tp / fn_plus_tp
            try:
                f1 = 0.0 if ((p + r)==0.0 or math.isnan(2 * p * r / (p + r))) else 2 * p * r / (p + r)
            except Exception as e:
                print(e)
            self.class_performances[self.labels[i]] = (p, r, f1)

        if exclude_class is None:
            self.microf1 = np.float32(np.trace(self.matrix)) / np.sum(self.matrix)
            self.macrof1 = float(sum([x[2] for x in self.class_performances.values()])) / float(len(self.labels))
            self.macroP = float(sum([x[0] for x in self.class_performances.values()])) / float(len(self.labels))
            self.macroR = float(sum([x[1] for x in self.class_performances.values()])) / float(len(self.labels))
            self.accuracy = float(sum([self.matrix[i, i] for i in range(len(self.labels))])) / float(np.sum(self.matrix))
        else:
            #self.microf1 = np.float32(np.trace(self.matrix)) / np.sum(self.matrix)
            self.macrof1 = float(sum([x[2] for name, x in self.class_performances.items() if name != exclude_class])) / float(len(self.labels) -1)
            self.macroP = float(sum([x[0] for name, x in self.class_performances.items() if name != exclude_class])) / float(len(self.labels) -1)
            self.macroR = float(sum([x[1] for name, x in self.class_performances.items() if name != exclude_class])) / float(len(self.labels)-1)
            #self.accuracy = float(sum([self.matrix[i, i] for i in range(len(self.labels))])) / float(np.sum(self.matrix))


    def print_results(self):
        for l in self.labels:
            print(l + ": " + str(self.get_class_performance(l)))
        print("Macro avg F1: " + str(self.accuracy))
        print("Macro avg: " + str(self.macrof1))


    def get_class_performance(self, label):
        if label in self.labels:
            return self.class_performances[label]
        else:
            raise Exception("Unknown label")

    def get_all_results(self):
        output = ""
        for l in self.labels:
            output += str(l) + ": " + str(self.get_class_performance(l)) + "\n"

        output += "\nMacro f1: " + str(self.macrof1) + "\n"
        output += "Macro p: " + str(self.macroP) + "\n"
        output += "Macro r: " + str(self.macroR) + "\n"
        return output



def main():
    #cm = ConfusionMatrix(labels=multi_labels, gold=y_golds_single, predictions=y_preds_multi, one_hot_encoding=True, token_level=True, sequence_lengths=sequence_lengths)
    #cm.compute_all_scores(exclude_class="NONE\n")
    #print(cm.get_all_results())
    print("computed")


if __name__ == "__main__":
    main()