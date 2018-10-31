from nltk.tag import hmm
import time
import load_conll
import numpy as np
from lstm.evaluation.confusion_matrix import ConfusionMatrix

def transform_classes_to_binary(y, labels_list):
    labels_list = np.array(labels_list)
    y = np.array([np.array([np.array([1 if np.where(labels_list == label)[0] == i else 0 for i in range(0, len(labels_list))]) for label in sentence]) for sentence in y])
    return y

print("HMM script started")
start = time.time()
x_train_dev, y_arg_train_dev, y_rhet_train_dev, y_aspect_train_dev, y_summary_train_dev, y_citation_train_dev = load_conll.load_data_multiple(path="./../annotations_conll_final_splitted/train_dev/")
x_test, y_arg_test, y_rhet_test, y_aspect_test, y_summary_test, y_citation_test = load_conll.load_data_multiple(path="./../annotations_conll_final_splitted/test/")
print("Data loaded")

# provide token-label tuples to the trainer
xy_train_dev = [list(zip(x_sent, y_sent)) for (x_sent, y_sent) in list(zip(x_train_dev, y_citation_train_dev))]

# Setup a trainer with default(None) values
# And train with the data
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(xy_train_dev)
#x_test = [list(x_sent) for x_sent in x_test]
xy_pred = [tagger.tag(list(x_sent)) for x_sent in x_test]
#xy_prd = tagger.tag(x_test)
y_pred = [[y_token for x_token, y_token in sentence] for sentence in xy_pred]

labels = list(set([lab for sublist in y_citation_train_dev for lab in sublist]))
y_citation_test = transform_classes_to_binary(y_citation_test, labels)
y_pred = transform_classes_to_binary(y_pred, labels)

confusion_matrix = ConfusionMatrix(labels=labels, gold=y_citation_test, predictions=y_pred, token_level=True, one_hot_encoding=True)
confusion_matrix.compute_all_scores(exclude_class="NONE\n")
print(str(confusion_matrix.get_all_results()))

print("Total training time: " + str(time.time() - start))

