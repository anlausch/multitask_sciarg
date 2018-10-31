import sklearn_crfsuite
import load_conll
from evaluation.confusion_matrix import ConfusionMatrix
import numpy as np
import itertools
import utils

def word2features(sent, i):
    word = sent[i]
    features = {
        'word': word,
    }
    return features


def word2features_embeddings(sent, i, embd_dict):
    word = sent[i]
    features = {
        'vector': embd_dict[word]
    }
    return features

def word2features_embeddings2(sent, i, embd_dict):
    word = sent[i]
    vector = embd_dict[word]
    features = {str(i): value for (i, value) in enumerate(vector)}
    return features


def word2features_embeddings2_zero(sent, i, embd_dict):
    vectorlength = len(embd_dict["word"])
    features = {str(i): 0.0 for i in range(0, vectorlength)}
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2features_embeddings(sent, embd_dict):
    return [word2features_embeddings2(sent, i, embd_dict) if sent[i] in embd_dict else word2features_embeddings2_zero(sent, i, embd_dict) for i in range(len(sent))]


def transform_classes_to_binary(y, labels_list):
    labels_list = np.array(labels_list)
    y = np.array([np.array([np.array([1 if np.where(labels_list == label)[0] == i else 0 for i in range(0, len(labels_list))]) for label in sentence]) for sentence in y])
    return y


def grid_search_crf_lexical(task=""):
    print("Running grid search for crf, lexical, " + str(task))
    print("================================================================")

    x_train_dev, y_arg_train_dev, y_rhet_train_dev, y_aspect_train_dev, y_summary_train_dev, y_citation_train_dev = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted/train_dev/")
    x_test, y_arg_test, y_rhet_test, y_aspect_test, y_summary_test, y_citation_test = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted/test/")
    x_train, y_arg_train, y_rhet_train, y_aspect_train, y_summary_train, y_citation_train = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted_with_val_split/train/")
    x_dev, y_arg_dev, y_rhet_dev, y_aspect_dev, y_summary_dev, y_citation_dev = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted_with_val_split/dev/")

    if task == "citation":
        exclude_class = "NONE\n"
        y_train_dev = y_citation_train_dev
        y_train = y_citation_train
        y_dev = y_citation_dev
        y_test = y_citation_test
    elif task == "argumentation":
        exclude_class = "Token_Label.OUTSIDE"
        y_train_dev = y_arg_train_dev
        y_train = y_arg_train
        y_dev = y_arg_dev
        y_test = y_arg_test
    print("Data prepared")

    print("Data loaded")
    labels = list(set([lab for sublist in y_train_dev for lab in sublist]))

    c1s = [0.1, 0.01, 0.001, 0.0001]
    c2s = [0.1, 0.01, 0.001, 0.0001]

    configs = list(itertools.product(c1s, c2s))
    best_macro_f1 = 0.0
    best_config = None
    print("Data prepared")
    print("Grid search configs: {!s:s}".format(configs))

    for config in configs:
        print("Using config " + str(config))
        c1 = config[0]
        c2 = config[1]

        x_train_transformed = [sent2features(sent) for sent in x_train]
        x_dev_transformed = [sent2features(sent) for sent in x_dev]

        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=c1,
            c2=c2,
            max_iterations=100,
            all_possible_transitions=True
        )

        crf.fit(x_train_transformed, y_train)

        y_pred = crf.predict(x_dev_transformed)
        #transform it to our metrics
        y_pred = transform_classes_to_binary(y_pred, labels)
        y_dev = transform_classes_to_binary(y_dev, labels)

        confusion_matrix = ConfusionMatrix(labels=labels, gold=y_dev, predictions=y_pred, token_level=True, one_hot_encoding=True)
        confusion_matrix.compute_all_scores(exclude_class=exclude_class)
        macro_f1 = confusion_matrix.macrof1
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_config = config
        print(str(confusion_matrix.get_all_results()))

    # we found the best config, do it all again on train_dev + test:
    print("Best Config " + str(best_config))
    print("Best Macro F1 " + str(best_macro_f1))
    c1 = best_config[0]
    c2 = best_config[1]

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=c1,
        c2=c2,
        max_iterations=100,
        all_possible_transitions=True
    )
    x_train_dev_transformed = [sent2features(sent) for sent in x_train_dev]
    x_test_transformed = [sent2features(sent) for sent in x_test]

    crf.fit(x_train_dev_transformed, y_train_dev)

    y_pred = crf.predict(x_test_transformed)
    y_pred = transform_classes_to_binary(y_pred, labels)
    y_test = transform_classes_to_binary(y_test, labels)

    confusion_matrix = ConfusionMatrix(labels=labels, gold=y_test, predictions=y_pred, token_level=True, one_hot_encoding=True)
    confusion_matrix.compute_all_scores(exclude_class=exclude_class)
    print(str(confusion_matrix.get_all_results()))



def grid_search_crf_embeddings(task="", embd_dict=None):
    print("Running grid search for crf, embeddings, " + str(task))
    print("================================================================")

    x_train_dev, y_arg_train_dev, y_rhet_train_dev, y_aspect_train_dev, y_summary_train_dev, y_citation_train_dev = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted/train_dev/")
    x_test, y_arg_test, y_rhet_test, y_aspect_test, y_summary_test, y_citation_test = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted/test/")
    x_train, y_arg_train, y_rhet_train, y_aspect_train, y_summary_train, y_citation_train = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted_with_val_split/train/")
    x_dev, y_arg_dev, y_rhet_dev, y_aspect_dev, y_summary_dev, y_citation_dev = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted_with_val_split/dev/")

    if task == "citation":
        exclude_class = "NONE\n"
        y_train_dev = y_citation_train_dev
        y_train = y_citation_train
        y_dev = y_citation_dev
        y_test = y_citation_test
    elif task == "argumentation":
        exclude_class = "Token_Label.OUTSIDE"
        y_train_dev = y_arg_train_dev
        y_train = y_arg_train
        y_dev = y_arg_dev
        y_test = y_arg_test
    print("Data prepared")

    print("Data loaded")
    labels = list(set([lab for sublist in y_train_dev for lab in sublist]))

    c1s = [0.1, 0.2, 0.001, 0.0001]
    c2s = [0.1, 0.2, 0.001, 0.0001]

    configs = list(itertools.product(c1s, c2s))
    best_macro_f1 = 0.0
    best_config = None
    print("Data prepared")
    print("Grid search configs: {!s:s}".format(configs))

    for config in configs:
        print("Using config " + str(config))
        c1 = config[0]
        c2 = config[1]

        x_train_transformed = [sent2features_embeddings(sent, embd_dict) for sent in x_train]
        x_dev_transformed = [sent2features_embeddings(sent, embd_dict) for sent in x_dev]

        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=c1,
            c2=c2,
            max_iterations=100,
            all_possible_transitions=True
        )

        crf.fit(x_train_transformed, y_train)

        y_pred = crf.predict(x_dev_transformed)
        y_pred = transform_classes_to_binary(y_pred, labels)
        y_dev = transform_classes_to_binary(y_dev, labels)

        confusion_matrix = ConfusionMatrix(labels=labels, gold=y_dev, predictions=y_pred, token_level=True, one_hot_encoding=True)
        confusion_matrix.compute_all_scores(exclude_class=exclude_class)
        macro_f1 = confusion_matrix.macrof1
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_config = config
        print(str(confusion_matrix.get_all_results()))

    # we found the best config, do it all again on train_dev + test:
    print("Best Config " + str(best_config))
    print("Best Macro F1 " + str(best_macro_f1))
    c1 = best_config[0]
    c2 = best_config[1]

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=c1,
        c2=c2,
        max_iterations=100,
        all_possible_transitions=True
    )
    x_train_dev_transformed = [sent2features_embeddings(sent, embd_dict) for sent in x_train_dev]
    x_test_transformed = [sent2features_embeddings(sent, embd_dict) for sent in x_test]

    crf.fit(x_train_dev_transformed, y_train_dev)

    y_pred = crf.predict(x_test_transformed)
    y_pred = transform_classes_to_binary(y_pred, labels)
    y_test = transform_classes_to_binary(y_test, labels)

    confusion_matrix = ConfusionMatrix(labels=labels, gold=y_test, predictions=y_pred, token_level=True, one_hot_encoding=True)
    confusion_matrix.compute_all_scores(exclude_class=exclude_class)
    print(str(confusion_matrix.get_all_results()))

def main():
    print("Loading embeddings")
    # load embeddings
    # embd_dict = utils.load_embeddings("~/GoogleNews-vectors-negative300.bin", word2vec=True)
    embd_dict = utils.load_embeddings("./glove.6B.300d.txt", word2vec=False)

    print("Grid Search with CRF for Embedding Features")
    print("===========================================")
    for task in ["argumentation", "citation"]:
        grid_search_crf_embeddings(task=task, embd_dict=embd_dict)

    print("Grid Search with CRF for Lexical Features")
    print("===========================================")
    for task in ["argumentation", "citation"]:
        grid_search_crf_lexical(task=task)


if __name__=="__main__":
    main()

