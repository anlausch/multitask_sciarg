import load_conll
import time
from evaluation.confusion_matrix import ConfusionMatrix
import numpy as np
import utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import itertools
import os

def transform_classes_to_binary(y, labels_list):
    labels_list = np.array(labels_list)
    y = np.array([np.array([1 if np.where(labels_list == label)[0] == i else 0 for i in range(0, len(labels_list))]) for label in y])
    return y

def grid_search_linear_svm_tfidf(task=""):
    print("Running grid search for svm, linear kernal, tfidf, " + str(task))
    print("================================================================")
    if task == "discourse":
        exclude_class = "DRI_Unspecified"
    elif task == "aspect":
        exclude_class = "NONE"
    elif task == "summary":
        exclude_class = "NONE"
    else:
        print("No valid task name provided")
        exit()

    print("SVM script started")
    start = time.time()
    x_train_dev, y_arg_train_dev, y_rhet_train_dev, y_aspect_train_dev, y_summary_train_dev, y_citation_train_dev = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted/train_dev/")
    x_test, y_arg_test, y_rhet_test, y_aspect_test, y_summary_test, y_citation_test = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted/test/")
    x_train, y_arg_train, y_rhet_train, y_aspect_train, y_summary_train, y_citation_train = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted_with_val_split/train/")
    x_dev, y_arg_dev, y_rhet_dev, y_aspect_dev, y_summary_dev, y_citation_dev = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted_with_val_split/dev/")

    print("Data loaded")
    x_train_dev = [utils.preprocess_string_tfidf(' '.join(sentence)) for sentence in x_train_dev]
    x_test = [utils.preprocess_string_tfidf(' '.join(sentence)) for sentence in x_test]
    x_dev = [utils.preprocess_string_tfidf(' '.join(sentence)) for sentence in x_dev]
    x_train = [utils.preprocess_string_tfidf(' '.join(sentence)) for sentence in x_train]

    if task == "discourse":
        exclude_class = "DRI_Unspecified"
        y_train_dev = y_rhet_train_dev
        y_train = y_rhet_train
        y_dev = y_rhet_dev
        y_test = y_rhet_test
    elif task == "aspect":
        exclude_class = "NONE"
        y_train_dev = y_aspect_train_dev
        y_train = y_aspect_train
        y_dev = y_aspect_dev
        y_test = y_aspect_test
    elif task == "summary":
        exclude_class = "NONE"
        y_train_dev = y_summary_train_dev
        y_train = y_summary_train
        y_dev = y_summary_dev
        y_test = y_summary_test

    y_train_dev = [sent[0] for sent in y_train_dev]
    y_test = [sent[0] for sent in y_test]
    y_dev = [sent[0] for sent in y_dev]
    y_train = [sent[0] for sent in y_train]
    print("Data prepared")

    labels = list(set([lab for lab in y_train_dev]))


    y_train_dev = transform_classes_to_binary(y_train_dev, labels)
    y_test = transform_classes_to_binary(y_test, labels)
    y_dev = transform_classes_to_binary(y_dev, labels)
    y_train = transform_classes_to_binary(y_train, labels)


    # grid search stuff
    possible_c = [0.1, 1.0, 10.0]
    print("Grid search configs are " + str(possible_c))

    best_f1 = 0.0
    best_c = ""

    for c in possible_c:
        print("Using config " + str(c))
        tfidf_vectorizer = TfidfVectorizer()

        # fit to train
        x_train_transformed = tfidf_vectorizer.fit_transform(x_train)
        clf = OneVsRestClassifier(SVC(kernel='linear', C=c))
        clf.fit(x_train_transformed, y_train)

        # predict on dev set
        x_dev_transformed = tfidf_vectorizer.transform(x_dev)
        y_pred = clf.predict(x_dev_transformed)
        confusion_matrix = ConfusionMatrix(labels=labels, gold=y_dev, predictions=y_pred, token_level=False, one_hot_encoding=True)
        confusion_matrix.compute_all_scores(exclude_class=exclude_class)
        if confusion_matrix.macrof1 > best_f1:
            best_f1 = confusion_matrix.macrof1
            best_c = c
        print(str(confusion_matrix.get_all_results()))

    # we found the best config, now train again on train + dev
    print("Best Config " + str(best_c))
    print("Best Macro F1 " + str(best_f1))
    tfidf_vectorizer = TfidfVectorizer()

    # fit to train_dev
    x_train_dev = tfidf_vectorizer.fit_transform(x_train_dev)
    clf = OneVsRestClassifier(SVC(kernel='linear', C=best_c))
    clf.fit(x_train_dev, y_train_dev)

    # predict on test set
    x_test = tfidf_vectorizer.transform(x_test)
    y_pred = clf.predict(x_test)
    confusion_matrix = ConfusionMatrix(labels=labels, gold=y_test, predictions=y_pred, token_level=False, one_hot_encoding=True)
    confusion_matrix.compute_all_scores(exclude_class=exclude_class)

    print(str(confusion_matrix.get_all_results()))
    print("Total training time: " + str(time.time() - start))


def grid_search_linear_svm_embeddings(embd_dict=None, task=""):
    start = time.time()
    print("Running grid search for svm, linear kernal, embeddings, " + str(task))
    print("================================================================")
    if task == "discourse":
        exclude_class = "DRI_Unspecified"
    elif task == "aspect":
        exclude_class = "NONE"
    elif task == "summary":
        exclude_class = "NONE"
    else:
        print("No valid task name provided")
        exit()

    x_train_dev, y_arg_train_dev, y_rhet_train_dev, y_aspect_train_dev, y_summary_train_dev, y_citation_train_dev = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted/train_dev/")
    x_test, y_arg_test, y_rhet_test, y_aspect_test, y_summary_test, y_citation_test = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted/test/")
    x_train, y_arg_train, y_rhet_train, y_aspect_train, y_summary_train, y_citation_train = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted_with_val_split/train/")
    x_dev, y_arg_dev, y_rhet_dev, y_aspect_dev, y_summary_dev, y_citation_dev = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted_with_val_split/dev/")

    print("Data loaded")
    x_train_dev = [utils.preprocess_string_tfidf(' '.join(sentence)) for sentence in x_train_dev]
    x_test = [utils.preprocess_string_tfidf(' '.join(sentence)) for sentence in x_test]
    x_dev = [utils.preprocess_string_tfidf(' '.join(sentence)) for sentence in x_dev]
    x_train = [utils.preprocess_string_tfidf(' '.join(sentence)) for sentence in x_train]

    if task == "discourse":
        exclude_class = "DRI_Unspecified"
        y_train_dev = y_rhet_train_dev
        y_train = y_rhet_train
        y_dev = y_rhet_dev
        y_test = y_rhet_test
    elif task == "aspect":
        exclude_class = "NONE"
        y_train_dev = y_aspect_train_dev
        y_train = y_aspect_train
        y_dev = y_aspect_dev
        y_test = y_aspect_test
    elif task == "summary":
        exclude_class = "NONE"
        y_train_dev = y_summary_train_dev
        y_train = y_summary_train
        y_dev = y_summary_dev
        y_test = y_summary_test

    y_train_dev = [sent[0] for sent in y_train_dev]
    y_test = [sent[0] for sent in y_test]
    y_dev = [sent[0] for sent in y_dev]
    y_train = [sent[0] for sent in y_train]
    print("Data prepared")

    labels = list(set([lab for lab in y_train_dev]))


    y_train_dev = transform_classes_to_binary(y_train_dev, labels)
    y_test = transform_classes_to_binary(y_test, labels)
    y_dev = transform_classes_to_binary(y_dev, labels)
    y_train = transform_classes_to_binary(y_train, labels)


    # grid search stuff
    possible_c = [0.1, 1.0, 10.0]
    print("Grid search configs are " + str(possible_c))

    best_f1 = 0.0
    best_c = 0.0000001

    for c in possible_c:
        print("Using config " + str(c))
        embedding_vectorizer = utils.MeanEmbeddingVectorizer(embds=embd_dict)

        # fit to train
        x_train_transformed = embedding_vectorizer.transform(x_train)
        clf = OneVsRestClassifier(SVC(kernel='linear', C=c))
        clf.fit(x_train_transformed, y_train)

        # predict on dev set
        x_dev_transformed = embedding_vectorizer.transform(x_dev)
        y_pred = clf.predict(x_dev_transformed)
        confusion_matrix = ConfusionMatrix(labels=labels, gold=y_dev, predictions=y_pred, token_level=False, one_hot_encoding=True)
        confusion_matrix.compute_all_scores(exclude_class=exclude_class)
        if confusion_matrix.macrof1 > best_f1:
            best_f1 = confusion_matrix.macrof1
            best_c = c
        print(str(confusion_matrix.get_all_results()))

    # we found the best config, now train again on train + dev
    print("Best Config " + str(best_c))
    print("Best Macro F1 " + str(best_f1))
    embedding_vectorizer = utils.MeanEmbeddingVectorizer(embds=embd_dict)

    # fit to train_dev
    x_train_dev = embedding_vectorizer.transform(x_train_dev)
    clf = OneVsRestClassifier(SVC(kernel='linear', C=best_c))
    clf.fit(x_train_dev, y_train_dev)

    # predict on test set
    x_test = embedding_vectorizer.transform(x_test)
    y_pred = clf.predict(x_test)
    confusion_matrix = ConfusionMatrix(labels=labels, gold=y_test, predictions=y_pred, token_level=False, one_hot_encoding=True)
    confusion_matrix.compute_all_scores(exclude_class=exclude_class)

    print(str(confusion_matrix.get_all_results()))
    print("Total training time: " + str(time.time() - start))



def grid_search_rbf_svm_tfidf(task=""):
    print("Running grid search for svm, rbf kernal, tfidf, " + str(task))
    print("================================================================")
    if task == "discourse":
        exclude_class = "DRI_Unspecified"
    elif task == "aspect":
        exclude_class = "NONE"
    elif task == "summary":
        exclude_class = "NONE"
    else:
        print("No valid task name provided")
        exit()

    print("SVM script started")
    start = time.time()
    x_train_dev, y_arg_train_dev, y_rhet_train_dev, y_aspect_train_dev, y_summary_train_dev, y_citation_train_dev = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted/train_dev/")
    x_test, y_arg_test, y_rhet_test, y_aspect_test, y_summary_test, y_citation_test = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted/test/")
    x_train, y_arg_train, y_rhet_train, y_aspect_train, y_summary_train, y_citation_train = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted_with_val_split/train/")
    x_dev, y_arg_dev, y_rhet_dev, y_aspect_dev, y_summary_dev, y_citation_dev = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted_with_val_split/dev/")

    print("Data loaded")
    x_train_dev = [utils.preprocess_string_tfidf(' '.join(sentence)) for sentence in x_train_dev]
    x_test = [utils.preprocess_string_tfidf(' '.join(sentence)) for sentence in x_test]
    x_dev = [utils.preprocess_string_tfidf(' '.join(sentence)) for sentence in x_dev]
    x_train = [utils.preprocess_string_tfidf(' '.join(sentence)) for sentence in x_train]

    if task == "discourse":
        exclude_class = "DRI_Unspecified"
        y_train_dev = y_rhet_train_dev
        y_train = y_rhet_train
        y_dev = y_rhet_dev
        y_test = y_rhet_test
    elif task == "aspect":
        exclude_class = "NONE"
        y_train_dev = y_aspect_train_dev
        y_train = y_aspect_train
        y_dev = y_aspect_dev
        y_test = y_aspect_test
    elif task == "summary":
        exclude_class = "NONE"
        y_train_dev = y_summary_train_dev
        y_train = y_summary_train
        y_dev = y_summary_dev
        y_test = y_summary_test

    y_train_dev = [sent[0] for sent in y_train_dev]
    y_test = [sent[0] for sent in y_test]
    y_dev = [sent[0] for sent in y_dev]
    y_train = [sent[0] for sent in y_train]
    print("Data prepared")

    labels = list(set([lab for lab in y_train_dev]))


    #y_train_dev = transform_classes_to_binary(y_train_dev, labels)
    #y_test = transform_classes_to_binary(y_test, labels)
    #y_dev = transform_classes_to_binary(y_dev, labels)
    #y_train = transform_classes_to_binary(y_train, labels)


    # grid search stuff
    possible_c = [0.1, 1.0, 10.0]
    possible_gamma = [0.01, 0.1, 1.0]
    configurations = list(itertools.product(possible_c, possible_gamma))
    print("Grid search configs: {!s:s}".format(configurations))

    best_f1 = 0.0
    best_conf = ""

    for (c, gamma) in configurations:
        print("Using config " + str((c, gamma)))
        tfidf_vectorizer = TfidfVectorizer()

        # fit to train
        x_train_transformed = tfidf_vectorizer.fit_transform(x_train)
        clf = SVC(kernel='rbf', C=c, gamma=gamma)
        clf.fit(x_train_transformed, y_train)

        # predict on dev set
        x_dev_transformed = tfidf_vectorizer.transform(x_dev)
        y_pred = clf.predict(x_dev_transformed)
        confusion_matrix = ConfusionMatrix(labels=labels, gold=y_dev, predictions=y_pred, token_level=False, one_hot_encoding=True)
        confusion_matrix.compute_all_scores(exclude_class=exclude_class)
        if confusion_matrix.macrof1 > best_f1:
            best_f1 = confusion_matrix.macrof1
            best_conf = (c,gamma)
        print(str(confusion_matrix.get_all_results()))

    # we found the best config, now train again on train + dev
    print("Best Config " + str(best_conf))
    print("Best Macro F1 " + str(best_f1))
    tfidf_vectorizer = TfidfVectorizer()

    # fit to train_dev
    x_train_dev = tfidf_vectorizer.fit_transform(x_train_dev)
    clf = SVC(kernel='rbf', C=best_conf[0], gamma=best_conf[1])
    clf.fit(x_train_dev, y_train_dev)

    # predict on test set
    x_test = tfidf_vectorizer.transform(x_test)
    y_pred = clf.predict(x_test)
    confusion_matrix = ConfusionMatrix(labels=labels, gold=y_test, predictions=y_pred, token_level=False, one_hot_encoding=True)
    confusion_matrix.compute_all_scores(exclude_class=exclude_class)

    print(str(confusion_matrix.get_all_results()))
    print("Total training time: " + str(time.time() - start))


def grid_search_rbf_svm_embeddings(embd_dict=None, task=""):
    print("Running grid search for svm, rbf kernal, embeddings, " + str(task))
    print("================================================================")
    if task == "discourse":
        exclude_class = "DRI_Unspecified"
    elif task == "aspect":
        exclude_class = "NONE"
    elif task == "summary":
        exclude_class = "NONE"
    else:
        print("No valid task name provided")
        exit()

    print("SVM script started")
    x_train_dev, y_arg_train_dev, y_rhet_train_dev, y_aspect_train_dev, y_summary_train_dev, y_citation_train_dev = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted/train_dev/")
    x_test, y_arg_test, y_rhet_test, y_aspect_test, y_summary_test, y_citation_test = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted/test/")
    x_train, y_arg_train, y_rhet_train, y_aspect_train, y_summary_train, y_citation_train = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted_with_val_split/train/")
    x_dev, y_arg_dev, y_rhet_dev, y_aspect_dev, y_summary_dev, y_citation_dev = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted_with_val_split/dev/")

    print("Data loaded")
    x_train_dev = [utils.preprocess_string_tfidf(' '.join(sentence)) for sentence in x_train_dev]
    x_test = [utils.preprocess_string_tfidf(' '.join(sentence)) for sentence in x_test]
    x_dev = [utils.preprocess_string_tfidf(' '.join(sentence)) for sentence in x_dev]
    x_train = [utils.preprocess_string_tfidf(' '.join(sentence)) for sentence in x_train]

    if task == "discourse":
        exclude_class = "DRI_Unspecified"
        y_train_dev = y_rhet_train_dev
        y_train = y_rhet_train
        y_dev = y_rhet_dev
        y_test = y_rhet_test
    elif task == "aspect":
        exclude_class = "NONE"
        y_train_dev = y_aspect_train_dev
        y_train = y_aspect_train
        y_dev = y_aspect_dev
        y_test = y_aspect_test
    elif task == "summary":
        exclude_class = "NONE"
        y_train_dev = y_summary_train_dev
        y_train = y_summary_train
        y_dev = y_summary_dev
        y_test = y_summary_test

    y_train_dev = [sent[0] for sent in y_train_dev]
    y_test = [sent[0] for sent in y_test]
    y_dev = [sent[0] for sent in y_dev]
    y_train = [sent[0] for sent in y_train]
    print("Data prepared")

    labels = list(set([lab for lab in y_train_dev]))


    y_train_dev = transform_classes_to_binary(y_train_dev, labels)
    y_test = transform_classes_to_binary(y_test, labels)
    y_dev = transform_classes_to_binary(y_dev, labels)
    y_train = transform_classes_to_binary(y_train, labels)


    # grid search stuff
    possible_c = [0.1, 1.0, 10.0]
    possible_gamma = [0.01, 0.1, 1.0]
    configurations = list(itertools.product(possible_c, possible_gamma))
    print("Grid search configs: {!s:s}".format(configurations))

    best_f1 = 0.0
    best_conf = ""

    for (c, gamma) in configurations:
        print("Using config " + str((c, gamma)))
        embedding_vectorizer = utils.MeanEmbeddingVectorizer(embds=embd_dict)

        # fit to train
        x_train_transformed = embedding_vectorizer.transform(x_train)
        clf = OneVsRestClassifier(SVC(kernel='rbf', C=c, gamma=gamma))
        clf.fit(x_train_transformed, y_train)

        # predict on dev set
        x_dev_transformed = embedding_vectorizer.transform(x_dev)
        y_pred = clf.predict(x_dev_transformed)
        confusion_matrix = ConfusionMatrix(labels=labels, gold=y_dev, predictions=y_pred, token_level=False, one_hot_encoding=True)
        confusion_matrix.compute_all_scores(exclude_class=exclude_class)
        if confusion_matrix.macrof1 > best_f1:
            best_f1 = confusion_matrix.macrof1
            best_conf = (c,gamma)
        print(str(confusion_matrix.get_all_results()))

    # we found the best config, now train again on train + dev
    print("Best Config " + str(best_conf))
    print("Best Macro F1 " + str(best_f1))
    embedding_vectorizer = utils.MeanEmbeddingVectorizer(embds=embd_dict)

    # fit to train_dev
    x_train_dev = embedding_vectorizer.transform(x_train_dev)
    clf = OneVsRestClassifier(SVC(kernel='rbf', C=best_conf[0], gamma=best_conf[1]))
    clf.fit(x_train_dev, y_train_dev)

    # predict on test set
    x_test = embedding_vectorizer.transform(x_test)
    y_pred = clf.predict(x_test)
    confusion_matrix = ConfusionMatrix(labels=labels, gold=y_test, predictions=y_pred, token_level=False, one_hot_encoding=True)
    confusion_matrix.compute_all_scores(exclude_class=exclude_class)

    print(str(confusion_matrix.get_all_results()))


def grid_search_rbf_svm_tfidf_embeddings(embd_dict=None, task=""):
    print("Running grid search for svm, rbf kernal, embeddings weighted tfidf, " + str(task))
    print("================================================================")
    if task == "discourse":
        exclude_class = "DRI_Unspecified"
    elif task == "aspect":
        exclude_class = "NONE"
    elif task == "summary":
        exclude_class = "NONE"
    else:
        print("No valid task name provided")
        exit()

    print("SVM script started")
    x_train_dev, y_arg_train_dev, y_rhet_train_dev, y_aspect_train_dev, y_summary_train_dev, y_citation_train_dev = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted/train_dev/")
    x_test, y_arg_test, y_rhet_test, y_aspect_test, y_summary_test, y_citation_test = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted/test/")
    x_train, y_arg_train, y_rhet_train, y_aspect_train, y_summary_train, y_citation_train = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted_with_val_split/train/")
    x_dev, y_arg_dev, y_rhet_dev, y_aspect_dev, y_summary_dev, y_citation_dev = load_conll.load_data_multiple(path="./../../annotations_conll_final_splitted_with_val_split/dev/")

    print("Data loaded")
    x_train_dev = [utils.preprocess_string_tfidf(' '.join(sentence)) for sentence in x_train_dev]
    x_test = [utils.preprocess_string_tfidf(' '.join(sentence)) for sentence in x_test]
    x_dev = [utils.preprocess_string_tfidf(' '.join(sentence)) for sentence in x_dev]
    x_train = [utils.preprocess_string_tfidf(' '.join(sentence)) for sentence in x_train]

    if task == "discourse":
        exclude_class = "DRI_Unspecified"
        y_train_dev = y_rhet_train_dev
        y_train = y_rhet_train
        y_dev = y_rhet_dev
        y_test = y_rhet_test
    elif task == "aspect":
        exclude_class = "NONE"
        y_train_dev = y_aspect_train_dev
        y_train = y_aspect_train
        y_dev = y_aspect_dev
        y_test = y_aspect_test
    elif task == "summary":
        exclude_class = "NONE"
        y_train_dev = y_summary_train_dev
        y_train = y_summary_train
        y_dev = y_summary_dev
        y_test = y_summary_test

    y_train_dev = [sent[0] for sent in y_train_dev]
    y_test = [sent[0] for sent in y_test]
    y_dev = [sent[0] for sent in y_dev]
    y_train = [sent[0] for sent in y_train]
    print("Data prepared")

    labels = list(set([lab for lab in y_train_dev]))


    y_train_dev = transform_classes_to_binary(y_train_dev, labels)
    y_test = transform_classes_to_binary(y_test, labels)
    y_dev = transform_classes_to_binary(y_dev, labels)
    y_train = transform_classes_to_binary(y_train, labels)


    # grid search stuff
    possible_c = [0.1, 1.0, 10.0]
    possible_gamma = [0.01, 0.1, 1.0]
    configurations = list(itertools.product(possible_c, possible_gamma))
    print("Grid search configs: {!s:s}".format(configurations))

    best_f1 = 0.0
    best_conf = ""

    for (c, gamma) in configurations:
        print("Using config " + str((c, gamma)))
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer = tfidf_vectorizer.fit(x_train)

        embedding_vectorizer = utils.TfidfEmbeddingVectorizer(embds=embd_dict, tfidf_vectorizer=tfidf_vectorizer)

        # fit to train
        x_train_transformed = embedding_vectorizer.transform(x_train)

        clf = OneVsRestClassifier(SVC(kernel='rbf', C=c, gamma=gamma))
        clf.fit(x_train_transformed, y_train)

        # predict on dev set
        x_dev_transformed = embedding_vectorizer.transform(x_dev)
        y_pred = clf.predict(x_dev_transformed)
        confusion_matrix = ConfusionMatrix(labels=labels, gold=y_dev, predictions=y_pred, token_level=False, one_hot_encoding=True)
        confusion_matrix.compute_all_scores(exclude_class=exclude_class)
        if confusion_matrix.macrof1 > best_f1:
            best_f1 = confusion_matrix.macrof1
            best_conf = (c,gamma)
        print(str(confusion_matrix.get_all_results()))

    # we found the best config, now train again on train + dev
    print("Best Config " + str(best_conf))
    print("Best Macro F1 " + str(best_f1))

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer = tfidf_vectorizer.fit(x_train_dev)

    embedding_vectorizer = utils.TfidfEmbeddingVectorizer(embds=embd_dict, tfidf_vectorizer=tfidf_vectorizer)

    # fit to train_dev
    x_train_dev = embedding_vectorizer.transform(x_train_dev)
    clf = OneVsRestClassifier(SVC(kernel='rbf', C=best_conf[0], gamma=best_conf[1]))
    clf.fit(x_train_dev, y_train_dev)

    # predict on test set
    x_test = embedding_vectorizer.transform(x_test)
    y_pred = clf.predict(x_test)
    confusion_matrix = ConfusionMatrix(labels=labels, gold=y_test, predictions=y_pred, token_level=False, one_hot_encoding=True)
    confusion_matrix.compute_all_scores(exclude_class=exclude_class)

    print(str(confusion_matrix.get_all_results()))


def main():
    print("Loading embeddings")
    #load embeddings
    if os.name == "nt":
        # embd_dict = utils.load_embeddings(
        #    "C:/Users/anlausch/workspace/cnn-text-classification/data/GoogleNews-vectors-negative300.bin", word2vec=True)
        embd_dict = utils.load_embeddings("C:/Users/anlausch/workspace/embedding_files/glove.6B/glove.6B.50d.txt",
                                          word2vec=False)
    else:
        # embd_dict = utils.load_embeddings("~/GoogleNews-vectors-negative300.bin", word2vec=True)
        embd_dict = utils.load_embeddings("./glove.6B.300d.txt", word2vec=False)

    # print("Grid Search with SVM for TFIDF Embedding Features")
    # print("===========================================")
    # for task in ["discourse", "aspect", "summary"]:
    #     #grid_search_linear_svm_tfidf(task=task)
    #     grid_search_rbf_svm_tfidf_embeddings(embd_dict=embd_dict, task=task)
    #
    # print("Grid Search with SVM for Embedding Features")
    # print("===========================================")
    # for task in ["discourse", "aspect", "summary"]:
    #     #grid_search_linear_svm_tfidf(task=task)
    #     grid_search_rbf_svm_embeddings(embd_dict=embd_dict, task=task)

    print("Grid Search with SVM for TFIDF")
    print("===========================================")
    for task in ["discourse", "aspect", "summary"]:
         #grid_search_linear_svm_tfidf(task=task)
         grid_search_rbf_svm_tfidf(task=task)

    print("Grid Search with SVM linear embeddings")
    print("===========================================")
    for task in ["discourse", "aspect", "summary"]:
        #grid_search_linear_svm_tfidf(task=task)
        grid_search_linear_svm_embeddings(embd_dict=embd_dict, task=task)

    print("Grid Search with SVM linear for TFIDF")
    print("===========================================")
    for task in ["discourse", "aspect", "summary"]:
        #grid_search_linear_svm_tfidf(task=task)
        grid_search_linear_svm_tfidf(task=task)


if __name__ == "__main__":
    main()