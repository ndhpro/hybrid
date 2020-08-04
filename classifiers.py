from __future__ import print_function
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC


def param_parser():
    parser = argparse.ArgumentParser(
        description='Machine Learning based Classifiers')
    parser.add_argument('--input-path',
                        nargs='?',
                        default='./output/scg.csv',
                        help='Input folder with CSV.')
    parser.add_argument('--output-path',
                        nargs='?',
                        default='./result/',
                        help='Result path.')
    parser.add_argument('--seed',
                        type=int,
                        default=2020,
                        help='Random seed. Default is 2020.')
    args = parser.parse_args()
    return args


def report(args, names, y_true, y_pred):
    with open(f'{args.output_path}/result.txt', 'w') as f:
        for name in names:
            clf_report = metrics.classification_report(
                y_true[name], y_pred[name], digits=4)

            cnf_matrix = metrics.confusion_matrix(y_true[name], y_pred[name])
            TN, FP, FN, TP = cnf_matrix.ravel()
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)

            f.write(str(name) + ':\n')
            f.write('Accuracy: %0.4f\n' %
                    metrics.accuracy_score(y_true[name], y_pred[name]))
            f.write('ROC AUC: %0.4f\n' %
                    metrics.roc_auc_score(y_true[name], y_pred[name]))
            f.write('TPR: %0.4f\nFPR: %0.4f\n' % (TPR, FPR))
            f.write('Classification report:\n' + str(clf_report) + '\n')
            f.write('Confusion matrix:\n' + str(cnf_matrix) + '\n')
            f.write(64*'-'+'\n')


def draw_roc(args, names, colors, y_true, y_pred):
    plt.figure()
    for name, color in zip(names, colors):
        fpr, tpr, _ = metrics.roc_curve(y_true[name], y_pred[name])
        auc = metrics.roc_auc_score(y_true[name], y_pred[name])
        plt.plot(fpr, tpr, color=color,
                 label='%s ROC (area = %0.3f)' % (name, auc))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'{args.output_path}/roc.png', dpi=300)


def load_data(args):
    data = pd.read_csv(args.input_path)
    X = data.values[:, 1:-1]
    y = data.values[:, -1].astype('int')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=args.seed, test_size=0.3, stratify=y)
    print(f'\tOriginal: {X_train.shape} {X_test.shape}')

    fs = SelectFromModel(
        LinearSVC(penalty="l1", dual=False, random_state=args.seed).fit(X_train, y_train), prefit=True)
    X_train = fs.transform(X_train)
    X_test = fs.transform(X_test)
    pickle.dump(fs, open(f'{args.output_path}model/fs.sav', 'wb'))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    pickle.dump(scaler, open(f'{args.output_path}model/scaler.sav', 'wb'))
    print(f'\tPreprocessed: {X_train.shape} {X_test.shape}')

    return X_train, y_train, X_test, y_test


def main(args):
    y_true = dict()
    y_pred = dict()

    names = ['Naive Bayes', 'Decision Tree',
             'k-Nearest Neighbors', 'SVM', 'Random Forest']
    fnames = ['NB', 'DT', 'kNN',  'SVM', 'RF']

    classifiers = [
        GaussianNB(),
        DecisionTreeClassifier(random_state=args.seed,
                               class_weight='balanced'),
        KNeighborsClassifier(n_jobs=-1),
        SVC(random_state=args.seed, class_weight='balanced'),
        RandomForestClassifier(random_state=args.seed,
                               class_weight='balanced', n_jobs=-1),
    ]

    hyperparam = [
        {},
        {'criterion': ['gini', 'entropy']},
        {'n_neighbors': [5, 100, 500], 'weights': ['uniform', 'distance']},
        {'C': np.logspace(-3, 3, 7), 'gamma': np.logspace(-3, 3, 7)},
        {'criterion': ['gini', 'entropy'], 'n_estimators': [
            10, 100, 1000], 'bootstrap': [True, False]},
    ]

    colors = ['blue', 'orange', 'green', 'red',
              'purple', 'brown', 'pink', 'gray']

    print('Preprocessing data...')
    t = time()
    X_train, y_train, X_test, y_test = load_data(args)

    for name, fname, est, hyper in zip(names, fnames, classifiers, hyperparam):
        print(f'{name}...')
        clf = GridSearchCV(est, hyper, cv=5, n_jobs=-1)

        t = time()
        clf.fit(X_train, y_train)
        print('\tTraining done in %0.2f' % (time()-t))

        t = time()
        y_true[name],  y_pred[name] = y_test, clf.predict(X_test)
        print('\tTesting done in %0.2f' % (time()-t))

        acc = 100 * metrics.accuracy_score(y_true[name], y_pred[name])
        print('\tAccuracy: %0.2f' % acc)

        pickle.dump(clf, open(f'{args.output_path}model/{fname}.sav', 'wb'))

    report(args, names, y_true, y_pred)
    draw_roc(args, names, colors, y_true, y_pred)


if __name__ == '__main__':
    args = param_parser()
    os.mkdir(f'{args.output_path}/model/')
    main(args)
