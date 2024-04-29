# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2023-2024, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd
import copy
# ------------------------ 
def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    return np.mean(L), np.std(L)


def crossval(X, Y, n_iterations, iteration):
    test_size = len(X) // n_iterations

    start_idx = iteration * test_size
    end_idx = (iteration + 1) * test_size

    Xtest = X[start_idx:end_idx]
    Ytest = Y[start_idx:end_idx]

    Xapp = np.concatenate([X[:start_idx], X[end_idx:]])
    Yapp = np.concatenate([Y[:start_idx], Y[end_idx:]])

    return Xapp, Yapp, Xtest, Ytest


def crossval_strat(X, Y, n_iterations, iteration):
    Xtest, Ytest = [], []
    all_test_indices = np.array([], dtype=int)

    for class_label in np.unique(Y):
        class_indices = np.where(Y == class_label)[0]
        test_size_per_class = len(class_indices) // n_iterations

        start_idx = iteration * test_size_per_class
        end_idx = (iteration + 1) * test_size_per_class

        test_indices = class_indices[start_idx:end_idx]

        Xtest.append(X[test_indices])
        Ytest.append(Y[test_indices])
        all_test_indices = np.concatenate((all_test_indices, test_indices))
        
    Xtest = np.concatenate(Xtest)
    Ytest = np.concatenate(Ytest)

    train_indices = np.setdiff1d(np.arange(len(Y)), all_test_indices)
    Xapp = X[train_indices]
    Yapp = Y[train_indices]

    return Xapp, Yapp, Xtest, Ytest


def validation_croisee(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    classif_copy = copy.deepcopy(C)
    Xm = DS[0]
    Ym = DS[1]
    perf = []
    
    for i in range(nb_iter):
        Xapp, Yapp, Xtest, Ytest = crossval_strat(Xm, Ym, nb_iter, i)
        classif_copy = copy.deepcopy(C)
        classif_copy.train(Xapp, Yapp)
        acc = classif_copy.accuracy(Xtest, Ytest)
        perf.append(acc)
        print('Itération %d: taille base app.= %d	taille base test= %d	Taux de bonne classif: %.4f' % (i, len(Yapp), len(Ytest), acc))
    
    taux_moyen, taux_ecart = analyse_perfs(perf)

    return perf, taux_moyen, taux_ecart    
# ------------------------ 

