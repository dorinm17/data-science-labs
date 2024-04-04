# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2023-2024, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2024

# Import de packages externes
import numpy as np
import pandas as pd
import copy
import math

# ---------------------------
class Classifier:
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        # ------------------------------
        predictions = np.array([self.predict(x) for x in desc_set])
        accuracy = np.mean(predictions == label_set)
        return accuracy


class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.k = k

    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        distances = np.linalg.norm(self.desc_set - x, axis=1)
        sorted_indices = np.argsort(distances)
        k_nearest_labels = self.label_set[sorted_indices[:self.k]]

        return k_nearest_labels.mean()
    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        return -1 if self.score(x) < 0 else 1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.desc_set = desc_set
        self.label_set = label_set


class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        v = np.random.uniform(-1, 1, input_dimension)
        self.w = v / np.linalg.norm(v)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        print("Pas d'apprentissage pour ce classifieur !")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.sign(np.dot(self.w, x))
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return -1 if self.score(x) < 0 else 1


class ClassifierKNN_MC(ClassifierKNN):
    def __init__(self, input_dimension, k, nb_classes):
        super().__init__(input_dimension, k)
        self.nb_classes = nb_classes

    def score(self, x):
        distances = np.linalg.norm(self.desc_set - x, axis=1)
        sorted_indices = np.argsort(distances)
        k_nearest_labels = self.label_set[sorted_indices[:self.k]]

        class_scores = []
        for c in range(self.nb_classes):
            class_scores.append(np.mean(k_nearest_labels == c))

        return np.array(class_scores)

    def predict(self, x):
        class_scores = self.score(x)
        return np.argmax(class_scores)


class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        self.eps = learning_rate
        self.w = np.zeros(input_dimension)

        if not init:
            self.w = np.random.uniform(0, 1, input_dimension)
            self.w = (self.w * 2 - 1) * 0.001
        
        self.allw =[self.w.copy()] # stockage des premiers poids
    
    def get_allw(self):
        return self.allw

    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        lst_idx = np.arange(desc_set.shape[0])
        np.random.shuffle(lst_idx)

        for i in lst_idx:
            x = desc_set[i]
            y = label_set[i]
            if y * self.score(x) <= 0:
                self.w += self.eps * y * x
                self.allw = np.vstack((self.allw, self.w))
     
    def train(self, desc_set, label_set, nb_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """        
        differences = []

        for _ in range(nb_max):
            initial_w = np.copy(self.w)
            self.train_step(desc_set, label_set)
            difference = np.linalg.norm(self.w - initial_w)
            differences.append(difference)

            if difference < seuil:
                break

        return differences
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(self.w, x)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return np.sign(self.score(x)).astype(int)
    

class ClassifierPerceptronBiais(ClassifierPerceptron):
    """ Perceptron de Rosenblatt avec biais
        Variante du perceptron de base
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        # Appel du constructeur de la classe mère
        super().__init__(input_dimension, learning_rate, init)
        # Affichage pour information (décommentez pour la mise au point)
        # print("Init perceptron biais: w= ",self.w," learning rate= ",learning_rate)
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """  
        lst_idx = np.arange(desc_set.shape[0])
        np.random.shuffle(lst_idx)

        for i in lst_idx:
            x = desc_set[i]
            y = label_set[i]
            f_x = self.score(x)

            if y * f_x < 1:
                self.w += self.eps * (y - f_x) * x
                self.allw = np.vstack((self.allw, self.w))


class ClassifierMultiOAA(Classifier):
    """ Classifieur multi-classes
    """
    def __init__(self, cl_bin):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - cl_bin: classifieur binaire positif/négatif
            Hypothèse : input_dimension > 0
        """
        self.cl_bin = cl_bin
        self.classifiers = []
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        classes = np.unique(label_set)
        for c in classes:
            cl = copy.deepcopy(self.cl_bin)
            y_tmp = np.where(label_set == c, 1, -1)
            cl.train(desc_set, y_tmp)
            self.classifiers.append(cl)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.array([cl.score(x) for cl in self.classifiers])
        
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        scores = self.score(x)
        return np.argmax(scores)


class ClassifierADALINE(Classifier):
    """ Perceptron de ADALINE
    """
    def __init__(self, input_dimension, learning_rate, history=False, niter_max=1000):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        self.eps = learning_rate
        self.hist = history
        self.niter_max = niter_max
        self.w = np.random.uniform(0, 1, input_dimension)
        self.w = (self.w * 2 - 1) * 0.001
        self.allw = [self.w.copy()] if history else None
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        for _ in range(self.niter_max):
            lst_idx = np.arange(desc_set.shape[0])
            np.random.shuffle(lst_idx)
            
            for i in lst_idx:
                x_i = desc_set[i]
                y_i = label_set[i]
                gradient = x_i.T.dot(x_i.dot(self.w) - y_i)
                self.w -= self.eps * gradient
                
                if self.hist:
                    self.allw.append(self.w.copy())
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return x.dot(self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        score = self.score(x)
        return np.sign(score)


class ClassifierADALINE2(ClassifierADALINE):
    def __init__(self,input_dimension):
        self.w = None
    
    def train(self, X, Y):
        self.w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))

def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    valeurs, nb_fois = np.unique(Y, return_counts=True)
    idx = np.where(nb_fois == np.max(nb_fois))[0][0]

    return valeurs[idx]


def shannon(P):
    """ list[Number] -> float
        Hypothèse: P est une distribution de probabilités
        - P: distribution de probabilités
        rend la valeur de l'entropie de Shannon correspondante
    """
    base = len(P)
    
    if base == 1:
        return 0.0
    
    arr_p = np.array([p * math.log(p, base) for p in P if p != 0])
    res = -np.sum(arr_p)
    
    if res == 0.0:
        return 0.0

    return res


def entropie(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
    n = len(Y)
    P = [len(np.where(Y==i)[0]) / n for i in np.unique(Y)]

    return shannon(P)   