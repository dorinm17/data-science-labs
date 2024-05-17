# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2023-2024, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd
import copy
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt

def normalisation(df):
    df_norm = copy.deepcopy(df)

    for col in df_norm.columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()

        df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    
    return df_norm


def dist_euclidienne(exemple1, exemple2):
    distance = np.linalg.norm(exemple1 - exemple2)
    return distance


def centroide(data):
    if isinstance(data, pd.DataFrame):
        return data.mean()

    return np.mean(data, axis=0)


def dist_centroides(centroid1, centroid2):
    c1 = centroide(centroid1)
    c2 = centroide(centroid2)

    return dist_euclidienne(c1, c2)


def initialise_CHA(DF):
    partition = {i: [i] for i in range(len(DF))}
    return partition


def fusionne(DF, partition, verbose=False):
    min_distance = float('inf')
    clusters_to_merge = None

    for i, c1 in partition.items():
        for j, c2 in partition.items():
            if i != j:
                distance = dist_centroides(DF.iloc[c1], DF.iloc[c2])

                if distance < min_distance:
                    min_distance = distance
                    clusters_to_merge = [i, j]

    cluster1, cluster2 = clusters_to_merge
    new_cluster = partition[cluster1] + partition[cluster2]

    new_partition = {key: value for key, value in partition.items() if key not in clusters_to_merge}
    new_partition[max(partition.keys()) + 1] = new_cluster

    if verbose:
        print("fusionne: distance minimale trouvée entre ", clusters_to_merge, " = ", min_distance)
        print("fusionne: les 2 clusters dont les clés sont ", clusters_to_merge, " sont fusionnés")
        print("fusionne: on crée la nouvelle clé", max(partition.keys()) + 1, " dans le dictionnaire.")
        print("fusionne: les clés de ", clusters_to_merge, " sont supprimées car leurs clusters ont été fusionnés.")
    
    return new_partition, cluster1, cluster2, min_distance


def CHA_centroid(DF, verbose=False, dendrogramme=False):
    curr_part = initialise_CHA(DF)
    final_res = []

    if verbose:
        print("CHA_centroid: clustering hiérarchique ascendant, version Centroid Linkage")

    for i in range(len(DF) - 1):
        new_part, cluster1, cluster2, dist = fusionne(DF, curr_part, verbose)
        new_size = len(curr_part[cluster1] + curr_part[cluster2])
        final_res.append([cluster1, cluster2, dist, new_size])

        if verbose:
            print(f"CHA_centroid: une fusion réalisée de {cluster1:d}  avec  {cluster2:d} de distance  {dist:.4f}")
            print(f"CHA_centroid: le nouveau cluster contient  {new_size:d}  exemples")

        curr_part = new_part

    if verbose:
        print("CHA_centroid: plus de fusion possible, il ne reste qu'un cluster unique.")

    if dendrogramme:
        plt.figure(figsize=(30, 15))
        plt.title('Dendrogramme (Approche Centroid linkage)', fontsize=25)
        plt.grid()
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        scipy.cluster.hierarchy.dendrogram(final_res, leaf_font_size=24.,)
        plt.show()

    return final_res


def dist_complete(df_cluster1, df_cluster2):
    max_distance = 0
    
    for i, point1 in df_cluster1.iterrows():
        for j, point2 in df_cluster2.iterrows():
            distance = dist_euclidienne(point1, point2)
            if distance > max_distance:
                max_distance = distance
                max_point1 = i
                max_point2 = j - df_cluster2.index[0]
                
    return max_distance, (max_point2, max_point1)


def dist_simple(df_cluster1, df_cluster2):
    min_distance = np.inf
    
    for i, point1 in df_cluster1.iterrows():
        for j, point2 in df_cluster2.iterrows():
            distance = dist_euclidienne(point1, point2)
            if distance < min_distance:
                min_distance = distance
                min_point1 = i
                min_point2 = j - df_cluster2.index[0]
                
    return min_distance, (min_point2, min_point1)


def dist_average(df_cluster1, df_cluster2):
    distances = []
    
    for _, point1 in df_cluster1.iterrows():
        for _, point2 in df_cluster2.iterrows():
            distances.append(dist_euclidienne(point1, point2))
                
    avg_dist = np.mean(distances)

    return avg_dist, len(distances)


def fusionne_complete(DF, partition, verbose=False):
    min_distance = float('inf')
    clusters_to_merge = None

    for i, c1 in partition.items():
        for j, c2 in partition.items():
            if i != j:
                distance = dist_complete(DF.iloc[c1], DF.iloc[c2])[0]

                if distance < min_distance:
                    min_distance = distance
                    clusters_to_merge = [i, j]

    cluster1, cluster2 = clusters_to_merge
    new_cluster = partition[cluster1] + partition[cluster2]

    new_partition = {key: value for key, value in partition.items() if key not in clusters_to_merge}
    new_partition[max(partition.keys()) + 1] = new_cluster

    if verbose:
        print("fusionne: distance minimale trouvée entre ", clusters_to_merge, " = ", min_distance)
        print("fusionne: les 2 clusters dont les clés sont ", clusters_to_merge, " sont fusionnés")
        print("fusionne: on crée la nouvelle clé", max(partition.keys()) + 1, " dans le dictionnaire.")
        print("fusionne: les clés de ", clusters_to_merge, " sont supprimées car leurs clusters ont été fusionnés.")
    
    return new_partition, cluster1, cluster2, min_distance


def CHA_complete(DF, verbose=False, dendrogramme=False):
    curr_part = initialise_CHA(DF)
    final_res = []

    if verbose:
        print("CHA_complete: clustering hiérarchique ascendant, version Complete Linkage")

    for i in range(len(DF) - 1):
        new_part, cluster1, cluster2, dist = fusionne_complete(DF, curr_part, verbose)
        new_size = len(curr_part[cluster1] + curr_part[cluster2])
        final_res.append([cluster1, cluster2, dist, new_size])

        if verbose:
            print(f"CHA_complete: une fusion réalisée de {cluster1:d}  avec  {cluster2:d} de distance  {dist:.4f}")
            print(f"CHA_complete: le nouveau cluster contient  {new_size:d}  exemples")

        curr_part = new_part

    if verbose:
        print("CHA_complete: plus de fusion possible, il ne reste qu'un cluster unique.")

    if dendrogramme:
        plt.figure(figsize=(30, 15))
        plt.title('Dendrogramme (Approche Complete linkage)', fontsize=25)
        plt.grid()
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        scipy.cluster.hierarchy.dendrogram(final_res, leaf_font_size=24.,)
        plt.show()

    return final_res


def fusionne_simple(DF, partition, verbose=False):
    min_distance = float('inf')
    clusters_to_merge = None

    for i, c1 in partition.items():
        for j, c2 in partition.items():
            if i != j:
                distance = dist_simple(DF.iloc[c1], DF.iloc[c2])[0]

                if distance < min_distance:
                    min_distance = distance
                    clusters_to_merge = [i, j]

    cluster1, cluster2 = clusters_to_merge
    new_cluster = partition[cluster1] + partition[cluster2]

    new_partition = {key: value for key, value in partition.items() if key not in clusters_to_merge}
    new_partition[max(partition.keys()) + 1] = new_cluster

    if verbose:
        print("fusionne_simple: distance minimale trouvée entre ", clusters_to_merge, " = ", min_distance)
        print("fusionne_simple: les 2 clusters dont les clés sont ", clusters_to_merge, " sont fusionnés")
        print("fusionne_simple: on crée la nouvelle clé", max(partition.keys()) + 1, " dans le dictionnaire.")
        print("fusionne_simple: les clés de ", clusters_to_merge, " sont supprimées car leurs clusters ont été fusionnés.")
    
    return new_partition, cluster1, cluster2, min_distance


def CHA_simple(DF, verbose=False, dendrogramme=False):
    curr_part = initialise_CHA(DF)
    final_res = []

    if verbose:
        print("CHA_simple: clustering hiérarchique ascendant, version Simple Linkage")

    for i in range(len(DF) - 1):
        new_part, cluster1, cluster2, dist = fusionne_simple(DF, curr_part, verbose)
        new_size = len(curr_part[cluster1] + curr_part[cluster2])
        final_res.append([cluster1, cluster2, dist, new_size])

        if verbose:
            print(f"CHA_simple: une fusion réalisée de {cluster1:d}  avec  {cluster2:d} de distance  {dist:.4f}")
            print(f"CHA_simple: le nouveau cluster contient  {new_size:d}  exemples")

        curr_part = new_part

    if verbose:
        print("CHA_simple: plus de fusion possible, il ne reste qu'un cluster unique.")

    if dendrogramme:
        plt.figure(figsize=(30, 15))
        plt.title('Dendrogramme (Approche Simple linkage)', fontsize=25)
        plt.grid()
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        scipy.cluster.hierarchy.dendrogram(final_res, leaf_font_size=24.,)
        plt.show()

    return final_res


def fusionne_average(DF, partition, verbose=False):
    min_distance = float('inf')
    clusters_to_merge = None

    for i, c1 in partition.items():
        for j, c2 in partition.items():
            if i != j:
                distance = dist_average(DF.iloc[c1], DF.iloc[c2])[0]

                if distance < min_distance:
                    min_distance = distance
                    clusters_to_merge = [i, j]

    cluster1, cluster2 = clusters_to_merge
    new_cluster = partition[cluster1] + partition[cluster2]

    new_partition = {key: value for key, value in partition.items() if key not in clusters_to_merge}
    new_partition[max(partition.keys()) + 1] = new_cluster

    if verbose:
        print("fusionne_average: distance moyenne minimale trouvée entre ", clusters_to_merge, " = ", min_distance)
        print("fusionne_average: les 2 clusters dont les clés sont ", clusters_to_merge, " sont fusionnés")
        print("fusionne_average: on crée la nouvelle clé", max(partition.keys()) + 1, " dans le dictionnaire.")
        print("fusionne_average: les clés de ", clusters_to_merge, " sont supprimées car leurs clusters ont été fusionnés.")
    
    return new_partition, cluster1, cluster2, min_distance


def CHA_average(DF, verbose=False, dendrogramme=False):
    curr_part = initialise_CHA(DF)
    final_res = []

    if verbose:
        print("CHA_average: clustering hiérarchique ascendant, version Average Linkage")

    for i in range(len(DF) - 1):
        new_part, cluster1, cluster2, dist = fusionne_average(DF, curr_part, verbose)
        new_size = len(curr_part[cluster1] + curr_part[cluster2])
        final_res.append([cluster1, cluster2, dist, new_size])

        if verbose:
            print(f"CHA_average: une fusion réalisée de {cluster1:d}  avec  {cluster2:d} de distance  {dist:.4f}")
            print(f"CHA_average: le nouveau cluster contient  {new_size:d}  exemples")

        curr_part = new_part

    if verbose:
        print("CHA_average: plus de fusion possible, il ne reste qu'un cluster unique.")

    if dendrogramme:
        plt.figure(figsize=(30, 15))
        plt.title('Dendrogramme (Approche Average linkage)', fontsize=25)
        plt.grid()
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        scipy.cluster.hierarchy.dendrogram(final_res, leaf_font_size=24.,)
        plt.show()

    return final_res


def CHA(DF, linkage='centroid', verbose=False, dendrogramme=False):
    if linkage == 'complete':
        return CHA_complete(DF, verbose, dendrogramme)

    if linkage == 'simple':
        return CHA_simple(DF, verbose, dendrogramme)

    if linkage == 'average':
        return CHA_average(DF, verbose, dendrogramme)

    return CHA_centroid(DF, verbose, dendrogramme)


def inertie_cluster(Ens):
    """ Array -> float
        Ens: array qui représente un cluster
        Hypothèse: len(Ens)> >= 2
        L'inertie est la somme (au carré) des distances des points au centroide.
    """
    centroid = centroide(Ens)
    Ens = np.array(Ens)
    inertia = sum(dist_euclidienne(point, centroid)**2 for point in Ens)
    return inertia


def init_kmeans(K,Ens):
    """ int * Array -> Array
        K : entier >1 et <=n (le nombre d'exemples de Ens)
        Ens: Array contenant n exemples
    """
    if isinstance(Ens, pd.DataFrame):
        Ens = Ens.values

    random_indices = np.random.choice(len(Ens), size=K, replace=False)
    
    return Ens[random_indices]


def plus_proche(Exe,Centres):
    """ Array * Array -> int
        Exe : Array contenant un exemple
        Centres : Array contenant les K centres
    """
    min_distance = float('inf')
    nearest_index = None
    
    for i, centroid in enumerate(Centres):
        distance = dist_euclidienne(Exe, centroid)
    
        if distance < min_distance:
            min_distance = distance
            nearest_index = i
        elif distance == min_distance :
            nearest_index = min(i, nearest_index)
    
    return nearest_index


def affecte_cluster(Base,Centres):
    """ Array * Array -> dict[int,list[int]]
        Base: Array contenant la base d'apprentissage
        Centres : Array contenant des centroides
    """
    if isinstance(Base, pd.DataFrame):
        Base = Base.values

    matrice_affectation = {i: [] for i in range(len(Centres))}

    for i, example in enumerate(Base):
        le_plus_proche_clust = plus_proche(example, Centres)
        matrice_affectation[le_plus_proche_clust].append(i)
    
    return matrice_affectation


def nouveaux_centroides(Base,U):
    """ Array * dict[int,list[int]] -> DataFrame
        Base : Array contenant la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    new_centroids = np.array([centroide(Base.iloc[idxs]) for idxs in U.values()])
    return new_centroids


def inertie_globale(Base, U):
    """ Array * dict[int,list[int]] -> float
        Base : Array pour la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    total_inertia = sum(inertie_cluster(Base.iloc[idxs]) for idxs in U.values())
    return total_inertia


def kmoyennes(K, Base, epsilon, iter_max):
    """ int * Array * float * int -> tuple(Array, dict[int,list[int]])
        K : entier > 1 (nombre de clusters)
        Base : Array pour la base d'apprentissage
        epsilon : réel >0
        iter_max : entier >1
    """
    centroids = init_kmeans(K, Base)
    prev_inertia = 1 + epsilon
    
    for i in range(1, iter_max + 1):
        U = affecte_cluster(Base, centroids)
        total_inertia = inertie_globale(Base, U)
        difference = np.abs(prev_inertia - total_inertia)

        print(f"iteration {i} Inertie: {total_inertia:.4f} Difference: {difference:.4f}")
        
        if difference < epsilon:
            break

        prev_inertia = total_inertia
        centroids = nouveaux_centroides(Base, U)

    return centroids, U