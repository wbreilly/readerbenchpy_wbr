import networkx as nx
import pandas as pd
from tabulate import tabulate
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from pprint import pprint

class GraphMetrics:

    def __init__(self, graph: "Graph"):
        self.graph = graph
        self.semantic_distances_between_articles = graph.get_distances_between_articles()
        self.semantic_distances_between_authors = graph.get_distances_between_authors()
        # self.semantic_similarities_between_articles = graph.get_similarities_between_articles()
        # self.semantic_similarities_between_authors = graph.get_similarities_between_authors()

    def compute_articles_closeness(self):
        distances_graph = build_distance_graph(self.semantic_distances_between_articles)
        return dict(nx.closeness_centrality(distances_graph, distance='weight'))

    def compute_articles_betweenness(self):
        distances_graph = build_distance_graph(self.semantic_distances_between_articles)
        return dict(nx.betweenness_centrality(distances_graph, weight='weight', normalized=True))

    def compute_authors_closeness(self):
        distances_graph = build_distance_graph(self.semantic_distances_between_authors)
        return dict(nx.closeness_centrality(distances_graph, distance='weight'))

    def compute_authors_betweenness(self):
        distances_graph = build_distance_graph(self.semantic_distances_between_authors)
        return dict(nx.betweenness_centrality(distances_graph, weight='weight', normalized=True))

    def get_top_n_articles_by_closeness(self, n):
        closeness = self.compute_articles_closeness()
        articles_closeness = [(k, v) for k, v in sorted(closeness.items(), key=lambda item: item[1], reverse=True)][:n]
        return articles_closeness

    def get_top_n_articles_by_betweenness(self, n):
        betweenness = self.compute_articles_betweenness()
        articles_betweenness = [(k, v)
                                for k, v in sorted(betweenness.items(), key=lambda item: item[1], reverse=True)][:n]
        return articles_betweenness

    def get_top_n_authors_by_closeness(self, n):
        closeness = self.compute_authors_closeness()
        authors_closeness = [(k, v) for k, v in sorted(closeness.items(), key=lambda item: item[1], reverse=True)][:n]
        return authors_closeness

    def get_top_n_authors_by_betweenness(self, n):
        betweenness = self.compute_authors_betweenness()
        authors_betweenness = [(k, v)
                               for k, v in sorted(betweenness.items(), key=lambda item: item[1], reverse=True)][:n]
        return authors_betweenness

    def perform_articles_agglomerative_clustering(self):
        distance_threshold = self.graph.articles_mean
        perform_agglomerative_clustering(self.graph.articles_set, self.semantic_distances_between_articles,
                                         distance_threshold)

    def perform_authors_agglomerative_clustering(self):
        distance_threshold = self.graph.authors_mean - self.graph.authors_std
        perform_agglomerative_clustering(self.graph.authors_set, self.semantic_distances_between_authors,
                                         distance_threshold)


def build_distance_graph(distances_pairs):
    distance_graph = nx.Graph()
    for pair in distances_pairs:
        entity1 = pair[0]
        entity2 = pair[1]
        distance = pair[2]
        distance_graph.add_node(entity1)
        distance_graph.add_node(entity2)
        distance_graph.add_edge(entity1, entity2, weight=distance)
    return distance_graph


def build_position_dictionary(entity_set):
    index = 0
    position_dictionary = {}
    for entity in entity_set:
        position_dictionary[entity] = index
        index += 1
    return position_dictionary


def build_distance_matrix(distances_pairs, side, position_dictionary):
    distance_matrix = [[1 for _ in range(side)] for _ in range(side)]
    for pair in distances_pairs:
        row = position_dictionary[pair[0]]
        column = position_dictionary[pair[1]]
        distance_matrix[row][column] = pair[2]
    return distance_matrix


def build_inverse_dictionary(position_dictionary):
    inverse_dictionary = {}
    for key, value in position_dictionary.items():
        inverse_dictionary[value] = key
    return inverse_dictionary


def perform_agglomerative_clustering(entity_set, semantic_distances, distance_threshold):
    position_dictionary = build_position_dictionary(entity_set)
    side = len(entity_set)
    distance_matrix = build_distance_matrix(semantic_distances, side, position_dictionary)
    model = AgglomerativeClustering(affinity='precomputed', n_clusters=None,
                                    distance_threshold=distance_threshold, linkage="average")
    results = model.fit(distance_matrix)
    print(max(results.labels_))
    # plot_clustering_labels(results.labels_, side)
    print_agglomerative_clustering_results(results, build_inverse_dictionary(position_dictionary))


def print_agglomerative_clustering_results(results, inverse_position_dictionary):
    clusters = {}
    for index, label in enumerate(results.labels_):
        clusters[label] = clusters.get(label, [])
        clusters[label].append(inverse_position_dictionary[index].name)
    pprint(clusters)


def plot_clustering_labels(labels, side):
    plt.scatter(range(side), range(side), c=labels, cmap='rainbow')
    plt.show()