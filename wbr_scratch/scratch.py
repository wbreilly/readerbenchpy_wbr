#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 13:49:28 2020

@author: WBR
"""
# for key in tst:
#      print(key)

# for x,y in cna.graph.edges():
#     print(x,y)
    
    
    
adj_mtx = nx.adjacency_matrix(cna.graph)
plt.show(adj_mtx)

[print(adj) for adj in g.adjacency()]

# this shows me the stuff!
[(n, nbrdict) for n, nbrdict in g.adj()]


# how to convert multi graph to adjacency matrix??