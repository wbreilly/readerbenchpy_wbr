"""
# Created on Wed Sep 16 13:49:28 2020
@author: WBR
"""
# for key in tst:
#      print(key)

# for x,y in cna.graph.edges():
#     print(x,y)
    
    
# kinda works
# note that it seems like a few nx methods don't work on CnaGraph class.
# Maybe because the keys are objects themselves? idk but factory functions don't work
# Have to manually pull out info as in graph_extractor.py 
adj_mtx = nx.adjacency_matrix(cna.graph)
plt.show(adj_mtx)

[print(adj) for adj in g.adjacency()]

# this shows me the stuff!
[(n, nbrdict) for n, nbrdict in g.adj()]


# how to convert multi graph to adjacency matrix??
import sys
sys.path.append('/Users/WBR/walter/diss_readerbenchpy/readerbenchpy/wbr_scratch')

from graph_extractor import compute_graph,create_df
from rb.core.lang import Lang

text = ['Mona was a good dog. She was the best dog named Mona.']
models = [{"model":"word2vec","corpus":"coca"},{"model":"lsa","corpus":"coca"}]
result = compute_graph(text,Lang.EN,models)

df = create_df(result)









