"""
# Created on Wed Sep 16 13:49:28 2020
@author: WBR
"""
import os
os.chdir('/Users/WBR/walter/diss_readerbenchpy/')


import sys
sys.path.append('/Users/WBR/walter/diss_readerbenchpy/readerbenchpy/wbr_scratch')

from graph_extractor import compute_graph,create_edge_df,create_node_df
from rb.core.lang import Lang
from pandas import DataFrame

text = ['Mona was a good dog. She was the best dog named Mona.']
models = [{"model":"word2vec","corpus":"coca"},{"model":"lsa","corpus":"coca"}]
result = compute_graph(text,Lang.EN,models)

df = create_edge_df(result)




    
    
    
    
    





































