"""
Created on Fri Sep 18 18:57:55 2020

Benchmarks of cna_graph with Hinze texts. Getting a feel for what it does.

@author: WBR
"""
import sys
sys.path.append('/Users/WBR/walter/diss_readerbenchpy/readerbenchpy/wbr_scratch')

from graph_extractor import compute_graph,create_edge_df,create_node_df
from rb.core.lang import Lang
from pandas import pandas as pd

# text = ['Mona was a good dog. She was the best dog named Mona.']

del text
with open('/Users/WBR/walter/diss_readerbenchpy/texts/endocrine3.txt') as f:
    text = f.read()
text = [text]

models = [{"model":"word2vec","corpus":"coca"},
          {"model":"lsa","corpus":"coca"},
          {"model":"lda","corpus":"coca"}]
result = compute_graph(text,Lang.EN,models)


df = create_node_df(result)

