"""
Created on Fri Sep 18 18:57:55 2020

Benchmarks of cna_graph with Hinze texts. Getting a feel for what it does.

@author: WBR
"""
import sys
# sys.path.append('/Users/WBR/walter/diss_readerbenchpy/readerbenchpy/wbr_scratch')

from graph_extractor import compute_graph,create_edge_df,create_node_df
from rb.core.lang import Lang
from pandas import pandas as pd

# text = ['Mona was a good dog. She was the best dog named Mona.']


with open('/Users/WBR/walter/diss_readerbenchpy/texts/endocrine.txt') as f:
    text = f.read()
text = [text]

models = [{"model":"word2vec","corpus":"coca"},
          {"model":"lsa","corpus":"coca"},
          {"model":"lda","corpus":"coca"}]
result = compute_graph(text,Lang.EN,models)


df = create_node_df(result)
dfe = create_edge_df(result)

text_names = ['endocrine_p3.txt',
              'endocrine_p2.txt',
              'endocrine_p1.txt',
              'endocrine.txt']

def import_texts(text_names: list):
    path = '/Users/WBR/walter/diss_readerbenchpy/texts/'
    text_list=[]
    for file in text_names:
        with open(path + file) as f:
            text_list.append(f.read())
    return text_list
        
def batch_texts(text_list):
    list_of_dfs=[]    
    for text in text_list:
        result = compute_graph([text],Lang.EN,models)
        list_of_dfs.append(create_node_df(result)) 
        list_of_dfs.append(create_edge_df(result))
    return list_of_dfs
    
    
    
text_list = import_texts(text_names)
list_of_dfs = batch_texts(text_list)
    
    
    