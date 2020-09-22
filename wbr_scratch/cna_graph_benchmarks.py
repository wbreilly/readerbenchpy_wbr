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
import numpy as np

# single text/ manual input
# text = ['Mona was a good dog. She was the best dog named Mona.']
# with open('/Users/WBR/walter/diss_readerbenchpy/texts/endocrine.txt') as f:
    # text = f.read()
# text = [text]
# models = [{"model":"word2vec","corpus":"coca"},
#           {"model":"lsa","corpus":"coca"},
#           {"model":"lda","corpus":"coca"}]
# result = compute_graph(text,Lang.EN,models)
# df = create_node_df(result)
# dfe = create_edge_df(result)

# batch method
models = [{"model":"word2vec","corpus":"coca"},
          {"model":"lsa","corpus":"coca"}]
          # {"model":"lda","corpus":"coca"}]

# text_names = ['endocrine.txt',
#               'endocrine_p1.txt',
#               'endocrine_p2.txt',
#               'endocrine_p3.txt',
#               'endocrine_p4.txt',
#               'endocrine_p5.txt']

text_names=['respiratory.txt',
            'viruses.txt',
            'vision.txt',
            'digestion.txt',
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
    
# gives a proportion of paragraphs where the first sentence is the most important one
def sentmax(list_of_dfs):
    first=[]
    # node dfs, so count by two
    for df in list_of_dfs[::2]:
        df[['junk','para','sent']] = df['node'].str.split('.',expand=True)
        df.loc[pd.isnull(df.sent), 'para'] = np.nan
        para_max = df.groupby(by='para')['importance'].max()
        df['para_max'] = df['para'].map(para_max)
        
        first_sent = df[df['sent'] =='1']
        first.append(first_sent.importance == first_sent.para_max)
        
    out = pd.concat(first)
    prop= np.sum(out)/len(out)
    return prop

prop = sentmax(list_of_dfs)

    
        
        
        



    
    