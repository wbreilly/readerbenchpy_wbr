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
import matplotlib.pyplot as plt

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
    for counter,df in enumerate(list_of_dfs[::2]):
        df[['junk','para','sent']] = df['node'].str.split('.',expand=True)
        df.loc[pd.isnull(df.sent), 'para'] = np.nan
        para_max = df.groupby(by='para')['importance'].max()
        df['para_max'] = df['para'].map(para_max)
        
        first_sent = df[df['sent'] =='1']
        
        firstmax=[]
        firstmax= first_sent.importance == first_sent.para_max
        prop=[]
        prop= np.sum(firstmax)/len(firstmax)
        
        first.append([prop,text_names[counter]])
        
        # first.append([first_sent.importance == first_sent.para_max,text_names[counter]])
        
    # out = pd.concat(first)
    # prop= np.sum(out)/len(out)
    return first

prop = sentmax(list_of_dfs)

#create adjacency matrix from weighted edge list

### COREF needs work. Weight is currently nan.


import networkx
edges = list_of_dfs[3]
# start with simpler case, single type of edge instead of multiple
new = edges.copy()
new['weight'] = new['weight'].astype(float) 

for edge in new.name.unique().tolist():
    new = edges.copy()
    new['weight'] = new['weight'].astype(float) 
    new = new[new.name == edge]
    new = new.filter(['source','target','weight'], axis=1)
    
    edge_list = new.values.tolist()
    g = networkx.DiGraph()
    for i in range(len(edge_list)):
        g.add_edge(edge_list[i][0], edge_list[i][1], weight=edge_list[i][2])
    
    A = networkx.adjacency_matrix(g).A
    plt.imshow(A)
    plt.show()
































    
        
        
        



    
    