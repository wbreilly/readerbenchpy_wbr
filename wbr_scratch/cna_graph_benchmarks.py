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
import numpy as np
import matplotlib.pyplot as plt

# single text/ manual input
def single_text():
    text = ['Mona was a good dog. She was the best dog named Mona.']
    with open('/Users/WBR/walter/diss_readerbenchpy/texts/endocrine.txt') as f:
        text = f.read()
    text = [text]
    models = [{"model":"word2vec","corpus":"coca"},
              {"model":"lsa","corpus":"coca"},
              {"model":"lda","corpus":"coca"}]
    result = compute_graph(text,Lang.EN,models)
    df = create_node_df(result)
    dfe = create_edge_df(result)

# batch method
models = [{"model":"word2vec","corpus":"coca"},
          {"model":"lsa","corpus":"coca"}]
          #{"model":"lda","corpus":"coca"}] # not sure this is working as intended. 
          # Extremely  high sim between everything in Hinze texts.

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
#%%
# single type of edge instead of multiple plots. Can't show in one plot because
# similarity between different models doesn't make sense.
import networkx
edges = list_of_dfs[1] #respiratory edges
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

# This sums the edge weights between nodes.
new = edges.copy()
new['weight'] = new['weight'].astype(float) 

g = networkx.MultiDiGraph()
for edge in new.name.unique().tolist():
    new = edges.copy()
    new['weight'] = new['weight'].astype(float) 
    new = new[new.name == edge]
    new = new.filter(['source','target','weight'], axis=1)
    
    edge_list = new.values.tolist()

    for i in range(len(edge_list)):
        g.add_edge(edge_list[i][0], edge_list[i][1], weight=edge_list[i][2])
 
# this sums the parallel edges
A = networkx.adjacency_matrix(g).A
plt.imshow(A)
plt.show()

#%%
# PLot within paragraph connections vs. between paragraph. Q: are paragraphs module like
def plot_edge_info(): 
    for counter,df in enumerate(list_of_dfs[1::2]):
        # only sentence-to-sentence connections
        df[['sjunk','spara','ssent']] = df['source'].str.split('.',expand=True)
        df[['tjunk','tpara','tsent']] = df['target'].str.split('.',expand=True)
        pat = "Sentence"
        filt = df['sjunk'].str.contains(pat)
        df = df[filt]
        filt = df['tjunk'].str.contains(pat)
        df = df[filt]
        
        df_within = df[df['spara'] == df['tpara']].copy()
        df_within['connection'] = "within"
        df_between = df[df['spara'] != df['tpara']].copy()
        df_between['connection'] = "between"
        
        new = pd.concat([df_within,df_between], ignore_index=True)
        pat = "COREF"
        filt = new['name'].str.contains(pat)
        new = new[~filt]
        
        new['weight'] = new['weight'].astype(float)
        #plot
        new.groupby(['name','connection'])['weight'].mean().unstack().plot.bar(title=str(text_names[counter] + " mean" ))
        new.groupby(['name','connection'])['weight'].median().unstack().plot.bar(title=str(text_names[counter] + " median" ))

plot_edge_info()


#%%
# select most and least important sentences from each para
def sentmin(list_of_dfs):
    # node dfs, so count by two
    for counter,df in enumerate(list_of_dfs[::2]):
        para_min = df.groupby(by='para')['importance'].min()
        df['para_min'] = df['para'].map(para_min)
        
sentmin(list_of_dfs)

def get_min_and_max():
    result = []
    for counter,df in enumerate(list_of_dfs[::2]):
        # max and min importance sentences
        out = df[(df['importance'] == df['para_min']) | (df['importance'] == df['para_max'])]
        out = out.filter(['node','importance','content'])    
        result.append(out)
    return result
    
result = get_min_and_max()    





















    
        
        
        



    
    