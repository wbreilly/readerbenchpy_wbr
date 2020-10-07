"""
Created on Fri Sep 18 18:57:55 2020

Benchmarks of cna_graph with Hinze texts. Getting a feel for what it does.

@author: WBR
"""
import os
os.chdir('/Users/WBR/walter/diss_readerbenchpy/')
import sys
sys.path.append('/Users/WBR/walter/diss_readerbenchpy/readerbenchpy/wbr_scratch')

from graph_extractor import compute_graph,create_edge_df,create_node_df
from rb.core.lang import Lang
from pandas import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    node_df = create_node_df(result)
    edge_df = create_edge_df(result)
    return node_df,edge_df

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
clean_text_names = [s.replace('.txt','') for s in text_names]

def import_texts(text_names: list):
    path = '/Users/WBR/walter/diss_readerbenchpy/texts/'
    text_list=[]
    for file in text_names:
        with open(path + file) as f:
            text_list.append(f.read())
    return text_list
        
def batch_texts(text_list):
    node_dfs=[]
    edge_dfs=[]
    for text in text_list:
        result = compute_graph([text],Lang.EN,models)
        node_dfs.append(create_node_df(result)) 
        edge_dfs.append(create_edge_df(result))
    return node_dfs, edge_dfs

    
text_list = import_texts(text_names)
node_dfs,edge_dfs = batch_texts(text_list)
    
# gives a proportion of paragraphs where the first sentence is the most important one
def sentmax(node_dfs):
    first=[]
    # node dfs, so count by two
    for counter,df in enumerate(node_dfs):
        df[['junk','para','sent']] = df['node'].str.split('.',expand=True)
        df.loc[pd.isnull(df.sent), 'para'] = np.nan
        para_max = df.groupby(by='para')['importance'].max()
        df['para_max'] = df['para'].map(para_max)
        
        first_sent = df[df['sent'] =='1']
        
        firstmax=[]
        firstmax= first_sent.importance == first_sent.para_max
        prop=[]
        prop= np.sum(firstmax)/len(firstmax)
        
        first.append([prop,clean_text_names[counter]])
    return first

prop = sentmax(node_dfs)

#%%
#create adjacency matrix from weighted edge list
### COREF needs work. Weight is currently nan.
# single type of edge instead of multiple plots. Can't show in one plot because
# similarity between different models doesn't make sense.
import networkx
edges = edge_dfs[1] #respiratory edges
new = edges.copy()
new['weight'] = new['weight'].astype(float) 

mtx_list=[]
edge_names=[]
for edge in new.name.unique().tolist():
    new = edges.copy()
    new['weight'] = new['weight'].astype(float) 
    new = new[new.name == edge]
    new = new.filter(['source','target','weight'], axis=1)
    
    edge_list = new.values.tolist()
    g = networkx.DiGraph()
    for i in range(len(edge_list)):
        g.add_edge(edge_list[i][0], edge_list[i][1], weight=edge_list[i][2])
    
    # numpy array
    A = networkx.adjacency_matrix(g).A
    #exception for COREF, nan is holding place of connection
    if (edge == 'COREF'):
        where_are_NaNs = np.isnan(A)
        A[where_are_NaNs] = 1   
    
    mtx_list.append(A)
    edge_names.append(edge)
    
    # plt.imshow(A)
    # plt.show()
    


#%%

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
    for counter,df in enumerate(edge_dfs):
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
        #bar plot
        new.groupby(['name','connection'])['weight'].mean().unstack().plot.bar(title=str(clean_text_names[counter] + " mean" ))
        new.groupby(['name','connection'])['weight'].median().unstack().plot.bar(title=str(clean_text_names[counter] + " median" ))
        
        # overlapping histograms
        g = sns.FacetGrid(new, col="name", hue="connection",hue_order=['between','within'], col_wrap=5)
        g.map(plt.hist, 'weight',alpha=.5)
        g.axes[-1].legend()
        g.set(xlim=(0, None))
        g.set_titles('{col_name}')
        plt.subplots_adjust(top=.8)
        g.fig.suptitle(str(clean_text_names[counter] + ' edge weights'))
        
plot_edge_info()

# histogram plot would be nice 


#%%
# select most and least important sentences from each para
def sentmin():
    # node dfs, so count by two
    for counter,df in enumerate(node_dfs):
        para_min = df.groupby(by='para')['importance'].min()
        df['para_min'] = df['para'].map(para_min)
        
def get_min_and_max():
    result = []
    for counter,df in enumerate(node_dfs):
        # max and min importance sentences
        out = df[(df['importance'] == df['para_min']) | (df['importance'] == df['para_max'])]
        out = out.filter(['node','importance','content'])    
        result.append(out)
    return result
    
sentmin()
result = get_min_and_max()    





















    
        
        
        



    
    