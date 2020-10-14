"""
Created on Fri Sep 18 18:57:55 2020

Benchmarks of cna_graph with Hinze texts. Getting a feel for what it does.

@author: WBR
"""

#%%
#setup
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
import networkx as nx
import re

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
text_names=['goldsmith_baldwin.txt']

clean_text_names = [s.replace('.txt','') for s in text_names]

# single text/ manual input
def single_text():
    text = ['Mona was a good dog. She was the best dog named Mona.']
    with open('/Users/WBR/walter/diss_readerbenchpy/texts/endocrine.txt') as f:
        text = f.read()
    text = [text]
    # remove parenthesized info if present. Used for journal example.
    text[0] = re.sub(r'\(.*?\)', '', text[0])
    models = [{"model":"word2vec","corpus":"coca"},
              {"model":"lsa","corpus":"coca"},
              {"model":"lda","corpus":"coca"}]
    result = compute_graph(text,Lang.EN,models)
    node_df = create_node_df(result)
    edge_df = create_edge_df(result)
    return node_df,edge_df

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

text_list = import_texts(text_names)                     
node_dfs,edge_dfs = batch_texts(text_list)
prop = sentmax(node_dfs)

#%%
#create adjacency matrices from weighted edge list
### COREF edges are nan, only present for subset of nodes, other edges are fully connected

# for each text return a df with edge type and np array of edges. One edge type per mtx
def get_adj_dfs(edge_dfs):
    adj_dfs=[]
    for text in edge_dfs:
        mtx_list=[]
        edge_names=[]
        for edge in text.name.unique().tolist():
            new = text.copy()
            new['weight'] = new['weight'].astype(float) 
            new = new[new.name == edge]
            new = new.filter(['source','target','weight'], axis=1)
            
            edge_list = new.values.tolist()
            g = nx.DiGraph()
            for i in range(len(edge_list)):
                g.add_edge(edge_list[i][0], edge_list[i][1], weight=edge_list[i][2])
            
            # numpy array
            A = nx.adjacency_matrix(g).A    
            mtx_list.append(A)
            edge_names.append(edge)
            # end edge
        df=pd.DataFrame({'edge':edge_names,'mtx':mtx_list})
        # df=pd.DataFrame(index=edge_names,data={'mtx':mtx_list})
        adj_dfs.append(df)
        # end text
    return adj_dfs

# This adds all edge types to a multi directed graph. adjacency_matrix sums the weights
# This nearly redundant function is necessary because COREF isn't fully connected
def get_multi_adj(edge_dfs):
    multi_adjs=[]
    for text in edge_dfs:
        g = nx.MultiDiGraph()
        for edge in text.name.unique().tolist():
            new = text.copy()
            new['weight'] = new['weight'].astype(float) 
            new = new[new.name == edge]
            new = new.filter(['source','target','weight'], axis=1)
            edge_list = new.values.tolist()            
            for i in range(len(edge_list)):
                g.add_edge(edge_list[i][0], edge_list[i][1], weight=edge_list[i][2])
            # end i
        # end edge
        A = nx.adjacency_matrix(g).A
        multi_adjs.append(A)
    # end text 
    return multi_adjs

def NormalizeData(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))     
   
# normalize multi_adjs becusae they are sum of edges and therefore greater than 1. 
# Not necessary if stand alone plot.     
def multi_adj_proc(multi_adjs):
    for i,junk in enumerate(multi_adjs):
        multi_adjs[i] = NormalizeData(multi_adjs[i])
        mask = np.isnan(multi_adjs[i])
        multi_adjs[i][mask] = 1
     
def plot_edge_adjacency(*args):
    adj_dfs = get_adj_dfs(edge_dfs)
    multi_adjs = get_multi_adj(edge_dfs)
    multi_adj_proc(multi_adjs)
    for itext in range(len(text_names)):
        df = adj_dfs[itext]
        # substitute coref for the multi_adj (coref are 1's everything else is normalized sum of edges)
        filt = df['edge'] =='COREF'
        df= df[~filt]
        df = df.append({'edge':'MultiDiGraph','mtx':multi_adjs[itext]},ignore_index=True)
        # Argument edges don't have connections to Document, so size is different without fix
        df[df[df.edge == ()]]
        
        # remove Document connections bc ARGUMENT don't have them
        filt = df['edge'].str.contains('SEMANTIC')
        df.loc[filt,'mtx'] = df.loc[filt,'mtx'].apply(lambda x:x[:-1,1:])
        
        # Appreciate how difficult this was to make!!
        g = sns.FacetGrid(df, col='edge')
        g.map_dataframe(lambda data, color: sns.heatmap(data.mtx.values[0], linewidths=0,vmin=0,vmax=1,xticklabels=False,yticklabels=False))
        g.set_titles('{col_name}')
        plt.subplots_adjust(top=.8)
        g.fig.suptitle(str(clean_text_names[itext] + ' adjacency matrices'))
        # removing colorbar from each plot is not worth coding. All or none or use Illustrator.
        fig_path= '/Users/WBR/walter/diss_readerbenchpy/figures/'
        if 'save_figs'in args: g.savefig(fig_path + 'adjacency_edges_' + clean_text_names[itext]  + '.png', format='png', dpi=1200)
        
plot_edge_adjacency('save_figs')

#%%
# PLot within paragraph connections vs. between paragraph. Q: are paragraphs module like
def plot_edge_bar_hist(edge_dfs,*args):
    '''
    Parameters
    ----------
    edge_dfs : list
        list of edge dfs from batch_texts
    *args : string
        'save_figs' : save figures
        'max_sent'  : source nodes are most important sentences
        'z_filt'    : filters out edges with z-score weight less than or equal to 1

    Returns
    -------
    None. PLots plots. 
    '''
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
        
        # new is the df used for plotting
        new = pd.concat([df_within,df_between], ignore_index=True)
        filt = new['name'].str.contains("COREF")
        new = new[~filt]
        new['weight'] = new['weight'].astype(float)
        
        if 'z_filt' in args:
            # filtered_graph
            new['z_score'] = df.groupby('name')['weight'].apply(lambda x: (x - x.mean())/x.std())
            filt = new[new['z_score' > 1]]
            new = new[filt]
        
        if 'max_sent' in args:
            # only most important sentence connections
            node_df = node_dfs[counter].copy()
            max_sent =  node_df[node_df['importance'] == node_df['para_max']]
            max_sent = max_sent.node.values.tolist()
            match = new['source'].isin(max_sent)
            new = new[match]
          
        # figure naming 
        if ('max_sent' in args) & ('z_filt' in args):
            fig_path= '/Users/WBR/walter/diss_readerbenchpy/figures/zfilt_max_I_'
        elif 'max_sent' in args:
            fig_path= '/Users/WBR/walter/diss_readerbenchpy/figures/max_I_'
        elif 'z_filt' in args:
            fig_path= '/Users/WBR/walter/diss_readerbenchpy/figures/zfilt_'
        else:
            fig_path= '/Users/WBR/walter/diss_readerbenchpy/figures/'
            
        #bar plot
        new.groupby(['name','connection'])['weight'].mean().unstack().plot.bar(title=str(clean_text_names[counter] + " mean" ))
        if 'save_figs' in args: plt.savefig(fig_path + 'bar_mean_edges_' + clean_text_names[counter]  + '.png', format='png', dpi=1200)
        new.groupby(['name','connection'])['weight'].median().unstack().plot.bar(title=str(clean_text_names[counter] + " median" ))
        if 'save_figs' in args: plt.savefig(fig_path + 'bar_median_edges_' + clean_text_names[counter]  + '.png', format='png', dpi=1200)
        
        # overlapping histograms
        g = sns.FacetGrid(new, col="name", hue="connection",hue_order=['between','within'], col_wrap=5)
        g.map(plt.hist, 'weight',alpha=.5)
        g.fig.get_axes()[0].set_yscale('log')
        g.axes[-1].legend()
        g.set(xlim=(0, None))
        g.set_titles('{col_name}')
        plt.subplots_adjust(top=.8)
        g.fig.suptitle(str(clean_text_names[counter] + ' edge weights'))
        if 'save_figs' in args: g.savefig(fig_path + 'histogram_edges_' + clean_text_names[counter]  + '.png', format='png', dpi=1200)
        
plot_edge_bar_hist(edge_dfs,'max_sent','z_filt','save_figs')

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

#%%


    
    