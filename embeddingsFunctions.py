import networkx as nx 
from node2vec import Node2Vec
import pandas as pd
def createGraphe(dfGroupes):
    g=nx.from_pandas_edgelist(dfGroupes,source='user_id',target='original_author',edge_attr='nb_retweeted')
    print(f"Le graphe final contient {g.number_of_nodes()} utilisateurs.")
    return g
def embedding32dimensions(graphe) :
    node=Node2Vec(graphe,dimensions=32,walk_length=10,num_walks=10,workers=2)
    model=node.fit(window=10,min_count=1,batch_words=4)
    nodes=graphe.nodes()
    embeddings=[model.wv[str(node)] for node in nodes]
    pd_save=pd.DataFrame(embeddings,index=nodes)
    pd_save.to_csv('embeddings32dimensions.csv')
def embedding64dimensions(graphe) :
    node=Node2Vec(graphe,dimensions=64,walk_length=10,num_walks=10,workers=2) 
    model=node.fit(window=10,min_count=1,batch_words=4)
    nodes=graphe.nodes()
    embeddings=[model.wv[str(node)] for node in nodes]
    pd_save=pd.DataFrame(embeddings,index=nodes)
    pd_save.to_csv('embeddings64dimensions.csv')

