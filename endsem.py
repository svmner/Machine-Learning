#!/usr/bin/env python
# coding: utf-8

# In[69]:


#Importing Dataset

import pandas as pd
import numpy as np 

df = pd.read_csv("decision.csv")
df


# # ID3

# In[71]:


import pprint

def find_entropy(df):
    Class = df.keys()[-1]
    values = df[Class].unique()
    entropy = 0
    for value in values:
        prob = df[Class].value_counts()[value]/len(df[Class])
        entropy += -prob * np.log2(prob)
    return float(entropy)
        
def find_entropy_attribute(df,attribute):
    Class = df.keys()[-1]
    target_values = df[Class].unique()
    attribute_values = df[attribute].unique()
    avg_entropy = 0
    
    for value in attribute_values:
        entropy = 0
        for value1 in target_values:
            num = len(df[attribute][df[attribute] == value][df[Class] == value1])
            den = len(df[attribute][df[attribute] == value])
            prob = num/den
            entropy += -prob * np.log2(prob + 0.000001)
        avg_entropy += (den/len(df))*entropy
    return float(avg_entropy)

def find_winner(df):
    IG = []
    for key in df.keys()[:-1]:
        IG.append(find_entropy(df) - find_entropy_attribute(df,key))
    max_IG_index = IG.index(max(IG))
    return df.keys()[:-1][max_IG_index]

def get_subtable(df,attribute,value):
    return df[df[attribute] == value].reset_index(drop = True)

def buildtree(df,tree = None):
    node = find_winner(df)
    attvalue = np.unique(df[node])
    Class = df.keys()[-1]
    if tree is None:
        tree = {}
        tree[node] = {}
    
    for value in attvalue:
        subtable = get_subtable(df,node,value)
        Clvalue,counts = np.unique(subtable[Class],return_counts = True)
        if len(counts) == 1:
            tree[node][value] = Clvalue[0]
        else:
            tree[node][value] = buildtree(subtable)
    
    return tree

tree = buildtree(df)
pprint.pprint(tree)

test= {'Outlook':'Overcast','Temperature':'Hot','Humidity':'High','Wind':'Weak'}

def predict(test,tree,default = None):
    attribute = next(iter(tree))
    if test[attribute] in tree[attribute].keys():
        result = tree[attribute][test[attribute]]
        if isinstance(result,dict):
            return predict(test,result)
        else:
            return result
    else:
        return default
    
ans = predict(test,tree)
print(f'Decision to be made is : {ans}')


# # CART (Using Gini formula instead of entropy)

# In[68]:


import pandas as pd
import numpy as np
import pprint

def find_gini(df):
    Class = df.keys()[-1]
    values = df[Class].unique()
    gini = 1.0
    for value in values:f
        prob = df[Class].value_counts()[value]/len(df[Class])
        gini -= prob**2
    return gini
        
def find_gini_attribute(df, attribute):
    Class = df.keys()[-1]
    target_values = df[Class].unique()
    attribute_values = df[attribute].unique()
    weighted_gini = 0
    
    for value in attribute_values:
        gini = 1.0
        for target_value in target_values:
            num = len(df[attribute][df[attribute] == value][df[Class] == target_value])
            den = len(df[attribute][df[attribute] == value])
            prob = num / (den + 0.000001)  # To avoid division by zero
            gini -= prob**2
        weighted_gini += (den / len(df)) * gini
        
    return weighted_gini

def find_winner(df):
    Gini_gain = []
    for key in df.keys()[:-1]:
        Gini_gain.append(find_gini(df) - find_gini_attribute(df, key))
    max_Gini_index = Gini_gain.index(max(Gini_gain))
    return df.keys()[:-1][max_Gini_index]

def get_subtable(df,attribute,value):
    return df[df[attribute] == value].reset_index(drop = True)

def buildtree(df,tree = None):
    node = find_winner(df)
    attvalue = np.unique(df[node])
    Class = df.keys()[-1]
    if tree is None:
        tree = {}
        tree[node] = {}
    
    for value in attvalue:
        subtable = get_subtable(df,node,value)
        Clvalue,counts = np.unique(subtable[Class],return_counts = True)
        if len(counts) == 1:
            tree[node][value] = Clvalue[0]
        else:
            tree[node][value] = buildtree(subtable)
    
    return tree

# Build and print the tree
tree = buildtree(df)
pprint.pprint(tree)

# Predict the decision for the test instance
test = {'Outlook':'Overcast', 'Temperature':'Hot', 'Humidity':'High', 'Wind':'Weak'}
ans = predict(test, tree)
print(f'Decision to be made is: {ans}')


# # C4.5(ID3 + Gain Ratio and find_winner codeblock edited)

# In[70]:


import pprint

def find_entropy(df):
    Class = df.keys()[-1]
    values = df[Class].unique()
    entropy = 0
    for value in values:
        prob = df[Class].value_counts()[value]/len(df[Class])
        entropy += -prob * np.log2(prob)
    return float(entropy)
        
def find_entropy_attribute(df,attribute):
    Class = df.keys()[-1]
    target_values = df[Class].unique()
    attribute_values = df[attribute].unique()
    avg_entropy = 0
    
    for value in attribute_values:
        entropy = 0
        for value1 in target_values:
            num = len(df[attribute][df[attribute] == value][df[Class] == value1])
            den = len(df[attribute][df[attribute] == value])
            prob = num/den
            entropy += -prob * np.log2(prob + 0.000001)
        avg_entropy += (den/len(df))*entropy
    return float(avg_entropy)

def find_gain_ratio(df, attribute):
    avg_entropy = find_entropy_attribute(df, attribute)
    intrinsic_value = 0
    attribute_values = df[attribute].unique()
    
    for value in attribute_values:
        subset_proportion = len(df[df[attribute] == value]) / len(df)
        intrinsic_value -= subset_proportion   * np.log2(subset_proportion)
        
    gain = find_entropy(df) - avg_entropy
    gain_ratio = gain / intrinsic_value if intrinsic_value != 0 else 0
    
    return gain_ratio

def find_winner(df):
    gain_ratios = []
    for key in df.keys()[:-1]:
        gain_ratios.append(find_gain_ratio(df, key))
    max_gain_ratio_index = gain_ratios.index(max(gain_ratios))
    return df.keys()[:-1][max_gain_ratio_index]


def get_subtable(df,attribute,value):
    return df[df[attribute] == value].reset_index(drop = True)

def buildtree(df,tree = None):
    node = find_winner(df)
    attvalue = np.unique(df[node])
    Class = df.keys()[-1]
    if tree is None:
        tree = {}
        tree[node] = {}
    
    for value in attvalue:
        subtable = get_subtable(df,node,value)
        Clvalue,counts = np.unique(subtable[Class],return_counts = True)
        if len(counts) == 1:
            tree[node][value] = Clvalue[0]
        else:
            tree[node][value] = buildtree(subtable)
    
    return tree

tree = buildtree(df)
pprint.pprint(tree)

test= {'Outlook':'Overcast','Temperature':'Hot','Humidity':'High','Wind':'Weak'}

def predict(test,tree,default = None):
    attribute = next(iter(tree))
    if test[attribute] in tree[attribute].keys():
        result = tree[attribute][test[attribute]]
        if isinstance(result,dict):
            return predict(test,result)
        else:
            return result
    else:
        return default
    
ans = predict(test,tree)
print(f'Decision to be made is : {ans}')


# In[ ]:




