import socket
import pickle
import pandas as pd
import numpy as np
import math
from itertools import combinations
import graphlab
import sys
import os
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
from scipy import sparse
import time
from copy import copy
import argparse
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import distance

start1=time.time()
ds=pd.read_csv('diabetic_data_1000_tuples_file.csv')

ds=ds.drop('Unnamed: 0',1)

ds=ds.drop('encounter_id',1)
ds=ds.drop('patient_nbr',1)
ds=ds.dropna(axis=1, how='all')
ds_original=ds.copy()

print ds.isnull().sum().sum()

cat_columns =ds.select_dtypes(exclude=['floating','int64','float64']).columns
ds[cat_columns]=ds[cat_columns].apply(lambda x:x.astype('category'))
ds[cat_columns] = ds[cat_columns].apply(lambda x: x.cat.codes)
ds=ds.fillna(0)
numpyMatrix = ds.as_matrix()
sparseMatrix=sparse.csr_matrix(numpyMatrix)



def generate_random_vectors(num_vector, dim):
    return np.random.randn(dim, num_vector)



def train_lsh(data, num_vector=16, seed=None):
    
    dim = data.shape[1]
    if seed is not None:
        np.random.seed(seed)
    random_vectors = generate_random_vectors(num_vector, dim)
  
    powers_of_two = 1 << np.arange(num_vector-1, -1, -1)
  
    table = {}
    
    # Partition data points into bins
    bin_index_bits = (data.dot(random_vectors) >= 0)
  
    # Encode bin index bits into integers
    bin_indices = bin_index_bits.dot(powers_of_two)
    
    # Update `table` so that `table[i]` is the list of document ids with bin index equal to i.
    for data_index, bin_index in enumerate(bin_indices):
        if bin_index not in table:
            # If no list yet exists for this bin, assign the bin an empty list.
            table[bin_index] = [] # YOUR CODE HERE
        # Fetch the list of document ids associated with the bin and add the document id to the end.
        table[bin_index].append(data_index) # YOUR CODE HERE

    model = {'data': data,
             'bin_index_bits': bin_index_bits,
             'bin_indices': bin_indices,
             'table': table,
             'random_vectors': random_vectors,
             'num_vector': num_vector}
    
    return model




model = train_lsh(sparseMatrix, num_vector=16, seed=143)



def search_nearby_bins(query_bin_bits, table, search_radius=2, initial_candidates=set()):
    """
    For a given query vector and trained LSH model, return all candidate neighbors for
    the query among all bins within the given search radius.
    """
    num_vector = len(query_bin_bits)
    powers_of_two = 1 << np.arange(num_vector-1, -1, -1)
    
    candidate_set = copy(initial_candidates)
    
    for different_bits in combinations(range(num_vector), search_radius):      
    
        alternate_bits = copy(query_bin_bits)
        for i in different_bits:
            alternate_bits[i] = 1-query_bin_bits[i] 
        
        # Convert the new bit vector to an integer index
        nearby_bin = alternate_bits.dot(powers_of_two)
        
        if nearby_bin in table:
            li=table[nearby_bin] # YOUR CODE HERE: Update candidate_set with the documents in this bin.
            candidate_set.update(li)
            
    return candidate_set



def find_memo(x):
    if(x.dtype == np.float64 or x.dtype == np.int64):
        return round(x.mean())
    else:
        ll=len(x.mode())
        if(ll==0):
            return np.nan
        hel=str(x.mode())
        val=hel.split()[1]
        return val


def query(vec, model, k, max_search_radius):
  
    data = model['data']
    table = model['table']
    random_vectors = model['random_vectors']
    num_vector = random_vectors.shape[1]
    
    
    # Compute bin index for the query vector, in bit representation.
    bin_index_bits = (vec.dot(random_vectors) >= 0).flatten()
    
    # Search nearby bins and collect candidates
    candidate_set = set()
    for search_radius in xrange(max_search_radius+1):
        candidate_set = search_nearby_bins(bin_index_bits, table, search_radius, initial_candidates=candidate_set)
    
    # Sort candidates by their true distances from the query
    nearest_neighbors = graphlab.SFrame({'id':candidate_set})
    candidates = data[np.array(list(candidate_set)),:]
    nearest_neighbors['distance'] = pairwise_distances(candidates, vec, metric='cosine').flatten()
    
    nearest_neighbors=nearest_neighbors.topk('distance', k, reverse=True)
    ids=nearest_neighbors['id'][1:]
    selected=ds_original.iloc[ids,:]
    return selected.apply(lambda x:find_memo(x))



def find_nearest(mat):
    my_dataframe=pd.DataFrame(columns=ds_original.columns)
    ll=range(mat.shape[0])
    for i in ll:
        tt=query(mat[i], model, k=10, max_search_radius=3)
        my_dataframe=my_dataframe.append(tt,ignore_index=True)
    return my_dataframe


start=time.time()
ds_original_null=ds_original[ds_original.isnull().any(axis=1)]
ds_original_not_null=ds_original.dropna(axis=0, how='any')
temp=ds_original_null.copy()
temp[cat_columns]=temp[cat_columns].apply(lambda x:x.astype('category'))
temp[cat_columns] = temp[cat_columns].apply(lambda x: x.cat.codes)
temp=temp.fillna(0)
numpyMatrix_null = temp.as_matrix()
sparseMatrix_null=sparse.csr_matrix(numpyMatrix_null)
nearest_tuples=find_nearest(sparseMatrix_null)
ds_original_null=ds_original_null.reset_index()
ds_original_null=ds_original_null.drop('index',axis=1)
ds_original_null=ds_original_null.combine_first(nearest_tuples)
ds=ds_original_null.append(ds_original_not_null)
end=time.time()
print "Total Time Till LSH"
print end-start

print ds.isnull().sum().sum()


i=-1
null_cols=ds.isnull().any()
cols=list(ds.columns)
for col in null_cols:
    i=i+1;
    col_name=cols[i]
    if(col==True):
        if(ds[col_name].dtype == np.float64 or ds[col_name].dtype == np.int64):
            ds[col_name]=ds[col_name].fillna(round(ds[col_name].mean()))
        else:
            ll=len(ds[col_name].mode())
            if(ll!=0):
                hel=str(ds[col_name].mode())
                val=hel.split()[1]
                ds[col_name]=ds[col_name].fillna(val)



ds.to_csv('data_warehouse_diabetic.csv')


target='diabetesMed'


var_col=pd.DataFrame(columns=['var'])
count=0
for col_name in cols:
    if(ds[col_name].dtype == np.float64 or ds[col_name].dtype == np.int64):
        my_var=ds[col_name].var()
        sac=pd.Series([my_var],index=['var'])
        var_col=var_col.append(sac,ignore_index=True)
        count=count+1


var_col.sort_values('var',ascending=False,inplace=True)
tot_num_col=len(var_col)
l=tot_num_col
if(tot_num_col<=8.0):
    thres=var_col.iloc[l-1]
else:
    thres=var_col.iloc[7]



for col_name in cols:
    if(ds[col_name].dtype == np.float64 or ds[col_name].dtype == np.int64):
        my_var=ds[col_name].var()
        if(math.isnan(my_var) or thres.isnull().any()):
            if(col_name!=target):
                ds=ds.drop(col_name,1)
        else:
            diff=int(my_var)-int(thres)
            if(diff<=0):
                if(col_name!=target):
                    ds=ds.drop(col_name,1)



def entropy(vec, base=2):
        vec = np.unique(vec, return_counts=True)
        vec=vec[1]
        prob_vec = np.array(vec/float(vec.sum()))
        if base == 2:
            logfn = np.log2
        elif base == 10:
            logfn = np.log10
        else:
            logfn = np.log  
        return prob_vec.dot(-logfn(prob_vec))


def conditional_entropy(x, y):
        uy, uyc = np.unique(y, return_counts=True)
        prob_uyc = uyc/float(uyc.sum())
        cond_entropy_x = np.array([entropy(x[y == v]) for v in uy])
        return prob_uyc.dot(cond_entropy_x)


def mutual_information(x, y):
        return entropy(x) - conditional_entropy(x, y)


def symmetrical_uncertainty(x, y):
          return 2.0*mutual_information(x, y)/(entropy(x) + entropy(y))



def getFirstElement(d):
        t = np.where(d[:,2]>0)[0]
        if len(t):
            return d[t[0],0], d[t[0],1], t[0]
        return None, None, None

def getNextElement(d, idx):
        t = np.where(d[:,2]>0)[0]
        t = t[t > idx]
        if len(t):
            return d[t[0],0], d[t[0],1], t[0]
        return None, None, None

def removeElement(d, idx):
        d[idx,2] = 0
        return d

def c_correlation(X, y):
        su = np.zeros(X.shape[1])
        for i in np.arange(X.shape[1]):
            su[i] = symmetrical_uncertainty(X[:,i], y)
        return su

def fcbf(X, y, thresh):
        n = X.shape[1]
        slist = np.zeros((n, 3))
        slist[:, -1] = 1
        #identify relevant features
        slist[:,0] = c_correlation(X, y) # compute 'C-correlation'
        idx = slist[:,0].argsort()[::-1]
        slist = slist[idx, ]
        slist[:,1] = idx
        if thresh < 0:
            thresh = np.median(slist[-1,0])
            print "Using minimum SU value as default threshold: {0}".format(thresh)
        elif thresh >= 1 or thresh > max(slist[:,0]):
            print "No relevant features selected for given threshold."
            print "Please lower the threshold and try again."
            exit()
            
        slist = slist[slist[:,0]>thresh,:] # desc. ordered per SU[i,c]
        # identify redundant features among the relevant ones
        cache = {}
        m = len(slist)
        p_su, p, p_idx = getFirstElement(slist)
        for i in xrange(m):
            q_su, q, q_idx = getNextElement(slist, p_idx)
            if q:
                while q:
                    if (p, q) in cache:
                        pq_su = cache[(p,q)]
                    else:
                        p=int(p)
                        q=int(q)
                        pq_su = symmetrical_uncertainty(X[:,p], X[:,q])
                        cache[(p,q)] = pq_su
                        

                    if pq_su >= q_su:
                        slist = removeElement(slist, q_idx)
                    q_su, q, q_idx = getNextElement(slist, q_idx)
                    
            p_su, p, p_idx = getNextElement(slist, p_idx)
            if not p_idx:
                break
        
        sbest = slist[slist[:,2]>0, :2]
        return sbest

def fcbf_wrapper(thresh,classAt=-1):
            d=ds.as_matrix()  
            if classAt == -1:
                X = d[:, :d.shape[1]-1]
                y = d[:,-1]
            else:
                idx = np.arange(d.shape[1])
                X = d[:, idx[idx != classAt]]
                y = d[:, classAt]
            try:
                print "Performing FCBF selection. Please wait ..."
                sbest = fcbf(X, y, thresh)
                print "Done!"
                try:
                    selected_features=sbest[:,1]
                except Exception, e:
                    print "Error encountered while saving file:", e
            except Exception, e:
                print "Error:", e
            return selected_features

st_fcbf=time.time()
tot_cols_now=len(ds.columns)-2
selected_features= np.array([])
selected_features=fcbf_wrapper(0.01,tot_cols_now)
selected_features=np.sort(selected_features)
selected_features=np.append(selected_features,tot_cols_now)

ds=ds.iloc[:,selected_features]
en_fcbf=time.time()
print "Time In FCBF"
print en_fcbf-st_fcbf


ds_sframe=graphlab.SFrame(ds)

ds_sframe[target]=ds_sframe[target].apply(lambda x:'Yes' if (x==1.0 or x==1 or x=='Yes' or x=='ckd') else 'No')
print "Selected Features Are"
print ds_sframe.column_names()

positive= ds_sframe[ds_sframe[target] == 'Yes']
negative= ds_sframe[ds_sframe[target] == 'No']

percentage1 = len(negative)/float(len(positive)+len(negative))
percentage2= len(positive)/float(len(negative)+len(positive))

if percentage1<percentage2:
    percentage=percentage1
    if(percentage==0):
        positive_rows=positive
    else:
        positive_rows = positive.sample(percentage, seed = 1)
    negative_rows = negative
    data = negative_rows.append(positive_rows)
else:
    percentage=percentage2
    if(percentage==0):
        negative_rows=negative
    else:
        negative_rows = negative.sample(percentage, seed = 1)
    positive_rows=positive
    data = negative_rows.append(positive_rows)

features=data.column_names()
features.remove(target)
my_features=[]
my_data=data.copy()

for feature in features:
    if(data[feature].dtype()==str):
        data_one_hot_encoded =data[feature].apply(lambda x: {x: 1})    
        data_unpacked = data_one_hot_encoded.unpack(column_name_prefix=feature)
    
         # Change None's to 0's
        for column in data_unpacked.column_names():
            data_unpacked[column] = data_unpacked[column].fillna(0)

        data.remove_column(feature)
        data.add_columns(data_unpacked)
    else:
        my_features.append(feature)

# Selecting Splliting Point For Numeric Attributes

st_tree=time.time()

def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0
    
    # Count the number of Yes's
    num_s=labels_in_node.apply(lambda x:1 if (x=='Yes') else 0)
    c_num_s=num_s.sum()
    
    # Count the number of No's 
    num_r=labels_in_node.apply(lambda x:1 if (x=='No') else 0)
    c_num_r=num_r.sum()
                
    # Return the number of mistakes that the majority classifier makes.
    if(c_num_s>c_num_r):
        return c_num_r
    else:
        return c_num_s;

def find_split_point(feature):
    values=data[feature]
    values=values.sort()
    values=values.unique()
    val_len=len(values)-1
    
    min_mistakes=2*len(data)+2
    s_point=0
    
    for i in xrange(val_len):
        mid=(values[i]+values[i+1])/2.0
        left_data=data[data[feature]<=mid]
        right_data=data[data[feature]>mid]
        left_mistakes=intermediate_node_num_mistakes(left_data[target])
        right_mistakes=intermediate_node_num_mistakes(right_data[target])
        total_mistakes=left_mistakes+right_mistakes
        
        if(total_mistakes<min_mistakes):
            min_mistakes=total_mistakes
            s_point=mid
    return s_point

features=data.column_names()
features.remove(target)
split_points=[]
tot_cols=features
f_time_s=time.time()

for feature in my_features:
    if(data[feature].dtype()!=str):
        split_point=find_split_point(feature)
        split_points.append(split_point)
        data[feature]=data[feature].apply(lambda x: 0 if (x<=split_point) else 1)
        
f_time_e=time.time()
print "Toatal Time In Finding Split-Points"
print f_time_e-f_time_s
train_data, test_data = data.random_split(.8, seed=1)

def best_splitting_feature(data,features):
    best_feature = None # Keep track of the best feature 
    best_error = 10     # Keep track of the best error so far 

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))  
    
    for feature in features:
        left_split = data[data[feature] == 0]
        right_split =  data[data[feature] == 1] 
        left_mistakes = intermediate_node_num_mistakes(left_split[target])            
        right_mistakes = intermediate_node_num_mistakes(right_split[target])   
        error =  (left_mistakes + right_mistakes) / num_data_points
        if error < best_error:
            best_error=error
            best_feature=feature
    return best_feature 

def create_leaf(target_values):
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True  }
    
    num_ones = len(target_values[target_values == 'Yes'])
    num_minus_ones = len(target_values[target_values =='No'])
    
    # For the leaf node, set the prediction to be the majority class.
    if num_ones > num_minus_ones:
        leaf['prediction'] = 'Yes'        
    else:
        leaf['prediction'] = 'No'
            
    return leaf 

def decision_tree_create(tr_data,features,current_depth = 0, max_depth = 10):
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = tr_data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
    

    # Stopping condition 1
    if(intermediate_node_num_mistakes(target_values))  == 0:  
        print "Stopping condition 1 reached."     
        # If not mistakes at current node, make current node a leaf node
        return create_leaf(target_values)
    
    # Stopping condition 2 (check if there are remaining features to consider splitting on)
    if remaining_features ==0 :
        print "Stopping condition 2 reached."    
        # If there are no remaining features to consider, make current node a leaf node
        return create_leaf(target_values)    
    
    # Additional stopping condition (limit tree depth)
    if current_depth >=max_depth :  ## YOUR CODE HERE
        print "Reached maximum depth. Stopping for now."
        return create_leaf(target_values)

    # Find the best splitting feature
    splitting_feature=best_splitting_feature(tr_data,features)
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature]==0]
    right_split =  data[data[splitting_feature]==1]
    remaining_features.remove(splitting_feature)
    print "Split on feature %s. (%s, %s)" % (\
                      splitting_feature, len(left_split), len(right_split))
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print "Creating leaf node."
        return create_leaf(left_split[target])
    if len(right_split) == len(data):
        print "Creating leaf node."
        return create_leaf(right_split[target])

        
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features,current_depth + 1, max_depth)        
    right_tree = decision_tree_create(right_split, remaining_features,current_depth + 1, max_depth)

    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}

def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])



my_decision_tree=decision_tree_create(train_data, features, max_depth = 6)



def classify(tree, x, annotate = False):   
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate: 
            print "At leaf, predicting %s" % tree['prediction']
        return tree['prediction'] 
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate: 
            print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate) 

def evaluate_classification_error(tree, data, target):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x))
    data['pred']=prediction
    target_values=data[target]
    mistakes=data.apply(lambda x: 1 if(x[target]!=x['pred']) else 0)
    cla_err=mistakes.sum()/float (len(mistakes))
    data.remove_column('pred')
    return target_values,prediction,cla_err

targets,predictions,error=evaluate_classification_error(my_decision_tree, test_data, target)
accuracy=(1-error)*100

print "Accuracy For Binary-Tree"
print accuracy


print "Confusion Matrix For Binary"
print graphlab.evaluation.confusion_matrix(targets, predictions)

tar_numeric=targets.apply(lambda x:1 if (x=='Yes') else 0)
pred_numeric=predictions.apply(lambda x:1 if (x=='Yes') else 0)

roc_mat=graphlab.evaluation.roc_curve(tar_numeric, pred_numeric)

FPR=roc_mat['fpr']

TPR=roc_mat['tpr']

plt.plot(FPR,TPR,'r-',label='Binary Decision Tree Classifier')

en_tree=time.time()
print "Time For Decision Tree"
print en_tree-st_tree

# Naive Bayesian Classifier

st_naive=time.time()
class_names=my_data[target].unique()

prob_class=[]

for clas in class_names:
    prob_class.append(float(len(my_data[my_data[target]==clas]))/float(len(my_data)))

def bayes_classify(x,my_train_data):
    max_prob=-1
    best_class=class_names[0]
    ind=0
    col_names=my_train_data.column_names()
    for clas in class_names:
        prob=prob_class[ind]
        pure_data=my_train_data[my_train_data[target]==clas]
        for col in col_names:
	    if(col==target):
                continue
            if(my_train_data[col].dtype()==str):
                prob=prob*(float(len(pure_data[pure_data[col]==x[col]]))/float(len(pure_data)))
            else:    
                mean=pure_data[col].mean()
                stdev=pure_data[col].std()
                stdev=stdev+0.1
                exponent = math.exp(-(math.pow(x[col]-mean,2)/(2*math.pow(stdev,2))))
                prob=prob*(1 / (math.sqrt(2*math.pi)) * stdev) * exponent
        
        if(prob>max_prob):
            max_prob=prob
            best_class=clas
        ind=ind+1
    return best_class

def evaluate_bayes_classifier(my_test_data,my_train_data):
    prediction = my_test_data.apply(lambda x: bayes_classify(x,my_train_data))
    my_test_data['pred']=prediction
    target_values=my_test_data[target]
    mistakes=my_test_data.apply(lambda x: 1 if(x[target]!=x['pred']) else 0)
    cla_err=mistakes.sum()/float (len(mistakes))
    my_test_data.remove_column('pred')
    return target_values,prediction,cla_err

my_train_data, my_test_data = my_data.random_split(.8, seed=1)
targets,predictions,error=evaluate_bayes_classifier(my_test_data,my_train_data)
accuracy=(1-error)*100

print "Accuracy For Naive Bayesian "
print accuracy

print "Confusion Matrix For Naive"
print graphlab.evaluation.confusion_matrix(targets, predictions)



tar_numeric=targets.apply(lambda x:1 if (x=='Yes') else 0)
pred_numeric=predictions.apply(lambda x:1 if (x=='Yes') else 0)

roc_mat=graphlab.evaluation.roc_curve(tar_numeric, pred_numeric)

FPR=roc_mat['fpr']
TPR=roc_mat['tpr']

plt.plot(FPR,TPR,'g-',label='Naive Bayesian Classifier')

en_naive=time.time()
print "Time In Naive"
print en_naive-st_naive

st_boost=time.time()

# Using Boosting To Increase Performance Of Binary Decidion Tree

def intermediate_node_weighted_mistakes(labels_in_node, data_weights):
    # Sum the weights of all entries with label +1
    total_weight_positive = data_weights[labels_in_node == 'Yes'].sum()
    
    # Weight of mistakes for predicting all -1's is equal to the sum above
    weighted_mistakes_all_negative = total_weight_positive
    
    # Sum the weights of all entries with label -1
    total_weight_negative = data_weights[labels_in_node == 'No'].sum()
    
    # Weight of mistakes for predicting all +1's is equal to the sum above
    weighted_mistakes_all_positive = total_weight_negative
    
    # Return the tuple (weight, class_label) representing the lower of the two weights
    # If the two weights are identical, return (weighted_mistakes_all_positive,+1)
    if(weighted_mistakes_all_negative<weighted_mistakes_all_positive):
        return (weighted_mistakes_all_negative,'No')
    else:
        return (weighted_mistakes_all_positive,'Yes')

def best_splitting_feature_weighted(data, features, target, data_weights):
    
    # These variables will keep track of the best feature and the corresponding error
    best_feature = None
    best_error = float('+inf') 
    num_points = float(len(data))
    data['weights']=data_weights

    # Loop through each feature to consider splitting on that feature
    for feature in features:
        
        left_split = data[data[feature] == 0]
        right_split = data[data[feature] == 1]
        
        # Apply the same filtering to data_weights to create left_data_weights, right_data_weights
        left_data_weights = left_split['weights']
        right_data_weights = right_split['weights']
                    
        # Calculate the weight of mistakes for left and right sides
        left_weighted_mistakes, left_class = intermediate_node_weighted_mistakes(left_split[target], left_data_weights)
        right_weighted_mistakes, right_class = intermediate_node_weighted_mistakes(right_split[target], right_data_weights)
        
        error = ( left_weighted_mistakes + right_weighted_mistakes ) / data_weights.sum()
        
        # If this is the best error we have found so far, store the feature and the error
        if error < best_error:
            best_feature = feature
            best_error = error
    
    # Return the best feature we found
    return best_feature

def create_leaf_weighted(target_values, data_weights):
    
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'is_leaf': True}
    
    # Computed weight of mistakes.
    weighted_error, best_class = intermediate_node_weighted_mistakes(target_values, data_weights)
    leaf['prediction'] = best_class 
    return leaf 

def weighted_decision_tree_create(data, features, target, data_weights, current_depth = 1, max_depth = 6):
    remaining_features = features[:] # Make a copy of the features.
    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
    
    # Stopping condition 1. Error is 0.
    if intermediate_node_weighted_mistakes(target_values, data_weights)[0] <= 1e-15:
        print "Stopping condition 1 reached."                
        return create_leaf_weighted(target_values, data_weights)
    
    # Stopping condition 2. No more features.
    if remaining_features == []:
        print "Stopping condition 2 reached."                
        return create_leaf_weighted(target_values, data_weights)    
    
    # Additional stopping condition (limit tree depth)
    if current_depth > max_depth:
        print "Reached maximum depth. Stopping for now."
        return create_leaf_weighted(target_values, data_weights)
    
    splitting_feature = best_splitting_feature_weighted(data, features, target, data_weights)
    remaining_features.remove(splitting_feature)
        
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    
    left_data_weights = data_weights[data[splitting_feature] == 0]
    right_data_weights = data_weights[data[splitting_feature] == 1]
    
    print "Split on feature %s. (%s, %s)" % (\
              splitting_feature, len(left_split), len(right_split))
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print "Creating leaf node."
        return create_leaf_weighted(left_split[target], data_weights)
    if len(right_split) == len(data):
        print "Creating leaf node."
        return create_leaf_weighted(right_split[target], data_weights)
    
    # Repeat (recurse) on left and right subtrees
    left_tree = weighted_decision_tree_create(
        left_split, remaining_features, target, left_data_weights, current_depth + 1, max_depth)
    right_tree = weighted_decision_tree_create(
        right_split, remaining_features, target, right_data_weights, current_depth + 1, max_depth)
    
    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}

def classify_weighted(tree, x, annotate = False):   
    # If the node is a leaf node.
    if tree['is_leaf']:
        if annotate: 
            print "At leaf, predicting %s" % tree['prediction']
        return tree['prediction'] 
    else:
        # Split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate: 
            print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)

def evaluate_classification_error(tree, data):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify_weighted(tree, x))
    
    # Once you've made the predictions, calculate the classification error
    return (prediction != data[target]).sum() / float(len(data))

example_data_weights = graphlab.SArray([1.0 for i in range(len(train_data))])
small_data_decision_tree=weighted_decision_tree_create(train_data,features,target,example_data_weights)
err=evaluate_classification_error(small_data_decision_tree, test_data)
acc=(1-err)*100
print "Accuracy For AdaBoost"
print acc

def adaboost_with_tree_stumps(data, features, target, num_tree_stumps):
    # start with unweighted data
    alpha = graphlab.SArray([1.]*len(data))
    weights = []
    tree_stumps = []
    target_values = data[target]
    
    for t in xrange(num_tree_stumps):
        print '====================================================='
        print 'Adaboost Iteration %d' % t
        print '====================================================='        
        # Learn a weighted decision tree stump. Use max_depth=1
        tree_stump = weighted_decision_tree_create(data, features, target, data_weights=alpha, max_depth=6)
        tree_stumps.append(tree_stump)
        
        # Make predictions
        predictions = data.apply(lambda x: classify_weighted(tree_stump, x))
        
        # Produce a Boolean array indicating whether
        # each data point was correctly classified
        is_correct = predictions == target_values
        is_wrong   = predictions != target_values
        
	temp=data['weights'][is_correct==0]
	
        # Compute weighted error
        #split_feature =tree_stump['splitting_feature']
        #left_split = data[data[split_feature] == 0]
        #right_split = data[data[split_feature] == 1]
        
        
        #left_data_weights = left_split['weights']
        #right_data_weights = right_split['weights']
        
        #left_weighted_mistakes, left_class = intermediate_node_weighted_mistakes(left_split[target], left_data_weights)
        #right_weighted_mistakes, right_class = intermediate_node_weighted_mistakes(right_split[target], right_data_weights)
        
        
        weighted_error=temp.sum()/data['weights'].sum()
        # Compute model coefficient using weighted error
        weight = math.log((1-weighted_error)/weighted_error)/2.0
        weights.append(weight)
        
        # Adjust weights on data point
        adjustment = is_correct.apply(lambda is_correct : math.exp(-weight) if is_correct else math.exp(weight))
        
        # Scale alpha by multiplying by adjustment 
        # Then normalize data points weights
        data_weights=data['weights']*adjustment
        su=data_weights.sum()
        data_weights=data_weights.apply(lambda x:x/su)
        alpha=data_weights
    
    return weights, tree_stumps

def predict_adaboost(stump_weights, tree_stumps, data):
    scores = graphlab.SArray([0.]*len(data))
    
    for i, tree_stump in enumerate(tree_stumps):
        predictions = data.apply(lambda x: classify_weighted(tree_stump, x))
        
        # Accumulate predictions on scores array
        data['pred']=predictions
        data['pred']=data['pred'].apply(lambda x: +1 if x=='Yes' else -1)
        predictions=data['pred']
        temp=stump_weights[i]*predictions
        scores=scores+temp
        data.remove_column('pred')
    return scores.apply(lambda score : 'Yes' if score > 0.0 else 'No')

stump_weights, tree_stumps = adaboost_with_tree_stumps(train_data,features, target, num_tree_stumps=3)
predictions = predict_adaboost(stump_weights, tree_stumps, test_data)
err=(predictions != test_data[target]).sum() / float(len(test_data))
accuracy=(1-err)*100
print "Accuracy For Adaboost with tree stumps"
print accuracy


print "Confusion Matrix For Adaboost"
print graphlab.evaluation.confusion_matrix(test_data[target], predictions)

targets=test_data[target]
tar_numeric=targets.apply(lambda x:1 if (x=='Yes') else 0)
pred_numeric=predictions.apply(lambda x:1 if (x=='Yes') else 0)

roc_mat=graphlab.evaluation.roc_curve(tar_numeric, pred_numeric)

FPR=roc_mat['fpr']
TPR=roc_mat['tpr']

en_boost=time.time()
print "Time In AdaBoost"
print en_boost-st_boost

plt.plot(FPR,TPR,'m-',label='AdaBoost Classifier')
plt.legend()
plt.show()


warehouse=pd.read_csv('data_warehouse_diabetic.csv')

warehouse=warehouse.drop('Unnamed: 0',1)

def curr_areas(mm,yy,ss):
	if(ss=='All'):
		sel_set=warehouse[(warehouse['year']==int(yy)) & (warehouse['month']==int(mm)) & (warehouse[target]=='Yes')]
		if(sel_set.shape[0]==0):
			return "No Area Is Currently Affected"
		else:
			dists=sel_set['district'].unique()
			freq=[]
			for dist in dists:
				freq.append(sel_set[sel_set['district']==dist].shape[0])

			Z = [x for _,x in sorted(zip(freq,dists))]
			return Z
	elif(ss=='Madhya Pradesh'):
		sel_set=warehouse[(warehouse['year']==int(yy)) & (warehouse['month']==int(mm)) 
				& (warehouse[target]=='Yes') & (warehouse['state']=='Madhya Pradesh')]
		if(sel_set.shape[0]==0):
			return "No Area Is Currently Affected"
		else:
			dists=sel_set['district'].unique()
			freq=[]
			for dist in dists:
				freq.append(sel_set[sel_set['district']==dist].shape[0])

			Z = [x for _,x in sorted(zip(freq,dists))]
			return Z
	elif(ss=='Uttar Pradesh'):
		sel_set=warehouse[(warehouse['year']==int(yy)) & (warehouse['month']==int(mm)) 
				& (warehouse[target]=='Yes') & (warehouse['state']=='Uttar Pradesh')]
		if(sel_set.shape[0]==0):
			return "No Area Is Currently Affected"
		else:
			dists=sel_set['district'].unique()
			freq=[]
			for dist in dists:
				freq.append(sel_set[sel_set['district']==dist].shape[0])

			Z = [x for _,x in sorted(zip(freq,dists))]
			return Z
	elif(ss=='Rajasthan'):
		sel_set=warehouse[(warehouse['year']==int(yy)) & (warehouse['month']==int(mm)) 
				& (warehouse[target]=='Yes') & (warehouse['state']=='Rajasthan')]
		if(sel_set.shape[0]==0):
			return "No Area Is Currently Affected"
		else:
			dists=sel_set['district'].unique()
			freq=[]
			for dist in dists:
				freq.append(sel_set[sel_set['district']==dist].shape[0])

			Z = [x for _,x in sorted(zip(freq,dists))]
			return Z

def starting_location():
	locations={'Ajmer':'Rajasthan','Alwar':'Rajasthan','Bharatpur':'Rajasthan',
			'Banswara':'Rajasthan','Baran':'Rajasthan','Barmer':'Rajasthan','Hanumangarh':'Rajasthan','Sri Ganganagar':'Rajasthan',
			'Bhilwara':'Rajasthan','Bikaner':'Rajasthan',
			'Bundi':'Rajasthan','Chittorgarh':'Rajasthan','Churu':'Rajasthan',
			'Dausa':'Rajasthan','Dhaulpur':'Rajasthan','Dungarpur':'Rajasthan','Jaipur':'Rajasthan',
			'Jaisalmer':'Rajasthan','Jalor':'Rajasthan','Jhalawar':'Rajasthan','Jhunjhunun':'Rajasthan',
			'Jodhpur':'Rajasthan','Karauli':'Rajasthan','Kota':'Rajasthan','Nagaur':'Rajasthan',
			'Pali':'Rajasthan','Pratapgarh':'Rajasthan','Rajsamand':'Rajasthan','Sawai Madhopur':'Rajasthan',
			'Sikar':'Rajasthan','Sirohi':'Rajasthan','Tonk':'Rajasthan','Udaipur':'Rajasthan',
			'Agra':'Uttar Pradesh','Mathura':'Uttar Pradesh','Aligarh':'Uttar Pradesh','Allahabaad':'Uttar Pradesh',
			'Azamgarh':'Uttar Pradesh','Shahjahanpur':'Uttar Pradesh','Basti':'Uttar Pradesh','Amethi':'Uttar Pradesh',
			'Gorakhpur':'Uttar Pradesh','Lucknow':'Uttar Pradesh','Gwalior':'Madhya Pradesh',
			'Guna':'Madhya Pradesh','Indore':'Madhya Pradesh','Bhopal':'Madhya Pradesh',
			'Bhind':'Madhya Pradesh','Balaghat':'Madhya Pradesh','Ujjain':'Madhya Pradesh',
			'Shivpuri':'Madhya Pradesh','Satna':'Madhya Pradesh','Ratlam':'Madhya Pradesh'
			}
	years=warehouse['year'].unique()
	months=warehouse['month'].unique()
	
	years.sort()
	months.sort()
	l=0
	for y in years:
		for m in months:
			selected_set=warehouse[(warehouse['year']==y) & (warehouse['month']==m) & (warehouse[target]=='Yes')]
			if(selected_set.shape[0]==0):
				continue
			else:
				sel_dis=selected_set['district'].unique()
				X=[]
				Y=[]
				for dis in sel_dis:
					temp=selected_set[selected_set['district']==dis].shape[0]
					Y.append(temp)
					lo=dis+' ('+locations[dis]+')'
					X.append(lo)
					l=l+1
				return [Y,X,l]
	return [-1,-1,-1]

def find_distribution(my_attributes,my_values,data):
    count=0
    temp=data
    for att in my_attributes:
        if(data[att].dtype==np.float64 or data[att].dtype==np.int64):
            l=my_values[count][0]
            r=my_values[count][1]
            temp=temp.loc[(temp[att]>=l) & (temp[att]<=r)]
        else:
            temp=temp.loc[temp[att]==my_values[count]]
        count=count+1
    l=temp.shape[0]
    if(l==0):
        return 0
    temp=temp.loc[temp[target]=='Yes']
    s=temp.shape[0]
    return float(s)/float(l)



def find_trends(att,data):
    values=data[att].unique()
    y=[]
    l=len(values)
    if(data[att].dtype==np.float64 or data[att].dtype==np.int64):
        temp=[]
        values.sort()
        if(l<=10):
            i=0
            while(i<l):
                temp.append([values[i],values[i]])
                i=i+1
        elif(l>10 and l<=100):
            i=0
            while(i<l):
                if((i+9)<l):
                    temp.append([values[i],values[i+9]])
                else:
                    temp.append([values[i],values[l-1]])
                i=i+10
        elif(l>100 and l<=1000):
            i=0
            while(i<l):
                if((i+50)<l):
                    temp.append([values[i],values[i+50]])
                else:
                    temp.append([values[i],values[l-1]])
                i=i+50
        else:
            i=0
            while(i<l):
                if((i+300)<l):
                    temp.append([values[i],values[i+300]])
                else:
                    temp.append([values[i],values[l-1]])
                i=i+300
        values=temp
    l=len(values)
    for i in xrange(l):
        temp=find_distribution([att],[values[i]],data)
        y.append(temp)
    return [y,values,l]



end1=time.time()
print "Total Time Taken"
print end1-start1

print "Total Data Size For Diabetic"
print data.shape[0]
print "Training Data Size"
print train_data.shape[0]
print "Test Data Size"
print test_data.shape[0]


cols=data.column_names()
def find_frame():
	ex_frame=data[0:1]
	for col in cols:
		ex_frame[col]=0
	
	return ex_frame


def convert_into_binary(data,features):
    for feature in features:
        if(data[feature].dtype()==str):
            data_one_hot_encoded =data[feature].apply(lambda x: {x: 1})    
            data_unpacked = data_one_hot_encoded.unpack(column_name_prefix=feature)

             # Change None's to 0's
            for column in data_unpacked.column_names():
                data_unpacked[column] = data_unpacked[column].fillna(0)

            data.remove_column(feature)
            data.add_columns(data_unpacked)
    return data

s = socket.socket()         # Create a socket object
 # Get local machine name
port = 12345                # Reserve a port for your service.
s.bind(('', port))        # Bind to the port

s.listen(5)                 # Now wait for client connection.
while True:
   c, addr = s.accept()     # Establish connection with client.
   print 'Got connection from', addr
   c.send('Thank you for connecting')
   

   rec=c.recv(1024)
   if(rec=='Load Map District'):
	[Y,X,l]=find_trends('district',warehouse)
	distribution_list=[x for (y,x) in sorted(zip(Y,X))]
        data1=pickle.dumps(distribution_list)
	c.send(data1)
	continue
   elif(rec=='Load Map State'):
	[Y,X,l]=find_trends('state',warehouse)
	distribution_list=[x for (y,x) in sorted(zip(Y,X))]
        data1=pickle.dumps(distribution_list)
	c.send(data1)
        continue
   elif(rec=='column distribution'):
	c.send('ok')
	cc=c.recv(1024)
	print cc
	[Y,X,l]=find_trends(cc,warehouse)
	data1=pickle.dumps([Y,X,l])
	c.send(data1)
	continue

   elif(rec=='starting location'):
	[Y,X,l]=starting_location()
	if(l==-1):
		c.send("This Disease Is Not Found Yet Anywhere")
	else:
		data1=pickle.dumps([Y,X,l])
		c.send(data1)
		continue

   elif(rec=='currently affected areas'):
	c.send('ok')
	mm=c.recv(1024)
	c.send('ok')
	yy=c.recv(1024)
	c.send('ok')
	ss=c.recv(1024)
	X=curr_areas(mm,yy,ss);
	data1=pickle.dumps(X)
	c.send(data1)
	continue
   elif(rec=='timeline_year'):
	c.send('ok')
	yy=c.recv(1024)
	print "Year Is"
	print yy
	sel_ds=warehouse[(warehouse['year']==int(yy)) & (warehouse[target]=='Yes')]
	X=sel_ds['month'].unique()
	Y=[]
	l=len(X)
	for y in X:
		temp=sel_ds[sel_ds['month']==y]
		Y.append(str(temp.shape[0]))

	data1=pickle.dumps([Y,X,l])
	c.send(data1)
	continue
   values=pickle.loads(rec)
   print values
   data_list=[]
   data_list.append(values)
   ds_values=graphlab.SFrame(data_list)
   res_ds_for_naive = ds_values.unpack('X1','')
   features=res_ds_for_naive.column_names()
   intfeatures=[]

   for feature in features:
	if(res_ds_for_naive[feature].dtype()!=str):
		intfeatures.append(feature)

   res_ds_for_tree=convert_into_binary(res_ds_for_naive,features)
   features=res_ds_for_tree.column_names()
   cnt=0
   for feature in intfeatures:
       if(res_ds_for_tree[feature].dtype()!=str):
           split_point=split_points[cnt]
           cnt=cnt+1
           res_ds_for_tree[feature]=res_ds_for_tree[feature].apply(lambda x: 0 if (x<=split_point) else 1)
   
   print res_ds_for_tree
   sel_cols=res_ds_for_tree.column_names()
   rem_cols=list(set(tot_cols)-set(sel_cols))
   ex_frame=find_frame()
   final_sframe_tree=ex_frame.select_columns(rem_cols)
   final_sframe_tree=final_sframe_tree.add_columns(res_ds_for_tree)
   print "Sachin Sharma"
   c.send(classify(my_decision_tree,final_sframe_tree[0]))









