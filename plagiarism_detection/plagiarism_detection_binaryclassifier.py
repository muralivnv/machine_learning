# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import subprocess
import time

# %% [markdown]
# ## Defining Features
# ### containment
# 
# ngram, a sequential group of n-words  
# $$ containment = \frac{\sum intersection(count(ngram_A), count(ngram_B))}{\sum(count(ngram_A))} $$
# 
# ### longest common subsequence
# 
# 

# %%
from sklearn.feature_extraction.text import CountVectorizer

a_text = "This is an answer text"
s_text = "This is a source text"

# set n
n = 2
# instantiate an ngram counter
counts = CountVectorizer(analyzer='word', ngram_range=(n,n))

# create a dictionary of n-grams by calling `.fit`
vocab2int = counts.fit([a_text, s_text]).vocabulary_

# print dictionary of words:index
print(vocab2int)

ngrams = counts.fit_transform([a_text, s_text])
ngrams_array = ngrams.toarray()

occurances_atext = np.where(ngrams_array[0, :] == 1)[0]
intersection_text = np.where(ngrams_array[1,occurances_atext]==1)[0]
containment_percentage = intersection_text.size/occurances_atext.size
print(containment_percentage)

# %% [markdown]
# ### Four categories of plagiarism 
# 
# Each text file has an associated plagiarism label/category:
# 
# 1. `cut`: An answer is plagiarized; it is copy-pasted directly from the relevant Wikipedia source text.
# 2. `light`: An answer is plagiarized; it is based on the Wikipedia source text and includes some copying and paraphrasing.
# 3. `heavy`: An answer is plagiarized; it is based on the Wikipedia source text but expressed using different words and structure. Since this doesn't copy directly from a source text, this will likely be the most challenging kind of plagiarism to detect.
# 4. `non`: An answer is not plagiarized; the Wikipedia source text is not used to create this answer.
# 5. `orig`: This is a specific category for the original, Wikipedia source text. We will use these files only for comparison purposes.
# 

# %%
file_info = 'data/plagiarism_detection_data/file_information.csv'
plagiarism_df = pd.read_csv(file_info)
plagiarism_df.head(10)


# %%
# task can be think of as different questions
# g0A -> question set 0 person A
n_tasks      = len(plagiarism_df['Task'].unique())
n_categories = len(plagiarism_df['Category'].unique())
n_text_files = plagiarism_df['File'].count()

print("num text files", n_text_files)
print("unique tasks", n_tasks)
print("unique categories", n_categories)


# %%
# create stats about the data
category_dict = {'cut': 0, 'light': 1, 'heavy': 2, 'non': 3, 'orig': 4}
def convert_category_to_int(category):
    global category_dict
    category_int = -1
    if category in category_dict:
        category_int = category_dict[category]

    return category_int

plagiarism_df['Category_int'] = plagiarism_df['Category'].apply(convert_category_to_int)
plagiarism_df.head(5)


# %%
# plot histogram of each category to see the amount of data available across each category
ax = plagiarism_df['Category_int'].plot(kind='hist')
ax.set_xlabel('Categories')
ax.set_title('Histogram of categories')

legend_list = []
for key, value in category_dict.items():
    legend_list.append(key+':%d'%(value))
ax.text(0.5, -10.0, legend_list)

# %% [markdown]
# ## Extracting Features
# As containment and longest common subsequence is selected as features, extracting those information from each of the available text file

# %%
def longest_common_subsequence(left_seq:str, right_seq:str):
    # convert all elements to lower case letters
    left_seq  = left_seq.lower()
    right_seq = right_seq.lower()
    n_cols    = len(left_seq) + 1
    n_rows    = len(right_seq) + 1
    lcs_array = np.zeros((n_rows, n_cols), dtype=np.int16)

    for left_seq_iter in range(1, n_cols):
        for right_seq_iter in range(1, n_rows):
            if (left_seq[left_seq_iter-1] == right_seq[right_seq_iter-1]):
                lcs_array[right_seq_iter][left_seq_iter] = 1 + lcs_array[right_seq_iter-1][left_seq_iter-1]
            else:
                lcs_array[right_seq_iter][left_seq_iter] = max(lcs_array[right_seq_iter-1][left_seq_iter], lcs_array[right_seq_iter][left_seq_iter-1])
    return lcs_array[-1][-1]

# another feature can be containment
def containment(ngrams_array):
    occurances_atext = np.where(ngrams_array[0, :] == 1)[0]
    intersection_text = np.where(ngrams_array[1,occurances_atext] == 1)[0]
    containment_percentage = intersection_text.size/occurances_atext.size

    return containment_percentage


# %%
data_prefix_path = 'data/plagiarism_detection_data'

# start with 2 n-grams
n_grams     = 2
ngram_count = CountVectorizer(analyzer='word', ngram_range=(n_grams, n_grams), 
                              decode_error='ignore')

source_text_indices = plagiarism_df['File'].str.contains('orig')
source_text_df      = plagiarism_df.loc[source_text_indices]
plagiarism_text_df  = plagiarism_df.loc[~source_text_indices]

source_text_dict    = dict()
for row in source_text_df.iterrows():
    source_text_dict[row[1]['Task']] = row[1]['File']

# initialize Containment and LCS columns to 0.0
plagiarism_text_df['containment'] = 0.0
plagiarism_text_df['lcs']         = 0.0

start_time = time.time()
for row in plagiarism_text_df.itertuples():
    test_taskid      = row.Task
    test_filename    = '{}/{}'.format(data_prefix_path, row.File)
    source_filename  = '{}/{}'.format(data_prefix_path, source_text_dict[test_taskid])

    # load test file and source file
    test_file = open(test_filename, 'rb')
    source_file = open(source_filename, 'rb')

    test_file_contents   = test_file.readlines()
    source_file_contents = source_file.readlines()

    test_file.close()
    source_file.close()

    # join lines together to form one continuous string   
    test_file_contents = b''.join(test_file_contents)
    source_file_contents = b''.join(source_file_contents)

    # calculate containment
    calculated_ngrams       = ngram_count.fit_transform([test_file_contents, source_file_contents])
    calculated_ngrams_array = calculated_ngrams.toarray()
    test_file_containment   = containment(calculated_ngrams_array)

    # calculate longest common subsequence
    # test_file_lcs_len_py = longest_common_subsequence(test_file_contents, source_file_contents)
    proc = subprocess.Popen(["longest_common_subsequence.cpp.exe", 
                             test_filename, 
                              source_filename], 
                              stdout=subprocess.PIPE)
    output = b'0' + proc.communicate()[0]

    test_file_lcs_len = int(output)
    lcs_percentage    = test_file_lcs_len / len(test_file_contents)

    # store it inside dataframe
    plagiarism_text_df['containment'].iloc[row.Index] = test_file_containment
    plagiarism_text_df['lcs'].iloc[row.Index] = lcs_percentage

end_time = time.time()

print("time elapsed: %f sec"%(end_time - start_time))
print("parsed features")


# %%
plagiarism_text_df.head(10)

# calculate percentage of non-plagiarized class
non_plagiarised_data_len = plagiarism_text_df[plagiarism_text_df['Category_int'] == 0].count()
print("non-plagiarized data percentage %f"%(non_plagiarised_data_len['Category_int']/ len(plagiarism_text_df)))

# %%
# now use containment and lcs as features and perform binary classifier
features = plagiarism_text_df[['containment', 'lcs']].to_numpy()
labels   = plagiarism_text_df['Category_int'].to_numpy()

# convert labels to plagiarised [1] and non-plagiarised [0]
labels   = np.where(labels == category_dict['non'], 0, 1)


# %%
# helper functions
def train_test_split(target_features, target_labels, train_frac=0.7, seed=1):
    
    # seed random number generator
    np.random.seed(seed)
    
    # get shuffled indices upto data length
    shuffled_indices = np.random.permutation(target_features.shape[0])
    
    train_data_len = int(train_frac*target_features.shape[0])
    
    train_features = target_features[shuffled_indices[0:train_data_len], :]
    train_labels   = target_labels[shuffled_indices[0:train_data_len]]
    
    test_features = target_features[shuffled_indices[train_data_len: ], :]
    test_labels = target_labels[shuffled_indices[train_data_len: ]]
    
    return (train_features, train_labels), (test_features, test_labels)

# helper functions to evaluate metrics
def evaluate(test_preds, test_labels, verbose=False):
       
    # calculate true positives, false positives, true negatives, false negatives
    tp = np.logical_and(test_labels, test_preds).sum()
    fp = np.logical_and(1-test_labels, test_preds).sum()
    tn = np.logical_and(1-test_labels, 1-test_preds).sum()
    fn = np.logical_and(test_labels, 1-test_preds).sum()

    # calculate binary classification metrics
    recall    = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy  = (tp + tn) / (tp + fp + tn + fn)
    f1_score  = 2.0*recall*precision/(recall + precision)
    
    if (verbose == True):
        # printing a table of metrics
        print("\n{:<11} {:.3f}".format('Recall:', recall))
        print("{:<11} {:.3f}".format('Precision:', precision))
        print("{:<11} {:.3f}".format('Accuracy:', accuracy))
    
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 
            'Precision': precision, 'Recall': recall, 'Accuracy': accuracy, 'F1_Score': f1_score}


# %%
def sigmoid_func(X):
    sigmoid_val = 1.0/(1.0 + np.exp(-X))
    return sigmoid_val

def sigmoid_derivative(X):   
    derivative  = np.subtract(1.0, X)
    derivative  = np.multiply(X, derivative)
    return derivative

# define helper activation functions
def apply_decision_boundary(X):
    return np.where(X > 0.5, 1, 0)

# define loss and fit functions for neural network
def linear_model(data, weights, bias):   
    # model we are fitting is, W*X^T + b
    WX = np.matmul(data, weights)
    WX_b = np.add(WX, bias)
    
    return WX_b

# \to-do: add regularization term to the cost
def cross_entropy_loss(predicted_class, class_labels, class_weight=None):
    sample_len = predicted_class.shape[0]

    # work around, instead of supplying 0 to log, supply some small value so as to get the desired affect in the cost
    predicted_class = np.where(predicted_class < 1e-5, 1e-8, predicted_class)
    
    predicted_neg_class = 1.0 - predicted_class
    predicted_neg_class = np.where(predicted_neg_class < 1e-5, 1e-8, predicted_neg_class)
    
    log_prediction_pos_class = np.log(predicted_class)
    log_prediction_neg_class = np.log(predicted_neg_class)
    
    desired_neg_class = np.subtract(1, class_labels)
    desired_pos_class = class_labels
    
    if (class_weight != None):
        desired_neg_class = np.multiply(class_weight[0], desired_neg_class)
        desired_pos_class = np.multiply(class_weight[1], desired_pos_class)

    cross_entropy =  np.dot(log_prediction_pos_class.T, desired_pos_class) + np.dot(log_prediction_neg_class.T, desired_neg_class)
      
    cross_entropy_mean = -cross_entropy/sample_len
    return cross_entropy_mean

# \to-do: update cost function gradient with the regularization term
def cross_entropy_loss_grad(predicted_class, class_labels, class_weight=None):
    
    gradient     = np.subtract(predicted_class, class_labels)
    denominator = np.subtract(1.0, predicted_class)
    denominator = np.multiply(predicted_class, denominator)
    denominator = np.where(denominator < 1e-5, 1e-5, denominator)
    
    gradient = np.divide(gradient, denominator)
    return gradient

def cross_entropy_loss_delta(predicted_class, class_labels, class_weight=None):
    pos_weight = 1.0
    neg_weight = 1.0
    
    if (class_weight != None):
        neg_weight = class_weight[0]
        pos_weight = class_weight[1]

    weight_diff = neg_weight - pos_weight
    
    predicted_class_w_weight = predicted_class
    class_labels_w_weight    = class_labels

    weights_available = False
    if ((weight_diff > 1e-3) or (weight_diff < 1e-3)):
        predicted_class_w_weight = np.multiply(neg_weight, predicted_class_w_weight)
        class_labels_w_weight    = np.multiply(pos_weight, class_labels_w_weight)
        weights_available = True

    delta = np.subtract(predicted_class_w_weight, class_labels_w_weight)

    if (weights_available == True):
        subtractive_term = np.multiply(predicted_class, class_labels)
        subtractive_term = weight_diff*subtractive_term
        delta = np.subtract(delta, subtractive_term)

    return delta

# %%
# define a struct neural layer to hold all the information
class NeuralLayer:
    def __init__(self):
        # weights are arranged as follows
        # for a layer of length 5, weights belonging to first node, coming from previous layer would be
        # _weights[0:len(weights):5]
        self._weights    = []
        self._bias       = []
        self._n_nodes    = 0
        self._activation = None

# %%
def feed_forward(data, neural_layers, class_weight=None):
    # compute equation W^T.X + b
    
    prev_layer_activation = data # n_samples x n_nodes
    n_samples            = data.shape[0]
    
    for layer_idx in range(1, len(neural_layers)):
        this_layer_activation = np.zeros((n_samples, neural_layers[layer_idx]._n_nodes))
        
        for node_idx in range(0, neural_layers[layer_idx]._n_nodes):
            weights_this_node = neural_layers[layer_idx]._weights[node_idx: : neural_layers[layer_idx]._n_nodes]
            this_layer_activation[:, node_idx] = linear_model(prev_layer_activation, 
                                                              weights_this_node,
                                                              neural_layers[layer_idx]._bias[node_idx])[:, 0]
            # apply sigmoid function
            this_layer_activation[:, node_idx] = sigmoid_func(this_layer_activation[:, node_idx])
            
        prev_layer_activation = this_layer_activation
    
    prev_layer_activation = apply_decision_boundary(prev_layer_activation)
        
    # final layer output will be of size n_samples x n_nodes   
    return prev_layer_activation

def forward_backward_pass(data, class_labels, neural_layers, class_weight=None):
    weight_gradients = [None]*(len(neural_layers)-1) # size: n_weight_terms for each layer
    bias_gradients   = [None]*(len(neural_layers)-1) # size: n_nodes for each layer
    
    # first do feedforward and record all intermidiary output at each node in each layer
    layers_feedforward_info    = [None]*len(neural_layers) # each row will be of size n_samples x n_nodes
    layers_feedforward_info[0] = data
    n_samples                  = data.shape[0]
    
    for layer_idx in range(1, len(neural_layers)):
        layers_feedforward_info[layer_idx] = np.zeros((n_samples, neural_layers[layer_idx]._n_nodes))

        for node_idx in range(0, neural_layers[layer_idx]._n_nodes):
            weights = neural_layers[layer_idx]._weights[node_idx: : neural_layers[layer_idx]._n_nodes]
            layers_feedforward_info[layer_idx][:, node_idx] = linear_model(layers_feedforward_info[layer_idx-1], 
                                                                          weights,
                                                                          neural_layers[layer_idx]._bias[node_idx])[:, 0]
            
            # apply sigmoid for the current node for the activation
            layers_feedforward_info[layer_idx][:, node_idx] = sigmoid_func(layers_feedforward_info[layer_idx][:, node_idx])
            
    # with the stored intermediary output, calculate gradients of weights and bias using chain-rule
    # partial for the output layer first
    this_layer_sigmoid       = layers_feedforward_info[-1] # size: n_samples x n_nodes in this layer
    prev_layer_sigmoid       = layers_feedforward_info[-2] # size: n_samples x n_nodes in this layer
    
    output_func_grad         = cross_entropy_loss_delta(apply_decision_boundary(this_layer_sigmoid), class_labels, class_weight) # size: n_samples x n_nodes
    
    result = np.empty((n_samples, neural_layers[-1]._weights.shape[0]))
    for column_idx in range(0, prev_layer_sigmoid.shape[1]):
        n_cols = output_func_grad.shape[1]
        result[:, column_idx*n_cols:(column_idx+1)*n_cols] = np.multiply(output_func_grad, prev_layer_sigmoid[:, column_idx][:, np.newaxis])

    result  = np.mean(result, axis=0)

    # weight_gradients[-1]     = np.matmul(output_func_grad.T, prev_layer_sigmoid)
    weight_gradients[-1]     = result[:, np.newaxis]
    bias_gradients[-1]       = np.mean(output_func_grad, axis=0)[:, np.newaxis]

    # partials for the remaining layers starting from L-1, with L being index for the output layer
    for layer_idx in range(len(neural_layers)-2, 0, -1):
        
        # calculate sigmoid_derivative
        this_layer_sigmoid_prime = sigmoid_derivative(prev_layer_sigmoid)
        prev_layer_sigmoid       = layers_feedforward_info[layer_idx-1]

        # calculate delta
        this_layer_output_func_grad = np.matmul(output_func_grad, neural_layers[layer_idx+1]._weights.reshape((output_func_grad.shape[1], -1)))
        this_layer_output_func_grad = np.multiply(this_layer_output_func_grad, this_layer_sigmoid_prime)

        result = np.empty((n_samples, neural_layers[layer_idx]._weights.shape[0]))
        for column_idx in range(0, prev_layer_sigmoid.shape[1]):
            n_cols = this_layer_output_func_grad.shape[1]
            result[:, column_idx*n_cols:(column_idx+1)*n_cols] = np.multiply(this_layer_output_func_grad, prev_layer_sigmoid[:, column_idx][:, np.newaxis])
            
        result = np.mean(result, axis=0)
        weight_gradients[layer_idx-1] = result[:, np.newaxis]
        
        bias_gradients[layer_idx-1]   = np.mean(this_layer_output_func_grad, axis=0)[:, np.newaxis]
        
        output_func_grad = this_layer_output_func_grad
    
    predicted_result = apply_decision_boundary(layers_feedforward_info[-1])
    return predicted_result, weight_gradients, bias_gradients

# %%
def generate_batch_indices(data_len, batch_size=100):   
    shuffle_indexing = np.random.permutation(data_len)
    n_batches        = int(data_len / batch_size)
    
    total_batch_len = n_batches*batch_size
    left_over_data  = data_len - total_batch_len
    
    batch_indices   = [None]*n_batches
    for batch_iter in range(0, n_batches):
        batch_indices[batch_iter] = shuffle_indexing[batch_iter*batch_size : (batch_iter+1)*batch_size]
    
    if (left_over_data > 0):
        batch_indices.append(shuffle_indexing[total_batch_len : data_len])
    
    return batch_indices

def steepest_descent(data, class_labels, neural_layers,
                     cost_func, cost_func_gradient, 
                     n_iter=100, step_size=0.25, regularization_factor=0.01, batch_size=100,
                     class_weight=None, stopping_tol=1e-3, verbose=False):

    if (verbose == True):
      print("Train on %d samples and %d features"%(data.shape[0], data.shape[1]))

    cost_func_trend  = [None]*(n_iter+1)
    
    initial_class_predictions = feed_forward(data, neural_layers)
    total_cost                = cost_func(initial_class_predictions, class_labels)
    cost_func_trend[0]        = total_cost
    
    error              = total_cost
    weight_decay_rate  = (1.0 - step_size*regularization_factor/(2.0*data.shape[0]))
    for iteration in range(0, n_iter):
        if (error > stopping_tol):
            batch_indices_list = generate_batch_indices(data.shape[0], batch_size)
            
            loss_this_iter = 0.0
            
            for batch_indices in batch_indices_list:
                train_data   = data[batch_indices, :]
                train_labels = class_labels[batch_indices]

                class_predictions, weight_grad, bias_grad = forward_backward_pass(train_data, train_labels, neural_layers, class_weight)
                
                # update weights
                for layer_idx in range(1, len(neural_layers)):
                    neural_layers[layer_idx]._weights = np.add(weight_decay_rate*neural_layers[layer_idx]._weights, 
                                                               -step_size*weight_grad[layer_idx-1])
                    
                    neural_layers[layer_idx]._bias    = np.add(neural_layers[layer_idx]._bias, 
                                                               -step_size*bias_grad[layer_idx-1])
            
                loss_this_iter += cost_func(class_predictions, train_labels, class_weight)
            
            loss_this_iter  = loss_this_iter / len(batch_indices_list)

            if (verbose == True):
              print("Epoch %d/%d"%(iteration, n_iter))
              print("loss: %f"%(loss_this_iter))

            cost_func_trend[iteration+1] = loss_this_iter[0]
            error = np.abs(loss_this_iter - error)
    
    return neural_layers, cost_func_trend


# %%
(train_features, train_labels), (test_features, test_labels) = train_test_split(features, labels, train_frac=0.6, seed=1)

# %%
# specify layer configuration
# each element specifying number of nodes in that layer
layer_config           = [2, 4, 1]

n_layers         = len(layer_config)
neural_layers    = [None]*n_layers

for layer_idx in range(0, n_layers):
    neural_layers[layer_idx] = NeuralLayer()
    neural_layers[layer_idx]._n_nodes = layer_config[layer_idx]
    if (layer_idx == 0):
        neural_layers[layer_idx]._weights = []
        neural_layers[layer_idx]._bias    = []
    else:
        prev_layer_nodes_len = layer_config[layer_idx-1]
        this_layer_nodes_len = layer_config[layer_idx]
        
        initial_weight_factor = 1.0/np.sqrt(prev_layer_nodes_len)
        neural_layers[layer_idx]._weights = initial_weight_factor*np.random.randn(prev_layer_nodes_len*this_layer_nodes_len, 1)
        neural_layers[layer_idx]._bias    = initial_weight_factor*np.ones((this_layer_nodes_len, 1))

# %%
start_time = time.time()

# estimate parameters
train_labels = train_labels.reshape((-1, layer_config[-1]))

# reset neural network weights before estimation
for layer_idx in range(0, n_layers):
    if (layer_idx == 0):
        neural_layers[layer_idx]._weights = []
        neural_layers[layer_idx]._bias    = []
    else:
        prev_layer_nodes_len = layer_config[layer_idx-1]
        this_layer_nodes_len = layer_config[layer_idx]
        
        initial_weight_factor = 3.6/np.sqrt(prev_layer_nodes_len)
        neural_layers[layer_idx]._weights = initial_weight_factor*np.random.randn(prev_layer_nodes_len*this_layer_nodes_len, 1)
        neural_layers[layer_idx]._bias    = initial_weight_factor*np.ones((this_layer_nodes_len, 1))

# as non-plagiarized data is only 20%, it is worse to classify non-plagiarized data as plagiarized than classifying
# plagiarized data as non-plagiarized. To account for this, class_weights can be used
class_weights = [10, 1]

neural_layers, cost_func_trend = steepest_descent(train_features, train_labels, neural_layers,
                                                  cross_entropy_loss, cross_entropy_loss_grad, 
                                                  n_iter=300, step_size=0.05, regularization_factor=0.01,
                                                  batch_size=5, verbose=False)
end_time = time.time()
print("time to took to run SGD: %f sec"%(end_time - start_time))

# %%
test_labels          = test_labels.reshape((-1, layer_config[-1]))
predicted_labels      = feed_forward(test_features, neural_layers)
evaluated_result_test = evaluate(predicted_labels, test_labels)

predicted_labels_train = feed_forward(train_features, neural_layers)
evaluated_result_train = evaluate(predicted_labels_train, train_labels)

print("Accuracy on test data ------ ")
print(evaluated_result_test)

print("\nAccuracy on trained data ------ ")
print(evaluated_result_train)

# plot cost function trend
plt.figure(figsize=(12,7))
plt.plot(cost_func_trend, 'rx-', markersize=4, linewidth=2)
plt.xlabel('iterations', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Cost function trend', fontsize=14)
plt.grid(True)
plt.show(block=False)


# %%
x_train = train_features
y_train = train_labels

# plot estimated polynomial by neural network
n_points = 100
x1_bounds = np.linspace(np.min(x_train[:, 0]), np.max(x_train[:, 0]), n_points)
x2_bounds = np.linspace(np.min(x_train[:, 1]), np.max(x_train[:, 1]), n_points)
x1_mesh, x2_mesh     = np.meshgrid(x1_bounds, x2_bounds, sparse=True)
predicted_class_mesh = np.ndarray((n_points, n_points))
X_input_features = np.zeros((n_points, x_train.shape[1]))
X_input_features[:, 0] = x1_mesh
for point_idx in range(0, n_points):
    X_input_features[:, 1]               = x2_mesh[point_idx, 0]
    predicted_class_mesh[:, point_idx]   = feed_forward(X_input_features, neural_layers)[:, 0]

marker_colors = np.where(y_train[:, 0] == 0, 'g', 'r')

plt.figure(figsize=(12,7))
h = plt.contour(x1_bounds, x2_bounds, np.transpose(predicted_class_mesh))
plt.scatter(x_train[:, 0], x_train[:, 1], c=marker_colors, s=25, alpha=0.7)
plt.grid(True)
plt.xlabel('$Containment$', fontsize=14)
plt.ylabel('$LCS$', fontsize=14)
plt.text(0.5, -0.25, 'Green: NonPlagiarize, Red: Plagiarized', fontsize=14)
plt.title('Estimated polynomial by NN', fontsize=14)
plt.show()


# %%


