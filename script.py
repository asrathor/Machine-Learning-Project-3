import time
import numpy as np
from sklearn import svm
from scipy.io import loadmat
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pickle

def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.
    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector
    Output:
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    arr = np.ones((train_data[:,0].size,1))                             #create column of ones for bias
    train_data = np.concatenate((arr,train_data), axis=1)               #add the bias to the front of the train_data
    wx = train_data.dot(initialWeights)                                 #computing dot product of weights and train_data
    theta = sigmoid(wx)                                                 #using sigmoid function on the dot product (equation 1))
    thetaneg = sigmoid(-wx)                                             #1-theta = 1/(1+exp(wx))
    ylntheta = labeli.flatten() * np.log(theta)                         #convert label to array and multiply with log of theta
    temp = (1 - labeli).flatten() * np.log(thetaneg)                    #convert 1-label to array and multiply with log of 1-theta
    temp_one = np.sum((ylntheta + temp))                                #summation of both muliplied quantities from above
    temp_two = (theta - labeli.flatten()).dot(train_data)               #dot product of train data with difference of theta and label
    error = np.negative(temp_one/train_data[:, 0].size)                 #error is computed using equation 2
    temp_three = temp_two/train_data[:, 0].size                         #dividing dot product with train size for error_grad
    error_grad = temp_three.flatten()                                   #error_grad is computed from equation 3 and flattened
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix
    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    arr = np.ones((data[:, 0].size, 1))                                 #create bias column of ones
    data = np.concatenate((arr, data), axis=1)                          #add the bias in front of the train_data
    temp = data.dot(W)                                                  #compute the dot product of W and train_data
    postprob = sigmoid(temp)                                            #posterior probability is sigmoid of wx
    maxprob = np.argmax(postprob, axis=1)                               #maximum posterior probability is computed
    maxprob.resize((data.shape[0], 1))                                  #reshape can be used here too. Resize just changes the array
                                                                        #permanently
    label = maxprob

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.
    Input:
        initialWeights: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector
    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    params = params.reshape(n_feature+1, n_class)             # reshape initial weights
    bias = np.ones((n_data, 1))                               # creating bias terms
    train_data = np.concatenate((train_data, bias), axis=1)   # adding bias
    numerator = np.exp(np.dot(train_data, params))            # exp(W.T dot X)
    denominator = np.sum(numerator, axis=1)                   # Sum up all exp(W.T dot X)
    denominator = denominator.reshape(n_data, 1)              # reshape the results of summation
    posteriors = numerator / denominator                      # calculating posterior P(y=Ck|X)
    labeli = labeli.reshape(n_data, n_class)                  # reshape true labels
    predict_true = posteriors - labeli                        # posterior - true label (theta - y)
    error_grad = np.dot(train_data.T, predict_true)           # error_grad = summation of (theta - y) dot X
    error_grad = error_grad.flatten()                         # flatten the error_grad

    error = np.multiply(labeli, np.log(posteriors))           # multiply y and ln theta
    error = np.sum(error, axis=1)                             # sum up values in y axis
    error = -np.sum(error, axis=0)                            # sum up values in x axis
    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix
    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    bias = np.ones((data.shape[0], 1))                      # creating bias
    data = np.concatenate((data, bias), axis=1)             # adding bias
    _numerator = np.exp(np.dot(data, W))                    # exp(W.T dot X)
    _denominator = np.sum(_numerator, axis=1)               # Sum up all exp(W.T dot X)
    _denominator = _denominator.reshape(data.shape[0], 1)   # reshape the results of summation
    _posterior = _numerator / _denominator                  # calculating posterior P(y=Ck|X)
    label = np.argmax(_posterior, axis=1)                   # predicting labels based on the largest posterior term
    label = label.reshape(data.shape[0], 1)                 # reshape labels
    return label

"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
start_time = time.time()

for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))
predicted_label = blrPredict(W, train_data)
elapsed_time = time.time() - start_time
print("elapsed time to train "+str(elapsed_time)+"s")
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
#Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
#Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

# pickle for BLR
f1 = open('params.pickle', 'wb')
pickle.dump(W, f1)
f1.close()

"""
#Script for Support Vector Machine
"""
print('\n\n-------------------SVM-------------------\n\n')

########### Kernel Linear ###########
clf = svm.SVC(kernel='linear')                                         # set up the linear kernel svm
clf.fit(train_data, train_label.ravel())                               # train the svm
acc_train = clf.score(train_data, train_label)                         # calculating the accuracy in train data
print("Accuracy on train data using Linear Kernel: ", acc_train)

acc_vali = clf.score(validation_data, validation_label)                # calculating the accuracy in validation data
print("Accuracy on validation data using Linear Kernel: ", acc_vali)

acc_test = clf.score(test_data, test_label)                            # calculating the accuracy in test data
print("Accuracy on test data using Linear Kernel: ", acc_test)
#####################################

########### Kernel rbf (gamma = 1) ###########
clf = svm.SVC(kernel='rbf', gamma=1)                                   # set up the rbf svm with gamma = 1
clf.fit(train_data, train_label.ravel())                               # train the svm
acc_train = clf.score(train_data, train_label)                         # calculating the accuracy in train data
print("Accuracy on train data using rbf Kernel with gamma=1: ", acc_train)

acc_vali = clf.score(validation_data, validation_label)                # calculating the accuracy in validation data
print("Accuracy on validation data using rbf Kernel with gamma=1: ", acc_vali)

acc_test = clf.score(test_data, test_label)                            # calculating the accuracy in test data
print("Accuracy on test data using rbf Kernel with gamma=1: ", acc_test)
##############################################

########### Kernel rbf (gamma = default) ###########
clf = svm.SVC(kernel='rbf', gamma='auto')                              # set up the rbf svm with gamma = default
clf.fit(train_data, train_label.ravel())                               # train the svm
acc_train = clf.score(train_data, train_label)                         # calculating the accuracy in train data
print("Accuracy on train data using rbf Kernel with gamma=default: ", acc_train)

acc_vali = clf.score(validation_data, validation_label)                # calculating the accuracy in validation data
print("Accuracy on validation data using rbf Kernel with gamma=default: ", acc_vali)

acc_test = clf.score(test_data, test_label)                            # calculating the accuracy in test data
print("Accuracy on test data using rbf Kernel with gamma=default: ", acc_test)
####################################################

########### Kernel rbf (gamma = default, C = 1, 10, 20, ......, 100) ###########
cs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]                      # creating different C values
acc_train = np.zeros((11, 1))                                          # for storing different accuracy in train data using different C
acc_vali = np.zeros((11, 1))                                           # for storing different accuracy in validation data using different C
acc_test = np.zeros((11, 1))                                           # for storing different accuracy in test data using different C
index = 0
for c in cs:
    print("------------------------ C =", c, " ------------------------")
    clf = svm.SVC(kernel='rbf', gamma='auto', C=c)                     # set up rbf svm with gamma = default and different C values
    clf.fit(train_data, train_label.ravel())                           # train the svm
    acc_train[index] = clf.score(train_data, train_label)              # calculating the accuracy in train data
    print("Accuracy on train data using rbf Kernel with different C: ", acc_train[index])
    acc_vali[index] = clf.score(validation_data, validation_label)     # calculating the accuracy in validation data
    print("Accuracy on validation data using rbf Kernel with different C: ", acc_vali[index])
    acc_test[index] = clf.score(test_data, test_label)                 # calculating the accuracy in test data
    print("Accuracy on test data using rbf Kernel with different C: ", acc_test[index])
    index = index + 1
    print("------------------------------------------------------------\n")

# plot out the accuracy using different C values in train, validation, and test data
fig = plt.figure(figsize=[18, 6])

plt.subplot(1, 3, 1)
plt.plot(cs, acc_train)
plt.title('Accuracy for train data using different C')

plt.subplot(1, 3, 2)
plt.plot(cs, acc_vali)
plt.title('Accuracy for validation data using different C')

plt.subplot(1, 3, 3)
plt.plot(cs, acc_test)
plt.title('Accuracy for test data using different C')

plt.show()
################################################################################

"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
start_time = time.time()

W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')
elapsed_time = time.time() - start_time
print("elapsed time to train: "+str(elapsed_time)+" s")

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')
start_time = time.time()

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
elapsed_time = time.time() - start_time
print("elapsed time to Predict: "+str(elapsed_time)+"s")

# pickle for MLR
f2 = open('params_bonus.pickle', 'wb')
pickle.dump(W_b, f2)
f2.close()
