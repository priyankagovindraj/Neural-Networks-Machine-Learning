import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import pickle
from sklearn.svm import SVC


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
    # D = train_data.shape[1]
    initialWeights = initialWeights.reshape(n_features+1,1)
    # add bias
    X = np.ones((n_data,1))
#     print X.shape
    X = np.concatenate((X,train_data), axis=1)
    # p = w.t * X
    # print ('x.shape:',X.shape)
    p = np.dot(X,initialWeights)
#     print p.shape
    # theta = sigmoid(p)
    theta = sigmoid(p)
    # print ('theta.shape:',theta.shape)



    for i in range (n_data):
        y = labeli[i]
        error += (y * np.log(theta[i]) +  (1 - y) * np.log(1-theta[i]))
    error = -error / n_data
    # error_grad

    temp = theta - labeli

    error_grad = np.dot(temp.T, X) / n_data
    error_grad = error_grad.flatten()

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

    # W [716,10], data [50000,715]

    # add biass
    X = np.ones((data.shape[0],1))
    X = np.concatenate((X,data), axis=1)
    # posterior probability
    p = np.dot(X,W)
    post = sigmoid(p)
    # max
    label = np.argmax(post,1)
    label = label.reshape(label.shape[0],1)


    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """

    # train_data, y = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))


    ##################
    # YOUR CODE HERE #
    ##################
#     # HINT: Do not forget to add the bias term to your input data

    # D = train_data.shape[1]
    initialWeights_b = params
    initialWeights_b = initialWeights_b.reshape(n_feature+1,10)
    # add bias
    X = np.ones((n_data,1))
#     print X.shape
    X = np.concatenate((X,train_data), axis=1)
    # get theta_nk
    p = np.dot(X,initialWeights_b) #50000*10
    p = np.exp(p)
    p_sum = np.sum(p,1)
    theta = p/p_sum.reshape(n_data,1)

    # error
    error = -np.sum(Y*np.log(theta))/n_data

    # error_grad
    e = theta - Y
    for k in range (10):
        # print (e[:,k].shape)
        # print (X.shape)
        s = np.sum(e[:,k].reshape(n_data,1)*X,0)
        
        # s = s.reshape(D+1,1)
        # print (s.shape)
        error_grad[:,k] = s
        # error_grad[k,:] = np.sum(e[:,k].reshape(n_data,1)*X,0).reshape(D+1,1)

    error_grad = (error_grad)/n_data

    error_grad = error_grad.flatten()

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

    # W [716,10], data [50000,715]

    # add bias
    X = np.ones((data.shape[0],1))
    X = np.concatenate((X,data), axis=1)
    # posterior probability
    p = np.dot(X,W)
    post = sigmoid(p)
    # max
    label = np.argmax(post,1)
    label = label.reshape(label.shape[0],1)



    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();

# number of classes
n_class = 10;

# number of training samples
n_train = train_data.shape[0];

# number of features
n_feature = train_data.shape[1];

Y = np.zeros((n_train, n_class));
for i in range(n_class):
    Y[:,i] = (train_label == i).astype(int).ravel();
    
# Logistic Regression with Gradient Descent
W = np.zeros((n_feature+1, n_class));
initialWeights = np.zeros((n_feature+1,1));
opts = {'maxiter' : 100};
for i in range(n_class):
    labeli = Y[:,i].reshape(n_train,1);
    args = (train_data, labeli);
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
    W[:,i] = nn_params.x.reshape((n_feature+1,));

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data);
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data);
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data);
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

with open('params.pickle', 'wb') as f1: 
    pickle.dump(W, f1) 

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
# ##################
# # YOUR CODE HERE #
# ##################
#SVM function
svm1 = SVC()
print("Using linear kernel")
#Using linear kernel
svm1.set_params(kernel = 'linear')
#Fit the model on the training data

svm1.fit(train_data, train_label.ravel()) 

print('\n Using linear kernel (all other parameters are kept default)') 
print('  Training set Accuracy: '+ str(svm1.score(train_data, train_label.ravel())*100)+'%') 
print('  Validation set Accuracy: '+ str(svm1.score(validation_data, validation_label.ravel())*100)+'%') 
print('  Testing set Accuracy: '+ str(svm1.score(test_data, test_label.ravel())*100)+'%')



svm2 = SVC()
#Using radial basis function and gamma=1
svm2.set_params(kernel = 'rbf', gamma=1.0)
#Fit the model on the training data
svm2.fit(train_data, train_label.ravel())
print('\n Using radial basis function with value of gamma setting to 1') 
print('  Training set Accuracy: '+ str(svm2.score(train_data, train_label.ravel())*100)+'%') 
print('  Validation set Accuracy: '+ str(svm2.score(validation_data, validation_label.ravel())*100)+'%') 
print('  Testing set Accuracy: '+ str(svm2.score(test_data, test_label.ravel())*100)+'%')



svm3 = SVC()
print("Using rbf and gamma set to default")
#Using radial basis function and gamma default
svm3.set_params(kernel='rbf')
#Fit the model on the training data
svm3.fit(train_data, train_label.ravel())
print('\n Using radial basis function with value of gamma setting to default') 
print('  Training set Accuracy: '+ str(svm3.score(train_data, train_label.ravel())*100)+'%') 
print('  Validation set Accuracy: '+ str(svm3.score(validation_data, validation_label.ravel())*100)+'%') 
print('  Testing set Accuracy: '+ str(svm3.score(test_data, test_label.ravel())*100)+'%')


accuracy1 = np.zeros(11)
accuracy2 = np.zeros(11)
accuracy3 = np.zeros(11)

accuracy1[0] = svm3.score(train_data, train_label.ravel())*100
accuracy2[0] = svm3.score(validation_data, validation_label.ravel())*100
accuracy3[0] = svm3.score(test_data, test_label.ravel())*100

print("Using rbf and varying values of C")
values = ['10','20','30','40','50','60','70','80','90','100']
#Using radial basis function and varying values of C


index = 1
for i in values: #range(10,101,10):
    svm = SVC()
    svm.set_params(kernel='rbf', C = float(i))
    svm.fit(train_data, train_label.ravel()) 

    accuracy1[index] = svm.score(train_data, train_label.ravel())*100
    accuracy2[index] = svm.score(validation_data, validation_label.ravel())*100
    accuracy3[index] = svm.score(test_data, test_label.ravel())*100
    
    index = index + 1

print('\n Using radial basis function with value of gamma setting to default and varying value of C (1, 10, 20, 30, · · · , 100)')
print('  Training set Accuracy: '+ str(accuracy1)+'%') 
print('  Validation set Accuracy: '+ str(accuracy2)+'%') 
print('  Testing set Accuracy: '+ str(accuracy3)+'%')

"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')


with open('params_bonus.pickle', 'wb') as f2:
    pickle.dump(W_b, f2)

