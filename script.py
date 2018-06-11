import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

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
    
    X = np.concatenate((np.ones((n_data,1)),train_data),1)
    labeli = labeli.flatten()
    Y = sigmoid(np.dot(X,initialWeights))
    error = -1*(np.sum(labeli* np.log(Y) + (1.0-labeli)*np.log(1.0-Y)))/n_data    
    
    Yn = (Y-labeli).reshape(n_data,1)
    error_grad = Yn * X
    error_grad = np.sum(error_grad, axis=0)/n_data
    print("error: "+ str(error)+"\n")
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
    n_data = data.shape[0]
    
    Y = sigmoid(np.dot(data,W))
    Y = np.argmax(Y, axis=1)
    label = Y.reshape((n_data,1))
    return label

def softmax(x):
    num = np.exp(x)
    den = np.sum(num,axis = 1)
    den = den.reshape(den.shape[0],1)
    return num / den

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
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
    
    X = np.concatenate((np.ones((n_data,1)),train_data),1)
    params = params.reshape(n_feature+1,10)
    Y = softmax(np.dot(X,params))
    
    error = -1 * (np.sum(np.sum(labeli * np.log(Y))))/n_data
    print("error: "+ str(error)+"\n")
    
    Yn = (Y-labeli)
    error_grad = np.dot(np.transpose(X),Yn)/n_data
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
    n_data = data.shape[0]
    
    X = np.concatenate((np.ones((n_data,1)),data),1)
    Y = softmax(np.dot(data,W))
    Y = np.argmax(Y, axis=1)
    label = Y.reshape((n_data,1))

    
    return label

def plot_confusion_matrix(cls_pred,true_label):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    #cls_true = data.test.cls
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=true_label,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)


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
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
plot_confusion_matrix(cls_pred=predicted_label,true_label=train_label)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
plot_confusion_matrix(cls_pred=predicted_label,true_label=test_label)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
"""

Script for Support Vector Machine

"""
print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

clf_1 = SVC(kernel='linear')
clf_1.fit(train_data, train_label.flatten()) 

print('\n********************************************************************\n')
print('Linear kernel') 
print('\n Training set Accuracy:' + str(100 * clf_1.score(train_data, train_label.flatten())) + '%')
print('\n Validation Accuracy: '+ str(100 * clf_1.score(validation_data, validation_label.flatten())) + '%') 
print('\n Testing Accuracy: '+ str(100 * clf_1.score(test_data, test_label.flatten()))+'%')
print('\n********************************************************************\n')


clf_2 = SVC(gamma=1.0, kernel='rbf')
clf_2.fit(train_data, train_label.flatten()) 

print('\n********************************************************************\n')
print('Rbf kernel with gamma = 1')
print('\n Training set Accuracy:' + str(100 * clf_2.score(train_data, train_label.flatten())) + '%')
print('\n Validation set Accuracy: '+ str(100 * clf_2.score(validation_data, validation_label.flatten())) + '%') 
print('\n Testing set Accuracy: '+ str(100 * clf_2.score(test_data, test_label.flatten())) + '%')
print('\n********************************************************************\n')

clf_3 = SVC(gamma=0.0, kernel='rbf')
clf_3.fit(train_data, train_label.flatten()) 

print('\n********************************************************************\n')
print('Rbf kernel with gamma = 0') 
print('\n Training set Accuracy:' + str(100 * clf_3.score(train_data, train_label.flatten())) + '%')
print('\n Validation set Accuracy: '+ str(100 * clf_3.score(validation_data, validation_label.flatten())) + '%') 
print('\n Testing set Accuracy: '+ str(100 * clf_3.score(test_data, test_label.flatten())) + '%')
print('\n********************************************************************\n')



training_acc = np.zeros(11)
validation_acc = np.zeros(11)
testing_acc = np.zeros(11)

j = 0
clf = SVC(C=1.0, kernel='rbf')
clf.fit(train_data, train_label.flatten()) 
training_acc[j] = 100 * clf.score(train_data, train_label.flatten())
validation_acc[j] = 100 * clf.score(validation_data, validation_label.flatten())*100
testing_acc[j] = 100 * clf.score(test_data, test_label.flatten())*100
print('\n********************************************************************\n')  
print('Rbf with gamma = 0 and C varying from 1 to 100') 
print('Training set Accuracy: '+ str(training_acc)+'%') 
print('Validation set Accuracy: '+ str(validation_acc)+'%') 
print('Testing set Accuracy: '+ str(testing_acc)+'%')
print('\n********************************************************************\n')

j = j + 1

for i in range(10,101,10):
    clf = SVC(C=float(i), kernel='rbf')
    clf.fit(train_data, train_label.flatten()) 
    training_acc[j] = 100 * clf.score(train_data, train_label.flatten())
    validation_acc[j] = 100 * clf.score(validation_data, validation_label.flatten())*100
    testing_acc[j] = 100 * clf.score(test_data, test_label.flatten())*100
    j = j + 1
print('\n********************************************************************\n')  
print('Rbf with gamma = 0 and C varying from 1 to 100') 
print('Training set Accuracy: '+ str(training_acc)+'%') 
print('Validation set Accuracy: '+ str(validation_acc)+'%') 
print('Testing set Accuracy: '+ str(testing_acc)+'%')
print('\n********************************************************************\n')

pickle.dump((training_acc, validation_acc, testing_acc),open("rbf_cval.pickle","wb"))


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
plot_confusion_matrix(cls_pred=predicted_label_b,true_label=train_label)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
#plot_confusion_matrix(cls_pred=predicted_label_b,true_label=train_label)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
plot_confusion_matrix(cls_pred=predicted_label_b,true_label=test_label)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
