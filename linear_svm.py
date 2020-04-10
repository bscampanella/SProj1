from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero
    dWInstance = np.zeros(W.shape)
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        dWInstance = np.zeros(W.shape)
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            
            if margin > 0:
                loss += margin
                dWInstance[:,j] = X[i] #grad

        RowSum = dWInstance.sum(axis=1) #grad
        dWInstance[:,y[i]] = -RowSum #grad
        dW += dWInstance #grad

  
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    #reg gradient coming later
    dW += reg * 2 * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    numExamples = X.shape[0] 
    scoresMat = np.dot(X,W)

    LossMat = scoresMat - (scoresMat[range(numExamples),y])[:, None] + 1
    LossMat[LossMat < 0] = 0   
    LossMat[np.arange(numExamples),y] = 0
    #loss mat is a matrix with num_examples x num_classes that has the ammount of loss acumulated from each example and each class classifier

    LossPerItem = LossMat.sum(axis = 1) #the total loss accumulated from each example

    TotalLoss = LossPerItem.sum()
    TotalLoss /= numExamples
    loss = TotalLoss

    loss += reg * np.sum(W * W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    binaryLossMat = np.zeros(LossMat.shape)
    binaryLossMat = LossMat
    binaryLossMat[binaryLossMat > 0] = 1  
    #binary loss mat is a num_examples x num_classes matrix with 1 in i,j place if the i_th example created loss with the j_th class classifier
    #note that by definition of loss function if example[i] is part of class j then it contributes 0 loss and is 0 in bin loss mat

    totalBinLossPerExample = binaryLossMat.sum(axis = 1) #basically the count of how many classifiers accumulated loss on an example

    binaryLossMat[np.arange(numExamples), y] = -totalBinLossPerExample 
    #now binaryLossMat is matrix like before but if example[i] has corect class j, then the ixj spot has -totalBinLossPerExample[i] 

    dW = X.T.dot(binaryLossMat)
    #this is the trick part X is num_examples x num_dim_per_example matrix, transposed is num_dim_per_example x num_examples matrix
    #num_dim_per_example x num_examples matrix (dot) num_examples x num_classes = num_dim_per_example x num_classes matrix
    #pretend there is one class and so one classifier. Then multiplying the first colum of binaryLossMat by first row of X.T is
    #calculating the first element of the graident by adding up how many times the first element of each example that generates a loss for that class
    # (or subtracts if that example was part of that class)
    #moving to the next gradient element below, it how many times the second element of each example should be included in that class gradient

    dW /= numExamples
    dW += reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
