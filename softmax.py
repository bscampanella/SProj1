from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    numExamples = X.shape[0]
    numClasses = W.shape[1]

    LossMat = X.dot(W)
    for i in range(numExamples):
      LossMat[i] -= np.amax(LossMat[i])
      scoreArr = np.exp(LossMat[i])/np.sum(np.exp(LossMat[i]))
      loss += -np.log(scoreArr[y[i]])
      #tempGradMat = np.zeros(W.shape)
      for j in range(numClasses):
        tempGradArr = (scoreArr[j] - (y[i] == j)) * X[i]
        dW[:,j] += tempGradArr
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    loss /= numExamples
    loss -= reg * np.sum(W*W)

    dW /= numExamples
    dW += reg * 2 * W
    
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    numExamples = X.shape[0]
    
    #loss first
    scoreMat = np.dot(X, W)
    scoreMat -= np.max(scoreMat, axis=1)[:,None]
    correctVec = scoreMat[np.arange(numExamples), y]
    correctVecExp = np.exp(correctVec)

    sumExpRow = np.sum(np.exp(scoreMat), axis = 1)

    lossVec = correctVecExp / sumExpRow
    lossVec = -np.log(lossVec)
    loss = np.sum(lossVec)

    loss /= numExamples
    loss += reg * np.sum(W * W)

    #gradient
    expScoreMat = np.exp(scoreMat)
    expScoreMatNormalized = expScoreMat / sumExpRow[:,None]

    subtractMat = np.zeros_like(scoreMat)
    subtractMat[np.arange(numExamples), y] = 1

    gradCoefMat = expScoreMatNormalized - subtractMat

    dW = np.dot(X.T, gradCoefMat)
    dW /= numExamples
    dW += reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
