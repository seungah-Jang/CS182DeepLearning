import numpy as np

from deeplearning.layers import *
from deeplearning.layer_utils import *


'''
변수 | shape | role
W1  (D,H) 입력-> 은닉층 연결 가중치
b1  (H,)  은닉층 편향
W2  (H,C) 은닉층 -> 출력층 연결 가중치
b2  (C,)  출력층 편향
'''

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################

        self.params['W1'] = weight_scale * np.random.randn(input_dim,hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)

        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        정답 예측 (Foward Pass)
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        학습(Backward Pass)
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # Forward pass
        # Affine -> relu -> Affine

        # 첫번째 affine layer
        # cache: X,w,b
        a1, cache1 = affine_forward(X,W1,b1)

        # Relu layer
        h1, relu_cache = relu_forward(a1)

        # 두번째 affin layer
        scores, cache2 = affine_forward(h1,W2,b2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization on the weights,    #
        # but not the biases.                                                      #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        # scores: affine layer 결과
        # y 는 정답 레이블
        # score 와 y 를 비교하여 loss 와 gradient를 구함.
        # softmax: 예측 점수를 확률로 바꿔줌
        data_loss, dscores = softmax_loss(scores,y)

        #정규화
        reg_loss = 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        loss = data_loss + reg_loss 

        dh1, dW2, db2 = affine_backward(dscores, cache2)
        da1 = relu_backward(dh1, relu_cache)
        dX, dW1, db1 = affine_backward(da1, cache1)

        # gradient에 정규화 미분 더하기.
        dW2 += self.reg * W2
        dW1 += self.reg * W1

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        
        # 각 layer 마다 가중치, 편향 초기화
        layer_input_dim = input_dim # 첫 번째 레이어는 입력 데이터의 크기 / 그 다음 레이어들은 이전 레이어의 출력 크기를 입력으로 받는다.

        
        for i in range(self.num_layers):
            layer_output_dim = hidden_dims[i] if i < self.num_layers -1 else num_classes # 은닉층이면 hidden_dim, 마지막층이면 num_classes
            # 표준 정규분포를 따르는 랜덤숫자를 (layer_input_dim,layer_output_dim) 크기의 행렬로 생성
            # weight scale 을 곱하여, 크기를 조절함.
            self.params[f'W{i+1}'] = weight_scale * np.random.randn(layer_input_dim, layer_output_dim)
            # bias 를 0으로 초기화
            self.params[f'b{i+1}'] = np.zeros(layer_output_dim)

            if self.use_batchnorm and i < self.num_layers-1:
                self.params[f'gamma{i+1}'] = np.ones(layer_output_dim)
                self.params[f'beta{i+1}'] = np.zeros(layer_output_dim)
            layer_input_dim = layer_output_dim


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        caches = []
        out = X
        for i in range(1, self.num_layers):
            W, b = self.params[f'W{i}'], self.params[f'b{i}']
            out, fc_cache = affine_forward(out, W, b)

            if self.use_batchnorm:
                gamma, beta = self.params[f'gamma{i}'], self.params[f'beta{i}']
                bn_param = self.bn_params[i-1]
                out, bn_cache = batchnorm_forward(out, gamma, beta, bn_param)
            else:
                bn_cache = None

            out, relu_cache = relu_forward(out)

            if self.use_dropout:
                out, do_cache = dropout_forward(out, self.dropout_param)
            else:
                do_cache = None

            caches.append((fc_cache, bn_cache, relu_cache, do_cache))

        W_final, b_final = self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}']
        scores, final_cache = affine_forward(out, W_final, b_final)        


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization on the         #
        # weights, but not the biases.                                             #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        # softmax loss 계산, scores: forward에서 계산된 출력값(로짓)
        loss, dout = softmax_loss(scores, y)

        # 각 가중치 W 에 대해 L2 정규화 손실 추가
        for i in range(1, self.num_layers + 1):
            W = self.params[f'W{i}']
            loss += 0.5 * self.reg * np.sum(W * W)

        # dout : 다음 층에 넘길 gradient
        # dw, db : 해당 층의 weight, bias 에 대한 gradient
        # dw 에 정규화 추가 
        dout, dw, db = affine_backward(dout, final_cache)
        grads[f'W{self.num_layers}'] = dw + self.reg * self.params[f'W{self.num_layers}']
        grads[f'b{self.num_layers}'] = db

        # 각 은닉층에 대한 역전파
        for i in reversed(range(1, self.num_layers)):
            fc_cache, bn_cache, relu_cache, do_cache = caches[i-1]

            # dropout : 꺼졌던 뉴런 복원하는 gradient
            if self.use_dropout:
                dout = dropout_backward(dout, do_cache)

            # 0 이하였던 뉴런은 gradient 가 0 이 됨.
            dout = relu_backward(dout, relu_cache)

            # batch normalization 인 경우, scale(gamma)와 shift(beta) 에 대한 gradient 계산
            if self.use_batchnorm:
                dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)
                grads[f'gamma{i}'] = dgamma
                grads[f'beta{i}'] = dbeta

            # 마지막은 affine 역전파
            dout, dw, db = affine_backward(dout, fc_cache)
            grads[f'W{i}'] = dw + self.reg * self.params[f'W{i}']
            grads[f'b{i}'] = db
            '''
            [Output Layer]
            softmax_loss
            → affine_backward
            → grads[W3], grads[b3]

            [Hidden Layer 2]
            → dropout_backward (선택)
            → relu_backward
            → batchnorm_backward (선택)
            → affine_backward
            → grads[W2], b2, gamma2, beta2

            [Hidden Layer 1]
            → dropout_backward (선택)
            → relu_backward
            → batchnorm_backward (선택)
            → affine_backward
            → grads[W1], b1, gamma1, beta1
            '''
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return loss, grads
