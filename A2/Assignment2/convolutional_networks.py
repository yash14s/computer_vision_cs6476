"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from a2_helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, ReLU, adam, gtid


def hello_convolutional_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modfiy the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the convolutional forward pass.                  #
        # Hint: you can use function torch.nn.functional.pad for padding.  #
        # You are NOT allowed to use anything from torch.nn in other places. #
        ####################################################################
        # Replace "pass" statement with your code
        pass
        #####################################################################
        #                          END OF YOUR CODE                         #
        #####################################################################
        cache = (x, w, b, conv_param)
        return out, cache


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the max-pooling forward pass                     #
        ####################################################################
        # Replace "pass" statement with your code
        pass
        ####################################################################
        #                         END OF YOUR CODE                         #
        ####################################################################
        cache = (x, pool_param)
        return out, cache

class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = torch.zeros_like(tx), \
                         torch.zeros_like(layer.weight), \
                         torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = \
            pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A custom layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        out = None
        cache = None
        ######################################################################
        # TODO: Refer to the Conv_ReLU class above and implement a similar   #
        # forward function that performs first a convolution, then a ReLU    #
        # and lastly a pool                                                  #
        ######################################################################
        # Replace "pass" statement with your code
        pass
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool
        convenience layer
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db

class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ######################################################################
        ############################  TODO  ##################################
        # Initialize weights and biases for the three-layer convolutional network.
        # Weights should be initialized from a Gaussian distribution centered at 0.0
        # with a standard deviation equal to 'weight_scale'. Biases should be initialized to zero.
        # Store all weights and biases in the 'self.params' dictionary.

        # For the convolutional layer, use the keys 'W1' and 'b1'.
        # For the hidden linear layer, use keys 'W2' and 'b2'.
        # For the output linear layer, use keys 'W3' and 'b3'.

        # IMPORTANT: You can assume that the padding and stride of the first convolutional layer
        # are chosen so that the width and height of the input are preserved. Check the
        # 'loss()' function for more details on this.
        # HINT: Use input data (conv_param, pool_param, input_dim, filter_size etc) 
        #       to figure out the dimensions of weights and biases
        ######################################################################
        # Replace "pass" statement with your code
        pass
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Inputs:
        - X: Tensor of input data
        - y: int64 Tensor of labels, of shape (N,). y[i] gives the label for X[i].
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        # pass conv_param to the forward pass for the convolutional layer
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for the three-layer conv net.     #
        # Computing the class scores for X and storing them in the scores    #
        # variable. Store outputs of all the layers since they will be used  #
        # in computing gradients (see last TODO for this class)              #                                                         #
        #                                                                    #
        # HINT: Use forward functions of the custom layers that you have     #
        # already implemented before.                                        #
        ######################################################################
        # Replace "pass" statement with your code
        pass
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################

        if y is None:
            return scores

        # Following is the implementation of the backward pass for the 
        # three-layer convolutional network.
        loss, grads = 0.0, {}
        loss, dout = softmax_loss(scores, y)

        ####################################################################
        # TODO: Incorporate L2 regularization into loss
        # 
        ####################################################################
        # Replace "pass" statement with your code
        pass
        ###################################################################
        #                             END OF YOUR CODE                    #
        ###################################################################

        ####################################################################
        # TODO: Assign the 2nd output variables (cache) of your custom layers to the given below 
        # variables so that it can be used in calculating gradients below
        ####################################################################
        cache_L = None # replace None with 2nd output of forward for linear layer
        cache_LR = None # replace None with 2nd output of forward for linear_relu layer
        cache_CRP = None # replace None with 2nd output of forward for conv_relu_pool layer
        ###################################################################
        #                             END OF YOUR CODE                    #
        ###################################################################

        #Following code calculates gradients
        last_dout, dw, db  = Linear.backward(dout, cache_L)
        grads['W{}'.format(i)] = dw + 2*self.params['W{}'.format(i)]*self.reg
        grads['b{}'.format(i)] = db
        i-=1
        last_dout, dw, db  = Linear_ReLU.backward(last_dout, cache_LR)
        grads['W{}'.format(i)] = dw + 2*self.params['W{}'.format(i)]*self.reg
        grads['b{}'.format(i)] = db
        i-=1
        last_dout, dw, db  = Conv_ReLU_Pool.backward(last_dout, cache_CRP)
        grads['W{}'.format(i)] = dw + 2*self.params['W{}'.format(i)]*self.reg
        grads['b{}'.format(i)] = db

        return loss, grads


def create_convolutional_solver_instance(data_dict, dtype, device):

    #### put your final hyperparameters here ####
    num_filters = None
    filter_size = None
    hidden_dim = None
    reg = None
    weight_scale = None
    learning_rate = None
    num_epochs = None
    batch_size = None
    update_rule = None
    #############################################

    input_dims = data_dict['X_train'].shape[1:]
    model = ThreeLayerConvNet(num_filters=num_filters, filter_size=filter_size, 
                              hidden_dim=hidden_dim, reg=reg, weight_scale=weight_scale, 
                              dtype=torch.float, device='cpu')

    solver = Solver(model, data_dict,
                    num_epochs=num_epochs, batch_size=batch_size,
                    update_rule=adam,
                    optim_config={
                      'learning_rate': learning_rate,
                    },
                    device='cpu')
    return solver
