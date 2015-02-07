import numpy as np
from sklearn.utils.extmath import logistic_sigmoid


class RBM(object):
    """
    Restricted Boltzmann Machine trained with unsupervised learning.

    Option ``lr`` is the learning rate.

    Option ``hidden_size`` is the size of the hidden layer.

    Option ``CDk`` is the number of Gibbs sampling steps used
    by contrastive divergence.

    Option ``seed`` is the seed of the random number generator.
    
    Option ``n_epochs`` number of training epochs.
    """

    def __init__(self, 
                 lr,             # learning rate
                 hidden_size,    # hidden layer size
                 CDk=1,          # nb. of Gibbs sampling steps
                 seed=1234,      # seed for random number generator
                 n_epochs=10     # nb. of training iterations
                 ):
        self.n_epochs = n_epochs
        self.hidden_size = hidden_size
        self.lr = lr
        self.CDk = CDk
        self.seed = seed
        self.rng = np.random.mtrand.RandomState(self.seed)   # create random number generator
    
    def _mean_hiddens(self, v):
        """Computes the probabilities P(h=1|v), i.e. mean-field values of the hidden layer
        Parameters
        
        v: array of shape (input_size, ) (considered 1 X input_size in dot product below )
        
        Returns:
        
        h: array of shape (hidden_size,)
           Corresponding mean field values for the hidden layer
        """

        return logistic_sigmoid(np.dot(v, self.W)
                                + self.b)

    def _sample_hiddens(self, v, rng):
        """Sample from the distribution P(h|v).
        rng : Random Number generator to be used.
        
        Returns:
        
        h: array of shape (hidden_size,) 
           Values of the hidden layer.
        """
        p = self._mean_hiddens(v)
        p[rng.uniform(size=p.shape) < p] = 1.
        return np.floor(p, p)

    def _sample_visibles(self, h, rng):
        """Sample from the distribution P(v|h).
        rng:  Random Number generator to be used.
        
        Parameters :
        
        h : array of shape (hidden_size,) (considered 1 X hidden_size in dot product below)
        Returns:
        v : array of shape (input_size,)
            Values of the visible layer
        """
        p = logistic_sigmoid(np.dot(h, self.W.T)
                             + self.c)
        p[rng.uniform(size=p.shape) < p] = 1.
        return np.floor(p, p)
    
    def gibbs(self, v):
        """Perform one Gibbs sampling step.

        Parameters
        ----------
        v : array-like, shape (input_size,)
            Values of the visible layer to start from.

        Returns
        -------
        v_new : array-like, shape (input_size,)
            Values of the visible layer after one Gibbs step.
        """
        rng = self.rng
        h_ = self._sample_hiddens(v, rng)
        v_ = self._sample_visibles(h_, rng)

        return v_

    def train(self,trainset):
        """
        Train RBM for ``self.n_epochs`` iterations.
        """
        # Initialize parameters
        input_size = trainset.metadata['input_size']

        # Parameter initialization
        self.W = (self.rng.rand(input_size,self.hidden_size)-0.5)/(max(input_size,self.hidden_size))
        self.b = np.zeros((self.hidden_size,))
        self.c = np.zeros((input_size,))
        
        sampledData = []
        for it in range(self.n_epochs):
            for iter_input,input in enumerate(trainset):
                # Perform CD-k
                # - you must use the matrix self.W and the bias vectors self.b and self.c
                input_sampled = input
                if it > 0:
                    input_sampled = sampledData[iter_input]
                for i in range(self.CDk):
                    input_sampled = self.gibbs(input_sampled)
                
                x_pos = input
                h_pos = self._mean_hiddens(x_pos)
                x_neg = input_sampled
                h_neg = self._mean_hiddens(x_neg)
                W_update = np.outer(x_pos, h_pos) - np.outer(x_neg, h_neg)
                b_update = h_pos - h_neg
                c_update = x_pos - x_neg
                
                self.W = self.W + self.lr * W_update
                self.b += self.lr * b_update
                self.c += self.lr * c_update
                
                #Tieleman (2008) - Persistent CD (PCD) ; initialize gibbs sampling on x_neg in next epoch
                if it==0:
                    sampledData.append(input) #change input to x_neg for PCD
                else:
                    sampledData[iter_input] = input #change input to x_neg for PCD
                
                
    def show_filters(self):
        from matplotlib.pylab import show, draw, ion
        import visualize as mlvis
        mlvis.show_filters(0.5*self.W.T,
                           200,
                           16,
                           8,
                           10,20,2)
        show()
