import tensorflow as tf
from .utils import *


class Graph2Gauss:
    """
    Implementation of the method proposed in the paper:
    'Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via Ranking'
    by Aleksandar Bojchevski and Stephan Günnemann,
    published at the 6th International Conference on Learning Representations (ICLR), 2018.

    Copyright (C) 2018
    Aleksandar Bojchevski
    Technical University of Munich
    """
    def __init__(self, A, X, L, K=1, p_val=0.10, p_test=0.05, n_hidden=None,
                 max_iter=2000, tolerance=100, scale=False, seed=0, verbose=True):
        """
        Parameters
        ----------
        A : scipy.sparse.spmatrix
            Sparse unweighted adjacency matrix
        X : scipy.sparse.spmatrix
            Sparse attribute matirx
        L : int
            Dimensionality of the node embeddings
        K : int
            Maximum distance to consider
        p_val : float
            Percent of edges in the validation set, 0 <= p_val < 1
        p_test : float
            Percent of edges in the test set, 0 <= p_test < 1
        n_hidden : list(int)
            A list specifying the size of each hidden layer, default n_hidden=[512]
        max_iter :  int
            Maximum number of epoch for which to run gradient descent
        tolerance : int
            Used for early stopping. Number of epoch to wait for the score to improve on the validation set
        scale : bool
            Whether to apply the up-scaling terms.
        seed : int
            Random seed used to split the edges into train-val-test set
        verbose : bool
            Verbosity.
        """
        tf.reset_default_graph()
        tf.set_random_seed(seed)

        self.X = tf.SparseTensor(*sparse_feeder(X.astype(np.float32)))
        self.N, self.D = X.shape
        self.L = L
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.scale = scale
        self.verbose = verbose

        if n_hidden is None:
            n_hidden = [512]
        self.n_hidden = n_hidden

        # hold out some validation and/or test edges
        # pre-compute the hops for each node for more efficient sampling
        if p_val + p_test > 0:
            train_ones, val_ones, val_zeros, test_ones, test_zeros = train_val_test_split_adjacency(
                A=A, p_val=p_val, p_test=p_test, seed=seed, neg_mul=1, every_node=True, connected=False,
                undirected=(A != A.T).nnz == 0)
            A_train = edges_to_sparse(train_ones, self.N)
            hops = get_hops(A_train, K)
        else:
            hops = get_hops(A, K)

        scale_terms = {h if h != -1 else max(hops.keys()) + 1:
                           hops[h].sum(1).A1 if h != -1 else hops[1].shape[0] - hops[h].sum(1).A1
                       for h in hops}

        self.__build()
        self.__dataset_generator(hops, scale_terms)
        self.__build_loss()

        # setup the validation set for easy evaluation
        if p_val > 0:
            val_edges = np.row_stack((val_ones, val_zeros))
            self.neg_val_energy = -self.energy_kl(val_edges)
            self.val_ground_truth = A[val_edges[:, 0], val_edges[:, 1]].A1
            self.early_stopping = True
        else:
            self.early_stopping = False

        # setup the test set for easy evaluation
        if p_test > 0:
            test_edges = np.row_stack((test_ones, test_zeros))
            self.neg_test_energy = -self.energy_kl(test_edges)
            self.test_ground_truth = A[test_edges[:, 0], test_edges[:, 1]].A1

    def __build(self):
        w_init = tf.contrib.layers.xavier_initializer

        sizes = [self.D] + self.n_hidden

        for i in range(1, len(sizes)):
            W = tf.get_variable(name='W{}'.format(i), shape=[sizes[i - 1], sizes[i]], dtype=tf.float32,
                                initializer=w_init())
            b = tf.get_variable(name='b{}'.format(i), shape=[sizes[i]], dtype=tf.float32, initializer=w_init())

            if i == 1:
                encoded = tf.sparse_tensor_dense_matmul(self.X, W) + b
            else:
                encoded = tf.matmul(encoded, W) + b

            encoded = tf.nn.relu(encoded)

        W_mu = tf.get_variable(name='W_mu', shape=[sizes[-1], self.L], dtype=tf.float32, initializer=w_init())
        b_mu = tf.get_variable(name='b_mu', shape=[self.L], dtype=tf.float32, initializer=w_init())
        self.mu = tf.matmul(encoded, W_mu) + b_mu

        W_sigma = tf.get_variable(name='W_sigma', shape=[sizes[-1], self.L], dtype=tf.float32, initializer=w_init())
        b_sigma = tf.get_variable(name='b_sigma', shape=[self.L], dtype=tf.float32, initializer=w_init())
        log_sigma = tf.matmul(encoded, W_sigma) + b_sigma
        self.sigma = tf.nn.elu(log_sigma) + 1 + 1e-14

    def __build_loss(self):
        hop_pos = tf.stack([self.triplets[:, 0], self.triplets[:, 1]], 1)
        hop_neg = tf.stack([self.triplets[:, 0], self.triplets[:, 2]], 1)
        eng_pos = self.energy_kl(hop_pos)
        eng_neg = self.energy_kl(hop_neg)
        energy = tf.square(eng_pos) + tf.exp(-eng_neg)

        if self.scale:
            self.loss = tf.reduce_mean(energy * self.scale_terms)
        else:
            self.loss = tf.reduce_mean(energy)

    def energy_kl(self, pairs):
        """
        Computes the energy of a set of node pairs as the KL divergence between their respective Gaussian embeddings.

        Parameters
        ----------
        pairs : array-like, shape [?, 2]
            The edges/non-edges for which the energy is calculated

        Returns
        -------
        energy : array-like, shape [?]
            The energy of each pair given the currently learned model
        """
        ij_mu = tf.nn.embedding_lookup(self.mu, pairs)
        ij_sigma = tf.nn.embedding_lookup(self.sigma, pairs)

        sigma_ratio = ij_sigma[:, 1] / ij_sigma[:, 0]
        trace_fac = tf.reduce_sum(sigma_ratio, 1)
        log_det = tf.reduce_sum(tf.log(sigma_ratio + 1e-14), 1)

        mu_diff_sq = tf.reduce_sum(tf.square(ij_mu[:, 0] - ij_mu[:, 1]) / ij_sigma[:, 0], 1)

        return 0.5 * (trace_fac + mu_diff_sq - self.L - log_det)

    def __dataset_generator(self, hops, scale_terms):
        """
        Generates a set of triplets and associated scaling terms by:
            1. Sampling for each node a set of nodes from each of its neighborhoods
            2. Forming all implied pairwise constraints

        Uses tf.Dataset API to perform the sampling in a separate thread for increased speed.

        Parameters
        ----------
        hops : dict
            A dictionary where each 1, 2, ... K, neighborhoods are saved as sparse matrices
        scale_terms : dict
            The appropriate up-scaling terms to ensure unbiased estimates for each neighbourhood
        Returns
        -------
        """
        def gen():
            while True:
                yield to_triplets(sample_all_hops(hops), scale_terms)

        dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.float32), ([None, 3], [None]))
        self.triplets, self.scale_terms = dataset.prefetch(1).make_one_shot_iterator().get_next()

    def __save_vars(self, sess):
        """
        Saves all the trainable variables in memory. Used for early stopping.

        Parameters
        ----------
        sess : tf.Session
            Tensorflow session used for training
        """
        self.saved_vars = {var.name: (var, sess.run(var)) for var in tf.trainable_variables()}

    def __restore_vars(self, sess):
        """
        Restores all the trainable variables from memory. Used for early stopping.
        Parameters
        ----------
        sess : tf.Session
            Tensorflow session used for training
        """
        for name in self.saved_vars:
                sess.run(tf.assign(self.saved_vars[name][0], self.saved_vars[name][1]))

    def train(self, gpu_list='0'):
        """
        Trains the model.

        Returns
        -------
            sess : tf.Session
            Tensorflow session that can be used to obtain the trained embeddings
        """
        val_max = -1.0
        tolerance = self.tolerance

        train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list=gpu_list,
                                                                          allow_growth=True)))
        sess.run(tf.global_variables_initializer())

        for epoch in range(self.max_iter):
            loss, _ = sess.run([self.loss, train_op])

            if self.early_stopping:
                val_auc, val_ap = score_link_prediction(self.val_ground_truth, sess.run(self.neg_val_energy))

                if self.verbose:
                    print('epoch: {:3d}, loss: {:.4f}, val_auc: {:.4f}, val_ap: {:.4f}'.format(epoch, loss, val_auc, val_ap))

                if val_auc + val_ap > val_max:
                    val_max = val_auc + val_ap
                    tolerance = self.tolerance
                    self.__save_vars(sess)
                else:
                    tolerance -= 1

                if tolerance == 0:
                    break
            else:
                if self.verbose:
                    print('epoch: {:3d}, loss: {:.4f}'.format(epoch, loss))

        if self.early_stopping:
            self.__restore_vars(sess)

        return sess
