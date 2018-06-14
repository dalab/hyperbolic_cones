#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import math
import numpy as np
from collections import defaultdict, Counter
from numpy import random as np_random
from six import string_types

from gensim import utils, matutils
from GensimBasicKeyedVectors import Vocab,KeyedVectorsBase

try:
    from autograd import grad  # Only required for optionally verifying gradients while training
    from autograd import numpy as grad_np
    AUTOGRAD_PRESENT = True
except ImportError:
    AUTOGRAD_PRESENT = False


class DAGEmbeddingBatch(object):
    """Compute gradients and loss for a training batch.

    Class for computing gradients and loss for a training batch and storing intermediate state to
    avoid recomputing multiple times.
    """
    def __init__(self,
                 vectors_u,
                 vectors_v,
                 indices_u,
                 indices_v,
                 rels_reversed,
                 dag_embedding_model):
        """
        Initialize instance with sets of vectors for which distances are to be computed.

        Parameters
        ----------
        vectors_u : numpy.array
            Vectors of all nodes `u` in the batch.
            Expected shape (1, dim, batch_size).
        vectors_v : numpy.array
            Vectors of all positively related nodes `v` and negatively sampled nodes `v'`,
            for each node `u` in the batch.
            Expected shape (1 + neg_size, dim, batch_size).
        indices_u : list
            List of node indices for each of the vectors in `vectors_u`.
        indices_v : list
            Nested list of lists, each of which is a  list of node indices
            for each of the vectors in `vectors_v` for a specific node `u`.
        rels_reversed : bool
            If the true edges are from u to v, or reversed, from v to u.
        """
        self.vectors_u = vectors_u  # (1, dim, batch_size)
        self.vectors_v = vectors_v  # (1 + neg_size, dim, batch_size)
        self.indices_u = indices_u
        self.indices_v = indices_v

        self.norms_u = np.linalg.norm(self.vectors_u, axis=1)  # (1, batch_size)
        self.norms_u_sq = self.norms_u ** 2

        self.norms_v = np.linalg.norm(self.vectors_v, axis=1)  # (1 + neg_size, batch_size)
        self.norms_v_sq = self.norms_v ** 2

        self.one_minus_norms_sq_u = 1 - self.norms_u_sq  # (1, batch_size)
        self.one_minus_norms_sq_v = 1 - self.norms_v_sq  # (1 + neg_size, batch_size)

        self.rels_reversed = rels_reversed

        self.loss_gradients_u = None
        self.loss_gradients_v = None

        self.loss = None
        self.pos_loss = None # positive part of the loss
        self.neg_loss = None # negative part of the loss

        self._loss_gradients_computed = False
        self._loss_computed = False

    def compute_all(self):
        """Convenience method to perform all computations."""
        self._compute_loss()
        self._compute_loss_gradients()

    def _compute_loss(self):
        """Compute and store loss value for the given batch of examples."""
        raise NotImplementedError

    def _compute_loss_gradients(self):
        """Compute and store gradients of loss function for all input vectors."""
        raise NotImplementedError


class DAGEmbeddingKeyedVectors(KeyedVectorsBase):
    """Class to contain vectors and vocab for the
     :class:`~gensim.models.poincare.DAGEmbeddingKeyedVectorsModel` training class.
        Used to perform operations on the vectors such as vector lookup, distance etc.
    """
    def __init__(self):
        super(DAGEmbeddingKeyedVectors, self).__init__()

    def vector_distance(self, vector_1, vector_2):
        """
        Return poincare distance between two input vectors. Convenience method over `vector_distance_batch`.

        Parameters
        ----------
        vector_1 : numpy.array
            input vector
        vector_2 : numpy.array
            input vector

        Returns
        -------
        numpy.float
            Distance between `vector_1` and `vector_2`.

        """
        return DAGEmbeddingKeyedVectors.vector_distance_batch(vector_1, vector_2[np.newaxis, :])[0]


    def distance(self, w1, w2):
        """
        Return distance between vectors for nodes `w1` and `w2`.

        Parameters
        ----------
        w1 : str or int
            Key for first node.
        w2 : str or int
            Key for second node.

        Returns
        -------
        float
            distance between the vectors for nodes `w1` and `w2`.

        Examples
        --------

        >>> model.distance('mammal.n.01', 'carnivore.n.01')
        2.13

        Notes
        -----
        Raises KeyError if either of `w1` and `w2` is absent from vocab.

        """
        vector_1 = self.word_vec(w1)
        vector_2 = self.word_vec(w2)
        return self.vector_distance(vector_1, vector_2)


    def most_similar(self, node_or_vector, topn=10, restrict_vocab=None):
        """
        Find the top-N most similar nodes to the given node or vector, sorted in increasing order of distance.

        Parameters
        ----------

        node_or_vector : str/int or numpy.array
            node key or vector for which similar nodes are to be found.
        topn : int or None, optional
            number of similar nodes to return, if `None`, returns all.
        restrict_vocab : int or None, optional
            Optional integer which limits the range of vectors which are searched for most-similar values.
            For example, restrict_vocab=10000 would only check the first 10000 node vectors in the vocabulary order.
            This may be meaningful if vocabulary is sorted by descending frequency.

        Returns
        --------
        list of tuples (str, float)
            List of tuples containing (node, distance) pairs in increasing order of distance.

        Examples
        --------
        >>> vectors.most_similar('lion.n.01')
        [('lion_cub.n.01', 0.4484), ('lionet.n.01', 0.6552), ...]

        """
        if not restrict_vocab:
            all_distances = self.distances(node_or_vector)
        else:
            nodes_to_use = self.index2word[:restrict_vocab]
            all_distances = self.distances(node_or_vector, nodes_to_use)

        if isinstance(node_or_vector, string_types + (int,)):
            node_index = self.vocab[node_or_vector].index
        else:
            node_index = None
        if not topn:
            closest_indices = matutils.argsort(all_distances)
        else:
            closest_indices = matutils.argsort(all_distances, topn=1 + topn)
        result = [
            (self.index2word[index], float(all_distances[index]))
            for index in closest_indices if (not node_index or index != node_index)  # ignore the input node
        ]
        if topn:
            result = result[:topn]
        return result


    def vector_distance_batch(self, vector_1, vectors_all):
        """
        Return distances between one vector and a set of other vectors.

        Parameters
        ----------
        vector_1 : numpy.array
            vector from which distances are to be computed.
            expected shape (dim,)
        vectors_all : numpy.array
            for each row in vectors_all, distance from vector_1 is computed.
            expected shape (num_vectors, dim)

        Returns
        -------
        numpy.array
            Contains distance between vector_1 and each row in vectors_all.
            shape (num_vectors,)

        """
        raise NotImplementedError


    def distances_from_indices(self, node_index, other_indices=()):
        assert node_index < len(self.syn0)
        input_vector = self.syn0[node_index]
        if not other_indices:
            other_vectors = self.syn0
        else:
            other_vectors = self.syn0[other_indices]
        return self.vector_distance_batch(input_vector, other_vectors)


    def is_a_scores_vector_batch(self, alpha, parent_vectors, other_vectors, rel_reversed):
        raise NotImplementedError


    def is_a_scores_from_indices(self, alpha, parent_indices, other_indices, rel_reversed):
        parent_vectors = self.syn0[parent_indices]
        other_vectors = self.syn0[other_indices]
        return self.is_a_scores_vector_batch(alpha, parent_vectors, other_vectors, rel_reversed)


    def distances(self, node_or_vector, other_nodes=()):
        """
        Compute distances from given node or vector to all nodes in `other_nodes`.
        If `other_nodes` is empty, return distance between `node_or_vector` and all nodes in vocab.

        Parameters
        ----------
        node_or_vector : str/int or numpy.array
            Node key or vector from which distances are to be computed.

        other_nodes : iterable of str/int or None
            For each node in `other_nodes` distance from `node_or_vector` is computed.
            If None or empty, distance of `node_or_vector` from all nodes in vocab is computed (including itself).

        Returns
        -------
        numpy.array
            Array containing distances to all nodes in `other_nodes` from input `node_or_vector`,
            in the same order as `other_nodes`.

        Examples
        --------

        >>> model.distances('mammal.n.01', ['carnivore.n.01', 'dog.n.01'])
        np.array([2.1199, 2.0710]

        >>> model.distances('mammal.n.01')
        np.array([0.43753847, 3.67973852, ..., 6.66172886])

        Notes
        -----
        Raises KeyError if either `node_or_vector` or any node in `other_nodes` is absent from vocab.

        """
        if isinstance(node_or_vector, string_types):
            input_vector = self.word_vec(node_or_vector)
        else:
            input_vector = node_or_vector
        if other_nodes == None:
            other_vectors = self.syn0
        else:
            other_indices = [self.vocab[node].index for node in other_nodes]
            other_vectors = self.syn0[other_indices]
        return self.vector_distance_batch(input_vector, other_vectors)

###################################################################################################
class DAGEmbeddingModel(utils.SaveLoad):
    """Class for training, using and evaluating DAG Embeddings.

    The model can be stored/loaded via its :meth:`~DAGEmbeddingModel.save`
    and :meth:`~DAGEmbeddingModel.load` methods, or stored/loaded in the word2vec format
    via `model.kv.save_word2vec_format` and :meth:`~DAGEmbeddingKeyedVectors.load_word2vec_format`.

    Note that training cannot be resumed from a model loaded via `load_word2vec_format`, if you wish to train further,
    use :meth:`~DAGEmbeddingModel.save` and :meth:`~DAGEmbeddingModel.load`
    methods instead.
    """
    def __init__(self,
                 train_data,
                 dim=50,  # Number of dimensions of the trained model.
                 init_range=(-0.0001, 0.0001),  # Range within which the vectors are randomly initialized.
                 lr=0.1,  # Learning rate for training.
                 opt='',  # rsgd or exp_map or sgd
                 burn_in=50,  # Number of epochs to use for burn-in initialization (0 means no burn-in).
                 seed=0,
                 logger=None,
                 BatchClass=DAGEmbeddingBatch,
                 KeyedVectorsClass=DAGEmbeddingKeyedVectors,

                 num_negative=10,  # Number of negative samples to use.
                 ### How to sample negatives for an edge (u,v)
                 neg_sampl_strategy='true_neg',  # 'all' (all nodes used for negative sampling) or 'true_neg' (only not connected nodes)
                                                 # 'all_non_leaves' or 'true_neg_non_leaves'
                 where_not_to_sample='ancestors',  # both or ancestors or children. Has no effect if neg_sampl_strategy = 'all'.
                 neg_edges_attach='child',  # How to form negative edges: 'parent' (u,v') or 'child' (u', v) or 'both'
                 always_v_in_neg=True,  # always include the true edge (u,v) as negative.
                 neg_sampling_power=0.0,  # 0 for uniform, 1 for unigram, 0.75 for word2vec
                 ):
        """Initialize and train a DAG embedding model from an iterable of relations.

        Parameters
        ----------
        train_data : iterable of (str, str)
            Iterable of relations, e.g. a list of tuples, or a Relations instance streaming from a file.
            Note that the relations are treated as ordered pairs, i.e. a relation (a, b) does not imply the
            opposite relation (b, a). In case the relations are symmetric, the data should contain both relations
            (a, b) and (b, a).
        """

        self.logger = logger

        self.train_data = train_data
        self.kv = KeyedVectorsClass()
        self.BatchClass = BatchClass

        self.dim = dim
        self.only_leaves_updated = False

        self.train_lr = lr  # Learning rate for training
        self.lr = lr  # Current learning rate

        self.opt = opt
        assert self.opt in ['rsgd', 'exp_map', 'sgd']

        self.burn_in_lr = self.lr / 10  # Learning rate for burn-in
        self.burn_in = burn_in
        self._burn_in_done = False

        self.num_check_gradients = 0 # Check up to 10 gradients using Autograd
        self._loss_grad = None

        self.where_not_to_sample = where_not_to_sample
        assert self.where_not_to_sample in ['both', 'ancestors', 'children']
        self.neg_edges_attach = neg_edges_attach
        assert self.neg_edges_attach in ['parent', 'child', 'both']

        self.always_v_in_neg = always_v_in_neg

        self.num_negative = num_negative
        self._neg_sampling_power = neg_sampling_power
        assert self._neg_sampling_power >= 0 and self._neg_sampling_power <= 2
        self.neg_sampl_strategy = neg_sampl_strategy
        assert self.neg_sampl_strategy in ['all', 'true_neg', 'all_non_leaves', 'true_neg_non_leaves']

        self._np_rand = np_random.RandomState(seed)

        self._load_relations_and_indexes()
        self._init_embeddings(init_range)


    def _load_relations_and_indexes(self):
        """Load relations from the train data and build vocab."""
        self.kv.vocab = {} # word -> (index, count)
        self.kv.index2word = [] # index -> word
        self.all_relations = []  # List of all relation pairs
        self.adjacent_nodes = defaultdict(set)  # Mapping from node index to its neighboring node indices
        if '_non_leaves' in self.neg_sampl_strategy:
            self.non_leaves_indices_set = set()
            self.non_leaves_adjacent_nodes = defaultdict(set)

        self.logger.info("Loading relations from train data..")
        for relation in self.train_data:
            if len(relation) != 2:
                raise ValueError('Relation pair "%s" should have exactly two items' % repr(relation))
            # Ignore self-edges
            assert relation[0] != relation[1]

            for item in relation:
                if item not in self.kv.vocab:
                    self.kv.vocab[item] = Vocab(count=1, index=len(self.kv.index2word))
                    self.kv.index2word.append(item)

            # Like in https://github.com/facebookresearch/poincare-embeddings
            self.kv.vocab[relation[0]].count += 1

            node_1, node_2 = relation # Edge direction : node1 -> node2, swapped in the csv file, but correctly read in PoincareRelations.
            node_1_index, node_2_index = self.kv.vocab[node_1].index, self.kv.vocab[node_2].index

            if self.where_not_to_sample in ['both', 'children']:
                self.adjacent_nodes[node_1_index].add(node_2_index)
            if self.where_not_to_sample in ['both', 'ancestors']:
                self.adjacent_nodes[node_2_index].add(node_1_index)

            if '_non_leaves' in self.neg_sampl_strategy:
                self.non_leaves_indices_set.add(node_1_index)

            self.all_relations.append((node_1_index, node_2_index))

        for node_idx in range(len(self.kv.index2word)):
            self.adjacent_nodes[node_idx].add(node_idx) # Do not sample current node

        if '_non_leaves' in self.neg_sampl_strategy:
            for node_idx in range(len(self.kv.index2word)):
                self.non_leaves_adjacent_nodes[node_idx].add(node_idx)
                for adj_node_idx in self.adjacent_nodes[node_idx]:
                    if adj_node_idx in self.non_leaves_indices_set:
                        self.non_leaves_adjacent_nodes[node_idx].add(adj_node_idx)

        self.logger.info("Loaded %d relations from train data, %d nodes",
                    len(self.all_relations), len(self.kv.vocab))

        self.indices_set = set((range(len(self.kv.index2word))))  # Set of all node indices

        freq_array = np.array([self.kv.vocab[self.kv.index2word[i]].count
                                for i in range(len(self.kv.index2word))], dtype=np.float64)
        unigrams_at_power_array = np.power(freq_array, self._neg_sampling_power)

        self._node_probabilities = unigrams_at_power_array / unigrams_at_power_array.sum()
        self._node_probabilities_cumsum = np.cumsum(self._node_probabilities)

        if '_non_leaves' in self.neg_sampl_strategy:
            self.non_leaves_indices_array = np.array(list(self.non_leaves_indices_set))
            unigrams_at_power_array_non_leaves = unigrams_at_power_array[self.non_leaves_indices_array]
            self._node_probabilities_non_leaves = unigrams_at_power_array_non_leaves / \
                                                  unigrams_at_power_array_non_leaves.sum()
            self._node_probabilities_cumsum_non_leaves = np.cumsum(self._node_probabilities_non_leaves)


    def _init_embeddings(self, init_range):
        """Randomly initialize vectors for the items in the vocab."""
        shape = (len(self.kv.index2word), self.dim)
        self.kv.syn0 = self._np_rand.uniform(init_range[0], init_range[1], shape).astype(np.float64)


    def _sample_negatives(self, node_index, connected_node):
        """Return a sample of negatives for the given node.

        Parameters
        ----------
        node_index : int
            Index of the positive node for which negative samples are to be returned.

        Returns
        -------
        numpy.array
            Array of shape (self.num_negative,) containing indices of negative nodes for the given node index.

        """
        k = self.num_negative  # num negatives

        if self.neg_sampl_strategy == 'all' or len(self.adjacent_nodes[node_index]) == len(self.indices_set): # root node
            uniform_0_1_numbers = self._np_rand.random_sample(self.num_negative)
            negs = list(np.searchsorted(self._node_probabilities_cumsum, uniform_0_1_numbers))
        elif self.neg_sampl_strategy == 'all_non_leaves':
            uniform_0_1_numbers = self._np_rand.random_sample(self.num_negative)
            negs = list(np.searchsorted(self._node_probabilities_cumsum_non_leaves, uniform_0_1_numbers))
            negs = self.non_leaves_indices_array[negs]

        elif self.neg_sampl_strategy == 'true_neg':
            n = len(self.indices_set)
            a = len(self.adjacent_nodes[node_index])

            gamma = float(n) / (n - a)

            if gamma > n / (max(k,1) * math.log(n)):
                # Very expensive branch: O(n + k*log(n)). Should be avoided when possible.

                valid_negatives = np.array(list(self.indices_set - self.adjacent_nodes[node_index])) # O(n). Includes node_index.
                valid_node_probs = self._node_probabilities[valid_negatives]
                valid_node_probs = valid_node_probs / valid_node_probs.sum()
                valid_node_cumsum = np.cumsum(valid_node_probs) # O(n)
                uniform_0_1_numbers = self._np_rand.random_sample(k)
                valid_negative_indices = np.searchsorted(valid_node_cumsum, uniform_0_1_numbers) # O(k * log n)
                negs = list(valid_negatives[valid_negative_indices])
            else:
                # Less expensive branch: O(n / (n-a) * k * log n)
                # we sample gamma * k negatives and hope to find at least k true negatives
                negatives = []
                remain_to_sample = k
                while remain_to_sample > 0:
                    num_to_sample = int(gamma * remain_to_sample)
                    uniform_0_1_numbers = self._np_rand.random_sample(num_to_sample)
                    new_potential_negatives =\
                        np.searchsorted(self._node_probabilities_cumsum, uniform_0_1_numbers)  # O(gamma * k * log n)

                    # time complexity O(gamma * k),
                    # but len(new_good_negatives) is in expectation (1 - a/n) * len(new_potential_candidates)
                    new_good_negatives = [x for x in new_potential_negatives
                                          if x not in self.adjacent_nodes[node_index]]
                    num_new_good_negatives =  min(len(new_good_negatives), remain_to_sample)
                    negatives.extend(new_good_negatives[0 : num_new_good_negatives])
                    remain_to_sample -= num_new_good_negatives
                negs = negatives

        elif self.neg_sampl_strategy == 'true_neg_non_leaves':
            n = len(self.non_leaves_indices_set)
            a = len(self.non_leaves_adjacent_nodes[node_index])

            gamma = float(n) / (n - a)

            if gamma > n / (max(k,1) * math.log(n)):
                # Very expensive branch: O(n + k*log(n)). Should be avoided when possible.

                valid_negatives = np.array(list(self.non_leaves_indices_set - self.non_leaves_adjacent_nodes[node_index])) # O(n). Includes node_index.
                valid_node_probs = self._node_probabilities_non_leaves[valid_negatives]
                valid_node_probs = valid_node_probs / valid_node_probs.sum()
                valid_node_cumsum = np.cumsum(valid_node_probs) # O(n)
                uniform_0_1_numbers = self._np_rand.random_sample(k)
                valid_negative_indices = np.searchsorted(valid_node_cumsum, uniform_0_1_numbers) # O(k * log n)
                negs = list(valid_negatives[valid_negative_indices])
            else:
                # Less expensive branch: O(n / (n-a) * k * log n)
                # we sample gamma * k negatives and hope to find at least k true negatives
                negatives = []
                remain_to_sample = k
                while remain_to_sample > 0:
                    num_to_sample = int(gamma * remain_to_sample)
                    uniform_0_1_numbers = self._np_rand.random_sample(num_to_sample)
                    new_potential_negatives =\
                        np.searchsorted(self._node_probabilities_cumsum_non_leaves, uniform_0_1_numbers)  # O(gamma * k * log n)
                    new_potential_negatives = self.non_leaves_indices_array[new_potential_negatives]

                    # time complexity O(gamma * k),
                    # but len(new_good_negatives) is in expectation (1 - a/n) * len(new_potential_candidates)
                    new_good_negatives = [x for x in new_potential_negatives
                                          if x not in self.non_leaves_adjacent_nodes[node_index]]
                    num_new_good_negatives =  min(len(new_good_negatives), remain_to_sample)
                    negatives.extend(new_good_negatives[0 : num_new_good_negatives])
                    remain_to_sample -= num_new_good_negatives
                negs = negatives

        # Should we always include 'v' as negative in all batches ?
        if self.always_v_in_neg:
            negs[0] = connected_node
        return negs


    def _prepare_training_batch(self, relations, all_negatives, rels_reversed):
        """Creates training batch and computes gradients and loss for the batch.

        Parameters
        ----------
        relations : list of tuples
            List of tuples of positive examples of the form (node_1_index, node_2_index).
        all_negatives : list of lists
            List of lists of negative samples for each node_1 in the positive examples.
        rels_reversed : bool
            If the relations are from u to v or, reversed, from v to u.

        Returns
        -------
        :class:`~DAGEmbeddingBatchBatch`
            Contains node indices, computed gradients and loss for the batch.
        """
        batch_size = len(relations)
        indices_u, indices_v = [], []

        for relation, negatives in zip(relations, all_negatives):
            u, v = relation
            indices_u.append(u)
            indices_v.append(v)
            indices_v.extend(negatives)

        vectors_u = self.kv.syn0[indices_u].T[np.newaxis, :, :] # (1, dim, batch_size)
        vectors_v = self.kv.syn0[indices_v].reshape((batch_size, 1 + self.num_negative, self.dim))
        vectors_v = vectors_v.swapaxes(0, 1).swapaxes(1, 2) # (1 + neg, dim, batch_size)
        batch = self.BatchClass(vectors_u, vectors_v, indices_u, indices_v, rels_reversed, self)
        batch.compute_all()

        if self.num_check_gradients < 5:
            self.num_check_gradients += 1
            self._check_gradients(relations, all_negatives, batch, rels_reversed)
        return batch


    def _handle_duplicates(self, vector_updates, node_indices):
        """Handles occurrences of multiple updates to the same node in a batch of vector updates.

        Parameters
        ----------
        vector_updates : numpy.array
            Array with each row containing updates to be performed on a certain node.
        node_indices : list
            Node indices on which the above updates are to be performed on.

        Notes
        -----
        Mutates the `vector_updates` array.

        Required because vectors[[2, 1, 2]] += np.array([-0.5, 1.0, 0.5]) performs only the last update
        on the row at index 2.
        """
        counts = Counter(node_indices)
        for node_index, count in counts.items():
            if count == 1 and not (self.only_leaves_updated and node_index in self.non_leaves_indices_set):
                continue
            positions = [i for i, index in enumerate(node_indices) if index == node_index]
            # Move all updates to the same node to the last such update, zeroing all the others
            vector_updates[positions[-1]] = vector_updates[positions].sum(axis=0)
            vector_updates[positions[:-1]] = 0

            if self.only_leaves_updated and node_index in self.non_leaves_indices_set:
                vector_updates[positions[-1]] = 0


    def _clip_vectors(self, vectors):
        """Clip vectors. E.g. to have a norm of less than one for the hyperbolic setting.

        Parameters
        ----------
        vectors : numpy.array
            Can be 1-D,or 2-D (in which case the norm for each row is checked).

        Returns
        -------
        numpy.array
            Array with vectors clipped.
        """
        raise NotImplementedError


    @staticmethod
    def exponential_map(x, v):
        """
        Computes the exponential map exp_x(v).
        :param x, v: d-dimensional tensors, the last dimension is the embedding dimension
        """
        d = x.ndim
        lam_x = 2.0 / (1.0 - np.linalg.norm(x, axis=d-1) ** 2) # d-1 dims
        norm_v = np.linalg.norm(v, axis=d-1) # d-1 dims
        angle = lam_x * norm_v # d-1 dims
        cosh = np.cosh(np.minimum(np.maximum(angle, -20), 20)) # d-1 dims
        tanh = np.tanh(angle) # d-1 dims
        normalized_v = v / np.maximum(norm_v[:, np.newaxis], 1e-6) # d dims
        x_dot_norm_v = (x * normalized_v).sum(axis=d-1)  # d-1 dims
        numerator = (lam_x * (1 + x_dot_norm_v * tanh))[:, np.newaxis] * x + tanh[:, np.newaxis] * normalized_v # d dims
        numitor = 1 / cosh + (lam_x - 1) + lam_x * x_dot_norm_v * tanh # d-1 dims
        return numerator / (numitor[:, np.newaxis])


    def _update_vectors_batch(self, batch):
        """Updates vectors for nodes in the given batch using RSGD.

        Parameters
        ----------
        batch : :class:`~hyperbolic.DAGEmbeddingBatch`
            Batch containing computed loss gradients and node indices of the batch for which
            updates are to be done.
        """
        grad_u, grad_v = batch.loss_gradients_u, batch.loss_gradients_v # grad_u: (dim, batch_size) ; grad_v: (1 + neg_size, dim, batch_size)
        indices_u, indices_v = batch.indices_u, batch.indices_v
        batch_size = len(indices_u)

        if self.opt == 'rsgd' or self.opt == 'exp_map':
            # batch.one_minus_norms_sq_u: (1, batch_size), grad_u: (dim, batch_size)
            u_updates = (self.lr * (batch.one_minus_norms_sq_u ** 2) / 4 * grad_u).T # (batch_size, dim)
            self._handle_duplicates(u_updates, indices_u)

            # batch.one_minus_norms_sq_v: (1 + neg_size, batch_size) ,  grad_v: (1 + neg_size, dim, batch_size)
            v_updates = self.lr * (batch.one_minus_norms_sq_v ** 2)[:, np.newaxis] / 4 * grad_v
            v_updates = v_updates.swapaxes(1, 2).swapaxes(0, 1) # (batch_size, 1 + neg_size, dim)
            v_updates = v_updates.reshape(((1 + self.num_negative) * batch_size, self.dim))
            self._handle_duplicates(v_updates, indices_v)

            if self.opt == 'rsgd':
                self.kv.syn0[indices_u] -= u_updates
                self.kv.syn0[indices_v] -= v_updates
            elif self.opt == 'exp_map':
                self.kv.syn0[indices_u] = self.exponential_map(self.kv.syn0[indices_u], -u_updates)
                self.kv.syn0[indices_v] = self.exponential_map(self.kv.syn0[indices_v], -v_updates)

        elif self.opt == 'sgd':
            u_updates = (self.lr * grad_u).T # (batch_size, dim)
            self._handle_duplicates(u_updates, indices_u)

            v_updates = self.lr * grad_v
            v_updates = v_updates.swapaxes(1, 2).swapaxes(0, 1) # (batch_size, 1 + neg_size, dim)
            v_updates = v_updates.reshape(((1 + self.num_negative) * batch_size, self.dim))
            self._handle_duplicates(v_updates, indices_v)

            self.kv.syn0[indices_u] -= u_updates
            self.kv.syn0[indices_v] -= v_updates

        self.kv.syn0[indices_u] = self._clip_vectors(self.kv.syn0[indices_u])
        self.kv.syn0[indices_v] = self._clip_vectors(self.kv.syn0[indices_v])



    def _train_on_batch(self, relations):
        """Performs training for a single training batch.

        Parameters
        ----------
        relations : list of tuples
            List of tuples of positive examples of the form (node_1_index, node_2_index).

        Returns
        -------
        loss value
        """

        # Sample negatives in a batch: list of lists of negative node indices
        loss = 0.0
        pos_loss = 0.0
        neg_loss = 0.0
        if self.neg_edges_attach in ['child', 'both']:
            relations = [(v,u) for (u,v) in relations] #####
            all_negatives = [self._sample_negatives(relation[0], relation[1]) for relation in relations]
            batch = self._prepare_training_batch(relations, all_negatives, rels_reversed=True)
            self._update_vectors_batch(batch)
            loss += batch.loss
            pos_loss += batch.pos_loss
            neg_loss += batch.neg_loss

        if self.neg_edges_attach in ['parent', 'both']:
            all_negatives = [self._sample_negatives(relation[0], relation[1]) for relation in relations]
            batch = self._prepare_training_batch(relations, all_negatives, rels_reversed=False)
            self._update_vectors_batch(batch)
            loss += batch.loss
            pos_loss += batch.pos_loss
            neg_loss += batch.neg_loss

        return loss, pos_loss, neg_loss


    def _train_batchwise(self, epochs, batch_size, print_every=1):
        """Trains embeddings using specified parameters.

        Parameters
        ----------
        epochs : int
            Number of iterations (epochs) over the corpus.
        batch_size : int, optional
            Number of examples to train on in a single batch.
        print_every : int, optional
            Prints progress and average loss after every `print_every` epochs.
        """
        avg_loss = 0.0
        avg_pos_loss = 0.0
        avg_neg_loss = 0.0
        for epoch in range(1, epochs + 1):
            last_time = time.time()

            relation_indices = list(range(len(self.all_relations)))
            self._np_rand.shuffle(relation_indices)

            avg_loss = 0.0
            avg_pos_loss = 0.0
            avg_neg_loss = 0.0

            for batch_num, i in enumerate(range(0, len(relation_indices), batch_size), start=1):
                batch_indices = relation_indices[i:i + batch_size]
                relations_batch = [self.all_relations[idx] for idx in batch_indices]
                # Train on this mini-batch
                l, pl, nl = self._train_on_batch(relations_batch)
                avg_loss += l
                avg_pos_loss += pl
                avg_neg_loss += nl

            avg_loss *= float(batch_size) / len(relation_indices)
            avg_pos_loss *= float(batch_size) / len(relation_indices)
            avg_neg_loss *= float(batch_size) / len(relation_indices)

            if epoch % print_every == 0:
                self.logger.info('Epoch %d: LOSS: %.2f, POS LOSS: %.2f, NEG LOSS: %.2f' %
                                 (epoch, avg_loss, avg_pos_loss, avg_neg_loss))
                time_taken = time.time() - last_time
                speed = len(relation_indices) / time_taken
                self.logger.info(
                    'Time taken for %d examples: %.2f s, %.2f examples / s'
                    % (len(relation_indices), time_taken, speed))

        return avg_loss, avg_pos_loss, avg_neg_loss


    def train(self, epochs, batch_size, print_every=1):
        """Trains Poincare embeddings using loaded data and model parameters.

        Parameters
        ----------

        batch_size : int, optional
            Number of examples to train on in a single batch.
        epochs : int
            Number of iterations (epochs) over the corpus.
        print_every : int, optional
            Prints progress and average loss after every `print_every` epochs.
        """
        # Some divide-by-zero results are handled explicitly
        old_settings = np.seterr(divide='ignore', invalid='ignore')

        if self.burn_in > 0 and not self._burn_in_done:
            self.logger.info("Starting burn-in (%d epochs)-------------------------------", self.burn_in)
            self.lr = self.burn_in_lr
            self._train_batchwise(
                epochs=self.burn_in, batch_size=batch_size, print_every=print_every)
            self._burn_in_done = True

            freq_array = np.ones(len(self.kv.index2word), dtype=np.float64)
            unigrams_at_power_array = freq_array
            self._node_probabilities = unigrams_at_power_array / unigrams_at_power_array.sum()
            self._node_probabilities_cumsum = np.cumsum(self._node_probabilities)
            self._neg_sampling_power = 0.0 ###########

            self.logger.info("Burn-in finished")

        self.lr = self.train_lr
        self.logger.info("Starting training (%d epochs)----------------------------------------", epochs)
        avg_loss, avg_pos_loss, avg_neg_loss = self._train_batchwise(
            epochs=epochs, batch_size=batch_size, print_every=print_every)
        self.logger.info("Training finished")

        np.seterr(**old_settings)
        return avg_loss, avg_pos_loss, avg_neg_loss


    def save(self, *args, **kwargs):
        """Save complete model to disk, inherited from :class:`gensim.utils.SaveLoad`."""
        self._loss_grad = None  # Can't pickle autograd fn to disk
        super(DAGEmbeddingModel, self).save(*args, **kwargs)


    @classmethod
    def load(cls, *args, **kwargs):
        """Load model from disk, inherited from :class:`~gensim.utils.SaveLoad`."""
        model = super(DAGEmbeddingModel, cls).load(*args, **kwargs)
        return model


    def _check_gradients(self, relations, all_negatives, batch, rels_reversed, tol=1e-6):
        """Compare computed gradients for batch to autograd gradients.

        Parameters
        ----------
        batch : PoincareBatch instance
            Batch for which computed gradients are to checked.
        relations : list of tuples
            List of tuples of positive examples of the form (node_1_index, node_2_index).
        all_negatives : list of lists
            List of lists of negative samples for each node_1 in the positive examples.
        """
        if not AUTOGRAD_PRESENT:
            self.logger.warning('autograd could not be imported, cannot do gradient checking')
            self.logger.warning('please install autograd to enable gradient checking')
            return

        if self._loss_grad is None:
            self._loss_grad = grad(self._loss_fn)

        max_diff = 0.0
        for i, (relation, negatives) in enumerate(zip(relations, all_negatives)):
            u, v = relation
            auto_gradients = self._loss_grad(
                np.vstack((self.kv.syn0[u], self.kv.syn0[[v] + negatives])),
                rels_reversed
            )
            computed_gradients = np.vstack((batch.loss_gradients_u[:, i], batch.loss_gradients_v[:, :, i])) # (1 + neg_size, dim, batch_size)
            diff = np.abs(auto_gradients - computed_gradients).max()
            if diff > max_diff:
                max_diff = diff
                self.logger.info('Max difference between computed gradients and autograd gradients: %.10f. Tol = %.10f' % (max_diff, tol))
        assert max_diff < tol, (
                'Max difference between computed gradients and autograd gradients %.10f, '
                'greater than tolerance %.10f' % (max_diff, tol))


    def _loss_fn(self, matrix, rels_reversed):
        """Given a numpy array with vectors for u, v and negative samples, computes loss value.

        Parameters
        ----------
        matrix : numpy.array
            Array containing vectors for u, v and negative samples, of shape (2 + negative_size, dim).
        rels_reversed : bool

        Returns
        -------
        float
            Computed loss value.

        Warnings
        --------
        Only used for autograd gradients, since autograd requires a specific function signature.
        """
        raise NotImplementedError


