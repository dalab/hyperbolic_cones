#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Python implementation of Simple Euclidean embeddings which are trained according to [1] using the Euclidean distance.

.. [1] Maximilian Nickel, Douwe Kiela - "Poincaré Embeddings for Learning Hierarchical Representations"
    https://arxiv.org/abs/1705.08039
"""


from dag_emb_model import *

try:
    from autograd import grad  # Only required for optionally verifying gradients while training
    from autograd import numpy as grad_np
    AUTOGRAD_PRESENT = True
except ImportError:
    AUTOGRAD_PRESENT = False


class EuclSimpleModel(DAGEmbeddingModel):
    def __init__(self,
                 train_data,
                 dim=50,
                 init_range=(-0.0001, 0.0001),
                 lr=0.1,
                 burn_in=10,
                 seed=0,
                 logger=None,

                 num_negative=10,
                 ### How to sample negatives for an edge (u,v)
                 neg_sampl_strategy='true_neg',  # 'all' (all nodes for negative sampling) or 'true_neg' (only nodes not connected)
                 where_not_to_sample='ancestors',  # both or ancestors or children. Has no effect if neg_sampl_strategy = 'all'.
                 neg_edges_attach='child',  # How to form negative edges: 'parent' (u,v') or 'child' (u', v) or 'both'
                 always_v_in_neg=True,  # always include the true edge (u,v) as negative.
                 neg_sampling_power=0.0,  # 0 for uniform, 1 for unigram, 0.75 for word2vec
                 ):

        super().__init__(train_data=train_data,
                         dim=dim,
                         logger=logger,
                         init_range=init_range,
                         lr=lr,
                         opt='sgd',
                         burn_in=burn_in,
                         seed=seed,
                         BatchClass=EuclNIPSBatch,
                         KeyedVectorsClass=EuclNIPSKeyedVectors,
                         num_negative=num_negative,
                         neg_sampl_strategy=neg_sampl_strategy,
                         where_not_to_sample=where_not_to_sample,
                         always_v_in_neg=always_v_in_neg,
                         neg_sampling_power=neg_sampling_power,
                         neg_edges_attach=neg_edges_attach)

    def _clip_vectors(self, vectors):
        return vectors


    ### For autograd
    def _loss_fn(self, matrix, rels_reversed):
        """Given a numpy array with vectors for u, v and negative samples, computes loss value.

        Parameters
        ----------
        matrix : numpy.array
            Array containing vectors for u, v and negative samples, of shape (2 + negative_size, dim).

        Returns
        -------
        float
            Computed loss value.

        Warnings
        --------
        Only used for autograd gradients, since autograd requires a specific function signature.
        """
        vector_u = matrix[0]
        vectors_v = matrix[1:]
        euclidean_dists = grad_np.linalg.norm(vector_u - vectors_v, axis=1)
        return EuclSimpleModel._nll_loss_fn(euclidean_dists)


    @staticmethod
    def _nll_loss_fn(euclidean_dists):
        """
        Parameters
        ----------
        poincare_dists : numpy.array
            All distances d(u,v) and d(u,v'), where v' is negative. Shape (1 + negative_size).

        Returns
        ----------
        log-likelihood loss function from the NIPS paper, Eq (6).
        """
        exp_negative_distances = grad_np.exp(-euclidean_dists)

        # Remove the value for the true edge (u,v) from the partition function
        return euclidean_dists[0] + grad_np.log(exp_negative_distances[1:].sum())


class EuclNIPSBatch(DAGEmbeddingBatch):
    """Compute Poincare distances, gradients and loss for a training batch.

    Class for computing Poincare distances, gradients and loss for a training batch,
    and storing intermediate state to avoid recomputing multiple times.
    """
    def __init__(self,
                 vectors_u, # (1, dim, batch_size)
                 vectors_v, # (1 + neg_size, dim, batch_size)
                 indices_u,
                 indices_v,
                 rels_reversed,
                 poincare_model):
        super().__init__(
            vectors_u=vectors_u,
            vectors_v=vectors_v,
            indices_u=indices_u,
            indices_v=indices_v,
            rels_reversed=rels_reversed,
            dag_embedding_model=None)

        self.euclidean_dists = None

        self._distances_computed = False
        self._distance_gradients_computed = False
        self.distance_gradients_u = None
        self.distance_gradients_v = None

    def _compute_distances(self):
        """Compute and store norms, euclidean distances and poincare distances between input vectors."""
        if self._distances_computed:
            return
        self.euclidean_dists = np.linalg.norm(self.vectors_u - self.vectors_v, axis=1)  # (1 + neg_size, batch_size)
        self._distances_computed = True


    def _compute_distance_gradients(self):
        """Compute and store partial derivatives of poincare distance d(u, v) w.r.t all u and all v."""
        if self._distance_gradients_computed:
            return
        self._compute_distances()

        # gradient of |u-v| w.r.t. u and v
        self.euclidean_dists = self.euclidean_dists[:, np.newaxis, :] # (1 + neg, 1, batch_size)
        self.distance_gradients_u = (self.vectors_u - self.vectors_v) / self.euclidean_dists # (1 + neg, dim, batch_size)
        self.distance_gradients_v = - self.distance_gradients_u

        self._distance_gradients_computed = True


    def _compute_loss(self):
        """Compute and store loss value for the given batch of examples."""
        if self._loss_computed:
            return
        self._compute_distances()

        # NLL loss from the NIPS paper.
        exp_negative_distances = np.exp(-self.euclidean_dists)  # (1 + neg_size, batch_size)
        # Remove the value for the true edge (u,v) from the partition function
        Z = exp_negative_distances[1:].sum(axis=0)  # (batch_size)
        self.exp_negative_distances = exp_negative_distances  # (1 + neg_size, batch_size)
        self.Z = Z # (batch_size)

        self.pos_loss = self.euclidean_dists[0].sum()
        self.neg_loss = np.log(self.Z).sum()
        self.loss = self.pos_loss + self.neg_loss  # scalar


        self._loss_computed = True


    def _compute_loss_gradients(self):
        """Compute and store gradients of loss function for all input vectors."""
        if self._loss_gradients_computed:
            return
        self._compute_distances()
        self._compute_distance_gradients()
        self._compute_loss()

        self._compute_nll_loss_gradients()

        assert not np.isnan(self.loss_gradients_u).any()
        assert not np.isnan(self.loss_gradients_v).any()
        self._loss_gradients_computed = True


    def _compute_nll_loss_gradients(self):
        gradients_v = - self.exp_negative_distances[:, np.newaxis, :] / self.Z * self.distance_gradients_v # (1 + neg_size, dim, batch_size)
        # Remove the value for the true edge (u,v) from the partition function
        gradients_v[0] = self.distance_gradients_v[0]

        gradients_u = - self.exp_negative_distances[:, np.newaxis, :] / self.Z * self.distance_gradients_u # (1 + neg_size, dim, batch_size)
        # Remove the value for the true edge (u,v) from the partition function
        gradients_u = self.distance_gradients_u[0] + gradients_u[1:].sum(axis=0) # (dim, batch_size)

        self.loss_gradients_u = gradients_u
        self.loss_gradients_v = gradients_v



class EuclNIPSKeyedVectors(DAGEmbeddingKeyedVectors):
    """Class to contain vectors and vocab for the :class:`~PoincareModel` training class.
    Used to perform operations on the vectors such as vector lookup, distance etc.
    Inspired from KeyedVectorsBase.
    """
    def __init__(self):
        super(EuclNIPSKeyedVectors, self).__init__()


    def vector_distance_batch(self, vector_1, vectors_all):
        """
        Return poincare distances between one vector and a set of other vectors.

        Parameters
        ----------
        vector_1 : numpy.array
            vector from which Poincare distances are to be computed.
            expected shape (dim,)
        vectors_all : numpy.array
            for each row in vectors_all, distance from vector_1 is computed.
            expected shape (num_vectors, dim)

        Returns
        -------
        numpy.array
            Contains Poincare distance between vector_1 and each row in vectors_all.
            shape (num_vectors,)

        """
        return np.linalg.norm(vector_1 - vectors_all, axis=1)

    def is_a_scores_vector_batch(self, alpha, parent_vectors, other_vectors, rel_reversed):
        euclidean_dists = np.linalg.norm(parent_vectors - other_vectors, axis=1)
        parent_norms = np.linalg.norm(parent_vectors, axis=1)
        other_norms = np.linalg.norm(other_vectors, axis=1)
        sign = 1
        if rel_reversed:
            sign = -1
        return (1 + alpha * sign * (parent_norms - other_norms)) * euclidean_dists
