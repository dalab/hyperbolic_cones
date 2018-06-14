#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Python implementation of Poincaré Embeddings [1]_, an embedding that is better at capturing latent hierarchical
information than traditional Euclidean embeddings. The method is described in more detail in [1].

The main use-case is to automatically learn hierarchical representations of nodes from a tree-like structure,
such as a Directed Acyclic Graph, using a transitive closure of the relations. Representations of nodes in a
symmetric graph can also be learned, using an iterable of the direct relations in the graph.

This module allows training a Poincaré Embedding from a training file containing relations of graph in a
csv-like format, or a Python iterable of relations.

.. [1] Maximilian Nickel, Douwe Kiela - "Poincaré Embeddings for Learning Hierarchical Representations"
    https://arxiv.org/abs/1705.08039

Note: This implementation is inspired and extends the open-source Gensim implementation of Poincare Embeddings.
"""

from dag_emb_model import *

try:
    from autograd import grad  # Only required for optionally verifying gradients while training
    from autograd import numpy as grad_np
    AUTOGRAD_PRESENT = True
except ImportError:
    AUTOGRAD_PRESENT = False


class PoincareModel(DAGEmbeddingModel):
    """Class for training, using and evaluating Poincare Embeddings.

    The model can be stored/loaded via its :meth:`~hyperbolic.poincare_model.PoincareModel.save`
    and :meth:`~hyperbolic.poincare_model.PoincareModel.load` methods, or stored/loaded in the word2vec format
    via `model.kv.save_word2vec_format` and :meth:`~hyperbolic.poincare_model.PoincareKeyedVectors.load_word2vec_format`.

    Note that training cannot be resumed from a model loaded via `load_word2vec_format`, if you wish to train further,
    use :meth:`~hyperbolic.poincare_model.PoincareModel.save` and :meth:`~hyperbolic.poincare_model.PoincareModel.load`
    methods instead.
    """
    def __init__(self,
                 train_data,
                 dim=50,
                 init_range=(-0.0001, 0.0001),
                 lr=0.1,
                 opt='rsgd',  # rsgd or exp_map
                 burn_in=10,
                 epsilon=1e-5,
                 seed=0,
                 logger=None,

                 num_negative=10,
                 ### How to sample negatives for an edge (u,v)
                 neg_sampl_strategy='true_neg',  # 'all' (all nodes for negative sampling) or 'true_neg' (only nodes not connected)
                 where_not_to_sample='ancestors',  # both or ancestors or children. Has no effect if neg_sampl_strategy = 'all'.
                 neg_edges_attach='child',  # How to form negative edges: 'parent' (u,v') or 'child' (u', v) or 'both'
                 always_v_in_neg=True,  # always include the true edge (u,v) as negative.
                 neg_sampling_power=0.0,  # 0 for uniform, 1 for unigram, 0.75 for word2vec

                 loss_type='nll',  # 'nll', 'neg', 'maxmargin'
                 maxmargin_margin=1.0,
                 neg_r=2.0,
                 neg_t=1.0,
                 neg_mu=1.0,  # Balancing factor between the positive and negative terms
                 ):
        """Initialize and train a Poincare embedding model from an iterable of relations.

        Parameters
        ----------
        See DAGEmbeddingModel for other parameters.

        epsilon : float, optional
            Constant used for clipping embeddings below a norm of one.

        Examples
        --------
        Initialize a model from a list:

        >>> from poincare_model import PoincareModel
        >>> relations = [('kangaroo', 'marsupial'), ('kangaroo', 'mammal'), ('gib', 'cat')]
        >>> model = PoincareModel(relations, negative=2)

        Initialize a model from a file containing one relation per line:

        >>> from poincare_model import PoincareModel
        >>> from relations import Relations
        >>> from gensim.test.utils import datapath
        >>> file_path = datapath('poincare_hypernyms.tsv')
        >>> model = PoincareModel(Relations(file_path, set()), negative=2)

        See :class:`~hyperbolic.relations.Relations` for more options.
        """
        super().__init__(train_data=train_data,
                         dim=dim,
                         logger=logger,
                         init_range=init_range,
                         lr=lr,
                         opt=opt,
                         burn_in=burn_in,
                         seed=seed,
                         BatchClass=PoincareBatch,
                         KeyedVectorsClass=PoincareKeyedVectors,
                         num_negative=num_negative,
                         neg_sampl_strategy=neg_sampl_strategy,
                         where_not_to_sample=where_not_to_sample,
                         always_v_in_neg=always_v_in_neg,
                         neg_sampling_power=neg_sampling_power,
                         neg_edges_attach=neg_edges_attach)

        self.epsilon = epsilon
        assert self.opt in ['rsgd', 'exp_map']
        self.loss_type = loss_type
        assert self.loss_type in ['nll', 'neg', 'maxmargin']

        self.maxmargin_margin = maxmargin_margin
        self.neg_r = neg_r
        self.neg_t = neg_t
        self.neg_mu = neg_mu


    def _clip_vectors(self, vectors):
        """Clip vectors to have a norm of less than one.

        Parameters
        ----------
        vectors : numpy.array
            Can be 1-D,or 2-D (in which case the norm for each row is checked).

        Returns
        -------
        numpy.array
            Array with norms clipped below 1.
        """

        # u__v_prime: MAP: 0.379;
        # rank: 32.23
        # u_prime__v: MAP: 0.896;
        # rank: 1.80
        # Our clipping
        thresh = 1.0 - self.epsilon
        one_d = len(vectors.shape) == 1
        if one_d:
            norm = np.linalg.norm(vectors)
            if norm < thresh:
                return vectors
            else:
                return thresh * vectors / norm
        else:
            norms = np.linalg.norm(vectors, axis=1)
            if (norms < thresh).all():
                return vectors
            else:
                vectors[norms >= thresh] *= (thresh / norms[norms >= thresh])[:, np.newaxis]
                return vectors

        # Old methods
        # u__v_prime: rank: 32.23;
        # MAP: 0.379
        # u_prime__v: rank: 1.80;
        # MAP: 0.896
        # Our clipping
        # thresh = 1.0 - self.epsilon
        # one_d = len(vectors.shape) == 1
        # if one_d:
        #     norm = np.linalg.norm(vectors)
        #     if norm < thresh:
        #         return vectors
        #     else:
        #         return vectors / (norm + self.epsilon)
        # else:
        #     norms = np.linalg.norm(vectors, axis=1)
        #     if (norms < thresh).all():
        #         return vectors
        #     else:
        #         vectors[norms >= thresh] *= (1.0 / (norms[norms >= thresh] + self.epsilon))[:, np.newaxis]
        #         return vectors


        # u__v_prime: MAP: 0.418;
        # rank: 31.96
        # u_prime__v: MAP: 0.872;
        # rank: 2.06
        ## Original clipping
        # one_d = len(vectors.shape) == 1
        # threshold = 1 - self.epsilon
        # if one_d:
        #     norm = np.linalg.norm(vectors)
        #     if norm < threshold:
        #         return vectors
        #     else:
        #         return vectors / norm - (np.sign(vectors) * self.epsilon)
        # else:
        #     norms = np.linalg.norm(vectors, axis=1)
        #     if (norms < threshold).all():
        #         return vectors
        #     else:
        #         vectors[norms >= threshold] *= (threshold / norms[norms >= threshold])[:, np.newaxis]
        #         vectors[norms >= threshold] -= np.sign(vectors[norms >= threshold]) * self.epsilon
        #         return vectors


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
        norm = grad_np.linalg.norm(vector_u)
        all_norms = grad_np.linalg.norm(vectors_v, axis=1)
        poincare_dists = grad_np.arccosh(
            1 + 2 * (
                (euclidean_dists ** 2) / ((1 - norm ** 2) * (1 - all_norms ** 2))
            )
        )
        if self.loss_type == 'nll':
            return PoincareModel._nll_loss_fn(poincare_dists)
        elif self.loss_type == 'neg':
            return PoincareModel._neg_loss_fn(poincare_dists, self.neg_r, self.neg_t, self.neg_mu)
        elif self.loss_type == 'maxmargin':
            return PoincareModel._maxmargin_loss_fn(poincare_dists, self.maxmargin_margin)
        else:
            raise ValueError('Unknown loss type : ' + self.loss_type)


    @staticmethod
    def _nll_loss_fn(poincare_dists):
        """
        Parameters
        ----------
        poincare_dists : numpy.array
            All distances d(u,v) and d(u,v'), where v' is negative. Shape (1 + negative_size).

        Returns
        ----------
        log-likelihood loss function from the NIPS paper, Eq (6).
        """
        exp_negative_distances = grad_np.exp(-poincare_dists)

        # Remove the value for the true edge (u,v) from the partition function
        # return -grad_np.log(exp_negative_distances[0] / (- exp_negative_distances[0] + exp_negative_distances.sum()))
        return poincare_dists[0] + grad_np.log(exp_negative_distances[1:].sum())


    @staticmethod
    def _neg_loss_fn(poincare_dists, neg_r, neg_t, neg_mu):
        # NEG loss function:
        # loss = - log sigma((r - d(u,v)) / t) - \sum_{v' \in N(u)} log sigma((d(u,v') - r) / t)
        positive_term = grad_np.log(1.0 + grad_np.exp((- neg_r + poincare_dists[0]) / neg_t))
        negative_terms = grad_np.log(1.0 + grad_np.exp((neg_r - poincare_dists[1:]) / neg_t))
        return positive_term + neg_mu * negative_terms.sum()


    @staticmethod
    def _maxmargin_loss_fn(poincare_dists, maxmargin_margin):
        """
        Parameters
        ----------
        poincare_dists : numpy.array
            All distances d(u,v) and d(u,v'), where v' is negative. Shape (1 + negative_size).

        Returns
        ----------
        max-margin loss function: \sum_{v' \in N(u)} max(0, \gamma + d(u,v) - d(u,v'))
        """
        positive_term = poincare_dists[0]
        negative_terms = poincare_dists[1:]
        return grad_np.maximum(0, maxmargin_margin + positive_term - negative_terms).sum()


class PoincareBatch(DAGEmbeddingBatch):
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

        self.gamma = None
        self.poincare_dists = None
        self.euclidean_dists = None

        self._distances_computed = False
        self._distance_gradients_computed = False
        self.distance_gradients_u = None
        self.distance_gradients_v = None

        self.loss_type = poincare_model.loss_type
        self.maxmargin_margin = poincare_model.maxmargin_margin
        self.neg_r = poincare_model.neg_r
        self.neg_t = poincare_model.neg_t
        self.neg_mu = poincare_model.neg_mu


    def _compute_distances(self):
        """Compute and store norms, euclidean distances and poincare distances between input vectors."""
        if self._distances_computed:
            return
        self.euclidean_dists = np.linalg.norm(self.vectors_u - self.vectors_v, axis=1)  # (1 + neg_size, batch_size)
        self.gamma = 1 + 2 * ((self.euclidean_dists ** 2) / (self.one_minus_norms_sq_u * self.one_minus_norms_sq_v))  # (1 + neg_size, batch_size)
        self.poincare_dists = np.arccosh(self.gamma)  # (1 + neg_size, batch_size)
        self._distances_computed = True


    def _compute_distance_gradients(self):
        """Compute and store partial derivatives of poincare distance d(u, v) w.r.t all u and all v."""
        if self._distance_gradients_computed:
            return
        self._compute_distances()

        euclidean_dists_squared = self.euclidean_dists ** 2  # (1 + neg_size, batch_size)
        c_ = (4 / (self.one_minus_norms_sq_u * self.one_minus_norms_sq_v * np.sqrt(self.gamma ** 2 - 1)))[:, np.newaxis, :] # (1 + neg_size, 1, batch_size)
        u_coeffs = ((euclidean_dists_squared + self.one_minus_norms_sq_u) / self.one_minus_norms_sq_u)[:, np.newaxis, :] # (1 + neg_size, 1, batch_size)
        distance_gradients_u = u_coeffs * self.vectors_u - self.vectors_v  # (1 + neg_size, dim, batch_size)
        distance_gradients_u *= c_  # (1 + neg_size, dim, batch_size)

        nan_gradients = self.gamma == 1  # (1 + neg_size, batch_size)
        if nan_gradients.any():
            distance_gradients_u.swapaxes(1, 2)[nan_gradients] = 0
        self.distance_gradients_u = distance_gradients_u # (1 + neg_size, dim, batch_size)

        # (1 + neg_size, 1, batch_size)
        v_coeffs = ((euclidean_dists_squared + self.one_minus_norms_sq_v) / self.one_minus_norms_sq_v)[:, np.newaxis, :]
        distance_gradients_v = v_coeffs * self.vectors_v - self.vectors_u  # (1 + neg_size, dim, batch_size)
        distance_gradients_v *= c_  # (1 + neg_size, dim, batch_size)

        if nan_gradients.any():
            distance_gradients_v.swapaxes(1, 2)[nan_gradients] = 0
        self.distance_gradients_v = distance_gradients_v # (1 + neg_size, dim, batch_size)

        self._distance_gradients_computed = True


    def _compute_loss(self):
        """Compute and store loss value for the given batch of examples."""
        if self._loss_computed:
            return
        self._compute_distances()

        if self.loss_type == 'nll':
            # NLL loss from the NIPS paper.
            exp_negative_distances = np.exp(-self.poincare_dists)  # (1 + neg_size, batch_size)
            # Remove the value for the true edge (u,v) from the partition function
            Z = exp_negative_distances[1:].sum(axis=0)  # (batch_size)
            self.exp_negative_distances = exp_negative_distances  # (1 + neg_size, batch_size)
            self.Z = Z # (batch_size)

            self.pos_loss = self.poincare_dists[0].sum()
            self.neg_loss = np.log(self.Z).sum()
            self.loss = self.pos_loss + self.neg_loss  # scalar

        elif self.loss_type == 'neg':
            # NEG loss function:
            # - log sigma((r - d(u,v)) / t) - \sum_{v' \in N(u)} log sigma((d(u,v') - r) / t)
            positive_term = np.log(1.0 + np.exp((- self.neg_r + self.poincare_dists[0]) / self.neg_t))  # (batch_size)
            negative_terms = self.neg_mu * \
                             np.log(1.0 + np.exp((self.neg_r - self.poincare_dists[1:]) / self.neg_t)) # (1 + neg_size, batch_size)

            self.pos_loss = positive_term.sum()
            self.neg_loss = negative_terms.sum()
            self.loss = self.pos_loss + self.neg_loss  # scalar

        elif self.loss_type == 'maxmargin':
            # max - margin loss function: \sum_{v' \in N(u)} max(0, \gamma + d(u,v) - d(u,v'))
            self.loss = np.maximum(0, self.maxmargin_margin + self.poincare_dists[0] - self.poincare_dists[1:]).sum() # scalar
            self.pos_loss = self.loss
            self.neg_loss = self.loss

        else:
            raise ValueError('Unknown loss type : ' + self.loss_type)

        self._loss_computed = True


    def _compute_loss_gradients(self):
        """Compute and store gradients of loss function for all input vectors."""
        if self._loss_gradients_computed:
            return
        self._compute_distances()
        self._compute_distance_gradients()
        self._compute_loss()

        if self.loss_type == 'nll':
            self._compute_nll_loss_gradients()
        elif self.loss_type == 'neg':
            self._compute_neg_loss_gradients()
        elif self.loss_type == 'maxmargin':
            self._compute_maxmargin_loss_gradients()
        else:
            raise ValueError('Unknown loss type : ' + self.loss_type)

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


    def _compute_neg_loss_gradients(self):
        exp_poincare_dist = np.exp((self.neg_r - self.poincare_dists) / self.neg_t)
        gradients_v = - self.neg_mu /self.neg_t *\
                      (exp_poincare_dist / (1 + exp_poincare_dist))[:, np.newaxis, :] *\
                      self.distance_gradients_v # (1 + neg_size, dim, batch_size)
        gradients_v[0] = 1.0 /self.neg_t *\
                         (1 / (1 + exp_poincare_dist[0]))[np.newaxis, :] *\
                         self.distance_gradients_v[0] # (dim, batch_size)

        gradients_u = - self.neg_mu /self.neg_t *\
                      (exp_poincare_dist / (1 + exp_poincare_dist))[:, np.newaxis, :] *\
                      self.distance_gradients_u # (1 + neg_size, dim, batch_size)
        gradients_u = gradients_u[1:].sum(axis=0) # (dim, batch_size)
        gradients_u += 1.0 /self.neg_t *\
                       (1 / (1 + exp_poincare_dist[0]))[np.newaxis, :] *\
                       self.distance_gradients_u[0] # (dim, batch_size)

        self.loss_gradients_u = gradients_u
        self.loss_gradients_v = gradients_v


    def _compute_maxmargin_loss_gradients(self):

        update_cond = (self.maxmargin_margin + self.poincare_dists[0] - self.poincare_dists >= 0) # (1 + neg_size, batch_size)

        # negative part
        gradients_v = (-1.0 * update_cond)[:, np.newaxis, :] * self.distance_gradients_v # (1 + neg_size, dim, batch_size)
        # positive part
        gradients_v[0] += (1.0 * update_cond).sum(axis=0)[np.newaxis, :] * self.distance_gradients_v[0] # (dim, batch_size)

        # negative part
        gradients_u = ((-1.0 * update_cond)[:, np.newaxis, :] * self.distance_gradients_u).sum(axis=0)
        # positive part
        gradients_u += (1.0 * update_cond).sum(axis=0)[np.newaxis, :] * self.distance_gradients_u[0] # (dim, batch_size)

        self.loss_gradients_u = gradients_u
        self.loss_gradients_v = gradients_v


class PoincareKeyedVectors(DAGEmbeddingKeyedVectors):
    """Class to contain vectors and vocab for the :class:`~PoincareModel` training class.
    Used to perform operations on the vectors such as vector lookup, distance etc.
    Inspired from KeyedVectorsBase.
    """
    def __init__(self):
        super(PoincareKeyedVectors, self).__init__()


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
        euclidean_dists = np.linalg.norm(vector_1 - vectors_all, axis=1)
        norm = np.linalg.norm(vector_1)
        all_norms = np.linalg.norm(vectors_all, axis=1)
        return np.arccosh(
            1 + 2 * (
                (euclidean_dists ** 2) / ((1 - norm ** 2) * (1 - all_norms ** 2))
            )
        )


    def is_a_scores_vector_batch(self, alpha, parent_vectors, other_vectors, rel_reversed):
        euclidean_dists = np.linalg.norm(parent_vectors - other_vectors, axis=1)
        parent_norms = np.linalg.norm(parent_vectors, axis=1)
        other_norms = np.linalg.norm(other_vectors, axis=1)
        distances = np.arccosh(
            1 + 2 * ((euclidean_dists ** 2) / ((1 - parent_norms ** 2) * (1 - other_norms ** 2))))
        sign = 1
        if rel_reversed:
            sign = -1
        return (1 + alpha * sign * (parent_norms - other_norms)) * distances
