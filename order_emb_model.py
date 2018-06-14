#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Python implementation of Order Embeddings [1]"""

from dag_emb_model import *

try:
    from autograd import grad  # Only required for optionally verifying gradients while training
    from autograd import numpy as grad_np
    AUTOGRAD_PRESENT = True
except ImportError:
    AUTOGRAD_PRESENT = False


class OrderModel(DAGEmbeddingModel):
    """Class for training, using and evaluating Order Embeddings."""
    def __init__(self,
                 train_data,
                 dim=50,
                 init_range=(-0.1, 0.1),
                 lr=0.01,
                 seed=0,
                 logger=None,

                 num_negative=1,
                 ### How to sample negatives for an edge (u,v)
                 neg_sampl_strategy='true_neg',  # 'all' (all nodes for negative sampling) or 'true_neg' (only nodes not connected)
                 where_not_to_sample='children',  # both or ancestors or children. Has no effect if neg_sampl_strategy = 'all'.
                 neg_edges_attach='both',  # How to form negative edges: 'parent' (u,v') or 'child' (u', v) or 'both'
                 neg_sampling_power=0.0,  # 0 for uniform, 1 for unigram, 0.75 for word2vec

                 margin=1.0,  # Margin for the OE loss.
                 ):
        super().__init__(train_data=train_data,
                         dim=dim,
                         init_range=init_range,
                         lr=lr,
                         opt='sgd',
                         burn_in=0,
                         seed=seed,
                         logger=logger,
                         BatchClass=OrderBatch,
                         KeyedVectorsClass=OrderKeyedVectors,
                         num_negative=num_negative,
                         neg_sampl_strategy=neg_sampl_strategy,
                         where_not_to_sample=where_not_to_sample,
                         always_v_in_neg=False,
                         neg_sampling_power=neg_sampling_power,
                         neg_edges_attach=neg_edges_attach)
        self.margin = margin


    def _clip_vectors(self, vectors):
        return vectors

    ### For autograd
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
        vector_u = matrix[0]
        vectors_v = matrix[1:]
        if not rels_reversed:
            entailment_penalty = grad_np.maximum(0, vector_u - vectors_v) # (1 + negative_size, dim).
        else:
            entailment_penalty = grad_np.maximum(0, - vector_u + vectors_v) # (1 + negative_size, dim).

        energy_vec = grad_np.linalg.norm(entailment_penalty, axis=1) ** 2
        positive_term = energy_vec[0]
        negative_terms = energy_vec[1:]
        return positive_term + grad_np.maximum(0, self.margin - negative_terms).sum()


class OrderBatch(DAGEmbeddingBatch):
    """Compute gradients and loss for a training batch."""
    def __init__(self,
                 vectors_u, # (1, dim, batch_size)
                 vectors_v, # (1 + neg_size, dim, batch_size)
                 indices_u,
                 indices_v,
                 rels_reversed,
                 order_model):
        super().__init__(
            vectors_u=vectors_u,
            vectors_v=vectors_v,
            indices_u=indices_u,
            indices_v=indices_v,
            rels_reversed=rels_reversed,
            dag_embedding_model=None)
        self.margin = order_model.margin

    def _compute_loss(self):
        """Compute and store loss value for the given batch of examples."""
        if self._loss_computed:
            return
        self._loss_computed = True

        if not self.rels_reversed:
            self.entailment_penalty = np.maximum(0, self.vectors_u - self.vectors_v) # (1 + negative_size, dim, batch_size).
        else:
            self.entailment_penalty = np.maximum(0, - self.vectors_u + self.vectors_v) # (1 + negative_size, dim, batch_size).

        self.energy_vec = np.linalg.norm(self.entailment_penalty, axis=1)**2 # (1 + negative_size, batch_size).
        self.pos_loss = self.energy_vec[0].sum()
        self.neg_loss = np.maximum(0, self.margin - self.energy_vec[1:]).sum()
        self.loss = self.pos_loss + self.neg_loss


    def _compute_loss_gradients(self):
        """Compute and store gradients of loss function for all input vectors."""
        if self._loss_gradients_computed:
            return
        self._compute_loss()

        if not self.rels_reversed:
            entailment_update_cond = 1.0 * (self.vectors_u - self.vectors_v > 0) # (1 + neg_size, dim, batch_size)
        else:
            entailment_update_cond = -1.0 * (- self.vectors_u + self.vectors_v > 0) # (1 + neg_size, dim, batch_size)

        energy_vec_grad_u = 2 * self.entailment_penalty * entailment_update_cond # (1 + neg_size, dim, batch_size)
        energy_vec_grad_v = - 2 * self.entailment_penalty * entailment_update_cond # (1 + neg_size, dim, batch_size)

        # neg_loss gradients
        neg_update_cond = (self.margin - self.energy_vec > 0)[:, np.newaxis, :] # (1 + neg_size, dim, batch_size)
        gradients_v = (-1.0 * neg_update_cond) * energy_vec_grad_v # (1 + neg_size, dim, batch_size)
        gradients_u = ((-1.0 * neg_update_cond) * energy_vec_grad_u)[1:].sum(axis=0) # (dim, batch_size)

        # pos loss gradients
        gradients_v[0] = energy_vec_grad_v[0]
        gradients_u += energy_vec_grad_u[0]

        self.loss_gradients_u = gradients_u # (dim, batch_size)
        self.loss_gradients_v = gradients_v # (1 + neg_size, dim, batch_size)

        assert not np.isnan(self.loss_gradients_u).any()
        assert not np.isnan(self.loss_gradients_v).any()
        self._loss_gradients_computed = True


class OrderKeyedVectors(DAGEmbeddingKeyedVectors):
    """Class to contain vectors and vocab for the :class:`~OrderModel` training class.
    Used to perform operations on the vectors such as vector lookup, distance etc.
    Inspired from KeyedVectorsBase.
    """
    def __init__(self):
        super(OrderKeyedVectors, self).__init__()


    def is_a_scores_vector_batch(self, alpha, parent_vectors, other_vectors, rel_reversed):
        if not rel_reversed:
            return np.linalg.norm(np.maximum(0, parent_vectors - other_vectors), axis=1)
        else:
            return np.linalg.norm(np.maximum(0, - parent_vectors + other_vectors), axis=1)
