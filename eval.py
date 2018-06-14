import numpy as np
from relations import Relations

def pretty_print_eval_map(result_map):
    s,_ = pretty_print_eval_map_rec(result_map, '')
    return s

def pretty_print_eval_map_rec(result_map, spaces):
    assert type(result_map) is dict
    if type(list(result_map.values())[0]) is str or type(list(result_map.values())[0]) is int:
        return '; '.join(str(k) + ': ' + str(v) for k,v in result_map.items())  + '\n', True
    s = ''
    for k,v in result_map.items():
        pretty_v , is_leaf = pretty_print_eval_map_rec(v, spaces + '  ')
        if is_leaf:
            s += spaces + str(k) + ': ' + pretty_v
        else:
            s += spaces + str(k) + ': \n' + pretty_v
    return s, False


def eval_classification(
        logger, task, valid_pos_path, valid_neg_path, test_pos_path, test_neg_path, vocab, score_fn, alphas_to_validate):
    """
    Evaluates edge classification based on a scoring function.
    :param score_fn(alpha, parent_index, other_indices=None, rel_reversed):
        Function that scores each edge (u,v), v in other_nodes. The lower score means the higher the chance
        the edge exists. One example of such function is the score(is-a(u,v)) from M. Nickel's paper.
    """
    if valid_pos_path == test_pos_path:
        assert valid_neg_path == test_neg_path
        assert task == 'reconstruction'
    if task != 'reconstruction':
        assert valid_neg_path != test_neg_path

    results = {task: {}}
    res = results[task]

    valid_eval_obj = _EvalObj(logger, valid_pos_path, valid_neg_path, vocab)
    test_eval_obj = _EvalObj(logger, test_pos_path, test_neg_path, vocab)

    ###### Perform validation
    best_alpha = None
    best_valid_f1 = -1
    best_optimal_th = None
    for alpha in alphas_to_validate:
        logger.info('now validating alpha = ' + str(alpha))
        optimal_th, optimal_valid_f1 = \
            valid_eval_obj.find_best_classification_thresh_F1(score_fn=score_fn, alpha=alpha) ##### Expensive

        if optimal_valid_f1 > best_valid_f1:
            best_valid_f1 = optimal_valid_f1
            best_alpha = alpha
            best_optimal_th = optimal_th

    ###### Done validation
    best_key = 'alpha=' + str(best_alpha)
    res[best_key] = {}
    res[best_key]['VALID'] =\
        valid_eval_obj.evaluate_classification(score_fn=score_fn,
                                               alpha=best_alpha,
                                               threshold=best_optimal_th)
    res[best_key]['TEST'] = \
        test_eval_obj.evaluate_classification(score_fn=score_fn,
                                              alpha=best_alpha,
                                              threshold=best_optimal_th)

    best_test_f1 = res[best_key]['TEST']['f1']
    best_valid_f1 = res[best_key]['VALID']['f1']
    return results, best_alpha, float(best_optimal_th), float(best_test_f1), float(best_valid_f1)


class _EvalObj(object):
    """Evaluating reconstruction on given network for given embedding."""

    def __init__(self, logger, positive_rel_filepath, negative_rel_filepath, vocab):
        self.logger = logger

        self.pos_relations_parents = []
        self.pos_relations_children = []
        rels = Relations(positive_rel_filepath)
        for node_parent, node_child in rels:
            assert node_parent != node_child
            node_parent_idx = vocab[node_parent].index
            node_child_idx = vocab[node_child].index
            self.pos_relations_parents.append(node_parent_idx)
            self.pos_relations_children.append(node_child_idx)

        self.neg_relations_parents = []
        self.neg_relations_children = []
        rels = Relations(negative_rel_filepath)
        for node_parent, node_child in rels:
            assert node_parent != node_child
            node_parent_idx = vocab[node_parent].index
            node_child_idx = vocab[node_child].index
            self.neg_relations_parents.append(node_parent_idx)
            self.neg_relations_children.append(node_child_idx)

        logger.info('eval datasets file pos = ' + positive_rel_filepath + '  neg = ' + negative_rel_filepath +
                    '; eval num rels pos = ' + str(len(self.pos_relations_parents)) + '  neg = ' + str(len(self.neg_relations_parents)))


    def evaluate_classification(self, score_fn, alpha, threshold):
        """Evaluates P, R, F1 and Acc for link prediction.

        Parameters
        -------
        score_fn(alpha, parent_index, other_indices=None, rel_reversed) :
            Scores each edge (u,v), v in other_nodes. The lower score means the higher the chance
            the edge exists. The higher, the lower.
        threshold :
            Threshold for the scores. What is below is classified as an edge. What is above is not an edge.
        """
        pos_scores = score_fn(alpha, self.pos_relations_parents, self.pos_relations_children, False)
        tp = (pos_scores <= threshold).sum()
        fn = (pos_scores > threshold).sum()

        neg_scores = score_fn(alpha, self.neg_relations_parents, self.neg_relations_children, False)
        fp = (neg_scores <= threshold).sum()

        precision = 100 * tp / (tp + fp + 1e-6)
        recall = 100 * tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return {'precision': ('%.1f' % precision), 'recall': ('%.1f' % recall), 'f1': ('%.1f' % f1)}


    def find_best_classification_thresh_F1(self, score_fn, alpha):
        """Like in the Order Embeddings paper, we find the best classification threshold

        Parameters
        -------
        score_fn(alpha, parent_index, other_indices=None, rel_reversed) :
            Scores each edge (u,v), v in other_nodes. The lower score means the higher the chance
            the edge exists. The higher, the lower.

        """
        # Vector of type (label, score) for each edge or non-edge in our dataset.
        all_labels_and_scores = []
        num_grd_trth_pos = len(self.pos_relations_parents)

        pos_scores = score_fn(alpha, self.pos_relations_parents, self.pos_relations_children, False)
        neg_scores = score_fn(alpha, self.neg_relations_parents, self.neg_relations_children, False)

        all_labels_and_scores.extend(zip(pos_scores, np.ones(len(pos_scores))))
        all_labels_and_scores.extend(zip(neg_scores, np.zeros(len(neg_scores))))

        # Sort scores. In case of equal scores, put the negatives (0-labels) first.
        all_labels_and_scores = sorted(all_labels_and_scores) #### Expensive, O(n * log n)

        tp = 0.0
        fp = 0.0
        best_th = 0.0
        best_f1 = -1

        for score,label in all_labels_and_scores:
            tp += label
            fp += (1.0 - label)
            precision = 100 * tp / (tp + fp + 1e-6)
            recall = 100 * tp / (num_grd_trth_pos)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            if f1 > best_f1:
                best_f1 = f1
                best_th = score
        return best_th, best_f1