import os
import numpy as np
from collections import defaultdict


# Num all nodes = 82114
# Num edges norm before transitive closure = 660990
# Num edges norm after transitive closure = 661127
# Done with the full list of edges. num = 661127  578477

def create_all_data(full_edges_data_file, root_str):
    # Compute and output vocabulary.
    all_nodes = set()
    with open(full_edges_data_file, 'r') as f:
        for i, line in enumerate(f):
            all_nodes.update(line.split())
    all_nodes.remove(root_str) # Exclude root

    idx2node = list(all_nodes)
    node2idx = {}
    num_nodes = len(idx2node)
    print('Num all nodes = ' + str(num_nodes))

    # Output vocabulary in id \tab word format.
    vocab_file = open(full_edges_data_file + '.vocab', 'w')
    for idx, node in enumerate(idx2node):
        node2idx[node] = idx
        vocab_file.write(str(idx) + '\t' + str(node) + '\n')
    vocab_file.close()

    # Compute full transitive closure of this DAG using Warshall
    outgoing_edges = defaultdict(set)
    ingoing_edges = defaultdict(set)
    num_edges = 0
    with open(full_edges_data_file, 'r') as f:
        for i, line in enumerate(f):
            child, parent = line.split() ## Reverse order in the file.
            if child == parent or parent == root_str:  # Exclude root edges.
                continue
            child_idx = node2idx[child]
            parent_idx = node2idx[parent]

            outgoing_edges[parent_idx].add(child_idx)
            ingoing_edges[child_idx].add(parent_idx)
            num_edges += 1

    print('Num edges norm before transitive closure = ' + str(num_edges))

    # Optimized Warshall’s Algorithm for computing the transitive closure for a DAG.
    for k in range(num_nodes):
        for i in ingoing_edges[k]: # i -> k -> j
            assert k in outgoing_edges[i]
            if i in outgoing_edges[i]:
                print('Graph has cycles !')
                os._exit(1)

            for j in outgoing_edges[k]:
                assert k in ingoing_edges[j]
                if not j in outgoing_edges[i]:
                    outgoing_edges[i].add(j)
                    ingoing_edges[j].add(i)
                    num_edges += 1

    print('Num edges norm after transitive closure = ' + str(num_edges))


    # Output full transitive closure  file in id \tab id format.
    full_transitive_file = open(full_edges_data_file + '.full_transitive', 'w')
    for parent_idx in outgoing_edges.keys():
        for child_idx in outgoing_edges[parent_idx]:
            full_transitive_file.write(str(parent_idx) + '\t' + str(child_idx) + '\n')
    full_transitive_file.close()


    # Compute transitive reduction of the graph, i.e. basic edges. See https://en.wikipedia.org/wiki/Transitive_reduction
    basic_outgoing_edges = defaultdict(set)
    for i in range(num_nodes):
        for j in outgoing_edges[i]:
            basic_outgoing_edges[i].add(j)

    basic_ingoing_edges = defaultdict(set)
    for i in range(num_nodes):
        for j in ingoing_edges[i]:
            basic_ingoing_edges[i].add(j)

    for k in range(num_nodes):
        for i in ingoing_edges[k]: # i -> k -> j
            assert k in outgoing_edges[i]
            for j in outgoing_edges[k]:
                assert k in ingoing_edges[j]
                if j in basic_outgoing_edges[i]:
                    basic_outgoing_edges[i].remove(j)
                if i in basic_ingoing_edges[j]:
                    basic_ingoing_edges[j].remove(i)

    # Output basic edges.
    # num_basic_edges = 0
    # basic_edges_file = open(full_edges_data_file + '.basic_edges', 'w')
    # for parent_idx in basic_outgoing_edges.keys():
    #     for child_idx in basic_outgoing_edges[parent_idx]:
    #         basic_edges_file.write(str(parent_idx) + '\t' + str(child_idx) + '\n')
    #         num_basic_edges += 1
    # basic_edges_file.close()
    # print('Num basic edges = ' + str(num_basic_edges))


    all_edges_basic = []
    all_edges_non_basic = []
    all_edges = []
    for i in range(num_nodes):
        for j in outgoing_edges[i]:
            all_edges.append((i,j))
            if j not in basic_outgoing_edges[i]:
                all_edges_non_basic.append((i,j))
            else:
                all_edges_basic.append((i,j))

    print('Done with the full list of edges. num = ' + str(len(all_edges)) + '  ' + str(len(all_edges_non_basic)))

    # Output 10 times more negative transitive closure edges - for reconstruction experiment.
    def gen_negs(node_idx, excluded, num_neg_edges_per_node_per_dir = 5):
        negatives = set()
        while len(negatives) < num_neg_edges_per_node_per_dir:
            new_negs = set(np.random.choice(num_nodes, 2 * num_neg_edges_per_node_per_dir))
            new_negs = new_negs - excluded
            new_negs = new_negs - {node_idx}
            negatives.update(new_negs)
        return list(negatives)[:num_neg_edges_per_node_per_dir]

    def gen_and_write_negs(pos_edges, file):
        for (parent_idx, child_idx) in pos_edges:
            # Generate pairs (u, v')
            if len(outgoing_edges[parent_idx]) < num_nodes - 1: # non root
                negatives = gen_negs(parent_idx, excluded = outgoing_edges[parent_idx])
                for neg_idx in negatives:
                    assert neg_idx not in outgoing_edges[parent_idx]
                    file.write(str(parent_idx) + '\t' + str(neg_idx) + '\n')

            # Generate pairs (u', v)
            negatives = gen_negs(child_idx, excluded = ingoing_edges[child_idx])
            for neg_idx in negatives:
                assert neg_idx not in ingoing_edges[child_idx]
                file.write(str(neg_idx) + '\t' + str(child_idx) + '\n')


    full_neg_edges_file = open(full_edges_data_file + '.full_neg', 'w')
    gen_and_write_negs(all_edges, full_neg_edges_file)
    full_neg_edges_file.close()
    print('Done gen neg edges for the full set.')


    ########## Split transitive-closure edges into train - valid - test.
    #  Train always contains all non-transitive-closure edges.
    train_perc_list = [0, 10, 25, 50, 90]
    valid_perc = 5
    test_perc = 5

    num_non_basic_edges = len(all_edges_non_basic)
    non_basic_indices_set = set(range(num_non_basic_edges))

    # Generate test set:
    test_edges_indices = np.random.choice(list(non_basic_indices_set), int(test_perc * num_non_basic_edges/ 100.0), replace=False)
    non_basic_indices_set = non_basic_indices_set - set(test_edges_indices)
    test_edges_file = open(full_edges_data_file + '.test', 'w')
    test_edges = []
    for i in test_edges_indices:
        test_edges.append(all_edges_non_basic[i])
        parent_idx = all_edges_non_basic[i][0]
        child_idx = all_edges_non_basic[i][1]
        test_edges_file.write(str(parent_idx) + '\t' + str(child_idx) + '\n')
    test_edges_file.close()

    # Generate negative edges for test:
    test_neg_edges_file = open(full_edges_data_file + '.test_neg', 'w')
    gen_and_write_negs(test_edges, test_neg_edges_file )
    test_neg_edges_file.close()
    print('Done gen neg edges for the TEST set.')


    # Generate valid set:
    valid_edges_indices = np.random.choice(list(non_basic_indices_set), int(valid_perc * num_non_basic_edges/ 100.0), replace=False)
    non_basic_indices_set = non_basic_indices_set - set(valid_edges_indices)
    valid_edges_file = open(full_edges_data_file + '.valid', 'w')
    valid_edges = []
    for i in valid_edges_indices:
        valid_edges.append(all_edges_non_basic[i])
        parent_idx = all_edges_non_basic[i][0]
        child_idx = all_edges_non_basic[i][1]
        valid_edges_file.write(str(parent_idx) + '\t' + str(child_idx) + '\n')
    valid_edges_file.close()

    # Generate negative edges for valid:
    valid_neg_edges_file = open(full_edges_data_file + '.valid_neg', 'w')
    gen_and_write_negs(valid_edges, valid_neg_edges_file )
    valid_neg_edges_file.close()
    print('Done gen neg edges for the VALID set.')


    for train_perc in train_perc_list:
        # Sample non-basic edges.
        train_edges_indices = np.random.choice(list(non_basic_indices_set), int(train_perc * num_non_basic_edges/ 100.0), replace=False)
        train_edges_file = open(full_edges_data_file + '.train_' + str(train_perc) + 'percent', 'w')
        for i in train_edges_indices:
            train_edges_file.write(str(all_edges_non_basic[i][0]) + '\t' + str(all_edges_non_basic[i][1]) + '\n')

        # Add all the basic edges.
        for i in range(len(all_edges_basic)):
            train_edges_file.write(str(all_edges_basic[i][0]) + '\t' + str(all_edges_basic[i][1]) + '\n')
        train_edges_file.close()
        print('Done gen neg edges for the TRAIN set with ' + str(train_perc) + ' percent.')


current_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)))
data_directory = os.path.join(current_directory, 'data', 'maxn')

# full_data_filepath = os.path.join(data_directory, 'mammal_closure.tsv')
# create_all_data(full_data_filepath, 'mammal.n.01')

full_data_filepath = os.path.join(data_directory, 'noun_closure.tsv')
create_all_data(full_data_filepath, 'entity.n.01')