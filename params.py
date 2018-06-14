from collections import OrderedDict

threads = 24

default_params = OrderedDict([
    ############ Common params:
    ('wn', 'noun'), # mammal or noun
    ('print_every', 20),
    ('save', False), # Whether to save the model in the folder saved_models/
    ('num_negative', 10),  # Number of negative samples to use
    ('batch_size', 10),  # Size of batch to use for training
    ('epsilon', 1e-5),
    ('seed', 0),

    ('dim', 5),

    ('opt', 'rsgd'),  # rsgd or exp_map or sgd . Used for all hyperbolic models.  #### rsgd always better
    ('where_not_to_sample', 'ancestors'), # both or ancestors or children. Has no effect if neg_sampl_strategy = 'all'.
    ('neg_edges_attach', 'child'), # How to form negative edges: 'parent' (u,v') or 'child' (u', v) or 'both'

    ############## Angle loss:
    ('class', 'HypCones'), # 'EuclCones' , 'HypCones'  , 'OrderEmb'
    ('neg_sampl_strategy', 'true_neg_non_leaves'),  ########## true_neg_non_leaves worse than true_neg when init uses true_neg_non_leaves ?????
    ('lr', 0.0001),  ### 1e-4 the best for Hyp cones with rsgd ; 3e-4 better for Eucl cones


    ('resc_vecs', 0.7), ## 0.7 and 0.8 are similar
    ('epochs', 300),
    ('K', 0.1),
    ('margin', 0.01),

    ############### Init loss:
    ('init_class', 'PoincareNIPS'), # PoincareNIPS, EuclNIPS
    ('lr_init', 0.03), # 0.3, 0.03, 0.1 all kind of good; 0.03 the best 94%, but with 1/10 factor for burnin
    ('epochs_init', 100),
    ('neg_sampl_strategy_init', 'true_neg'), #  'true_neg' always better!

    ('epochs_init_burn_in', 20),
    ('neg_sampling_power_init', 0.75),  # 0 for uniform, 1 for unigram, 0.75 much better than 0 !!!!!! Do not put 0.
])


### We run 3 different jobs, but each of them will be ran in all.py:291 for all training settings (percentage of transitive closure).
non_default_params = [

### Our method : hyperbolic entailment cones
# File: task_50percent#dim_5#class_HypCones#init_class_PoincareNIPS#neg_sampl_strategy_true_neg#lr_0.0003#epochs_300#opt_rsgd#where_not_to_sample_children#neg_edges_attach_parent#lr_init_0.03#epochs_init_100#neg_sampl_strategy_init_true_neg
#  ======> best OVERALL  f1 CONES test = 92.80; CONES valid = 92.60 - after 260 epochs.
# To see the above result at the end of the training, one needs to run the following:
# for i in `ls ./logs/task_50percent#dim_5#*` ; do echo $i; cat $i |  grep best | grep CONES | grep OVERALL ; done | grep -A1 'HypCones' ;for i in `ls ./logs/task_50percent#epochs*` ; do echo $i; cat $i |  grep best | grep CONES | grep OVERALL  ; done
[('dim', 5), ('class', 'HypCones'), ('init_class', 'PoincareNIPS'), ('neg_sampl_strategy', 'true_neg'),  ('lr', 0.0003),  ('epochs', 300), ('opt', 'rsgd'),  ('where_not_to_sample', 'children'), ('neg_edges_attach', 'parent'), ('lr_init', 0.03), ('epochs_init', 100), ('neg_sampl_strategy_init', 'true_neg')],


### Poincare embeddings of Nickel et al., NIPS'18 - we look for the INIT results in this log file.
# File: task_50percent#dim_5#class_HypCones#init_class_PoincareNIPS#neg_sampl_strategy_true_neg_non_leaves#lr_0.0001#epochs_300#opt_exp_map#where_not_to_sample_ancestors#neg_edges_attach_child#lr_init_0.03#epochs_init_100#neg_sampl_strategy_init_true_neg
#  ======> best OVERALL  f1 INIT test = 83.60; INIT valid = 83.60 - after 80 epochs.
# To see the above result at the end of the training, one needs to run the following:
# for i in `ls ./logs/task_50percent#dim_5#*` ; do echo $i; cat $i |  grep best | grep INIT | grep OVERALL  ; done | grep -A1 'PoincareNIPS'; for i in `ls ./logs/task_50percent#epochs*` ; do echo $i; cat $i |  grep best | grep INIT | grep OVERALL  ; done
[('dim', 5), ('class', 'HypCones'), ('init_class', 'PoincareNIPS'), ('neg_sampl_strategy', 'true_neg_non_leaves'),  ('lr', 0.0001),  ('epochs', 300), ('opt', 'exp_map'),  ('where_not_to_sample', 'ancestors'), ('neg_edges_attach', 'child'), ('lr_init', 0.03), ('epochs_init', 100), ('neg_sampl_strategy_init', 'true_neg')],


### Order embeddings of Vendrov et al, ICLR'16
# File: task_50percent#dim_5#class_OrderEmb#neg_sampl_strategy_true_neg#lr_0.1#margin_1#epochs_500#where_not_to_sample_children#neg_edges_attach_parent
# ======> best OVERALL  f1 CONES test = 81.70; CONES valid = 81.60 - after 460 epochs.
# To see the above result at the end of the training, one needs to run the following:
# for i in `ls ./logs/task_50percent#dim_5#*` ; do echo $i; cat $i |  grep best | grep CONES | grep OVERALL ; done | grep -A1 'OrderEmb' ;for i in `ls ./logs/task_50percent#*` ; do echo $i; cat $i |  grep best | grep CONES | grep OVERALL ; done | grep -A1 'OrderEmb'
[('dim', 5), ('class', 'OrderEmb'), ('neg_sampl_strategy', 'true_neg'),  ('lr', 0.1), ('margin', 1), ('epochs', 500), ('where_not_to_sample', 'children'), ('neg_edges_attach', 'parent')],

]

### Remove duplicate commands
p = []
for i in range(len(non_default_params)):
    has_copy = False
    for j in range(i+1, len(non_default_params)):
        if non_default_params[i] == non_default_params[j]:
            has_copy = True
    if not has_copy:
        p.append(non_default_params[i])

non_default_params = p