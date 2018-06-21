from joblib import Parallel, delayed
import multiprocessing

from params import *
from eucl_simple_model import *
from order_emb_model import *
from poincare_model import *
from eucl_cones_model import *
from hyp_cones_model import *

from eval import *
from relations import *
from utils import *

############## Data directories ############
current_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)))
data_directory = os.path.join(current_directory, 'data', 'maxn') # Data downloaded from https://github.com/facebookresearch/poincare-embeddings
models_directory = os.path.join(current_directory, 'saved_models')

# p : list of pairs
def pretty_params(p, per_line=100):
    param_str_list = [('%s:%s' % (str(key), str(value))) for (key,value) in p]
    return '\n'.join(', '.join(param_str_list[per_line * i: per_line * (i + 1)]) for i in
                                  range(1 + int(len(param_str_list) / per_line)))


###############################################################################

def EVAL_ONE_MODEL(logger, full_data_filepath, task, model, ranking_alpha, validate_alphas):
    if task == 'reconstruction':
        valid_pos_path=full_data_filepath + '.full_transitive'
        valid_neg_path=full_data_filepath + '.full_neg'
        test_pos_path=full_data_filepath + '.full_transitive'
        test_neg_path=full_data_filepath + '.full_neg'
    else:
        valid_pos_path=full_data_filepath + '.valid'
        valid_neg_path=full_data_filepath + '.valid_neg'
        test_pos_path=full_data_filepath + '.test'
        test_neg_path=full_data_filepath + '.test_neg'

    ######## Classification #########
    if not validate_alphas:
        alphas_to_validate = [ranking_alpha]
    else:
        alphas_to_validate = [1000, 100, 30, 10, 3, 1, 0.3, 0.1, 0]

    # Validation
    eval_result_classif, best_alpha, _, best_test_f1, best_valid_f1 = eval_classification(
        logger=logger,
        task=task,
        valid_pos_path=valid_pos_path,
        valid_neg_path=valid_neg_path,
        test_pos_path=test_pos_path,
        test_neg_path=test_neg_path,
        vocab=model.kv.vocab,
        score_fn=model.kv.is_a_scores_from_indices,
        alphas_to_validate=alphas_to_validate, # 0 means only distance
    )
    logger.info('BEST classification ALPHA = %.3f' % best_alpha)
    logger.info(pretty_print_eval_map(eval_result_classif))
    return float(best_test_f1), float(best_valid_f1), pretty_print_eval_map(eval_result_classif)


def train_eval_one_model(logger_name, model_name, new_params, output_file):
    """Train a poincare embedding

    Args:
        model_name (str): Model name
        params (dict): parameters to train the model with
        output_file (str): Path to output file containing model
    Notes:
        If `output_file` already exists, skips training
    """
    logger = setup_logger(logger_name, also_stdout=False)

    if default_params['save'] and os.path.exists(output_file):
        logger.warning('File %s exists, skipping' % output_file)
        return

    params = default_params.copy()
    for key,value in new_params:
        params[key] = value

    full_data_filepath = os.path.join(data_directory, params['wn'] + '_closure.tsv')

    if params['task'] == 'reconstruction':
        train_path = full_data_filepath + '.full_transitive'
    else:
        train_path = full_data_filepath + '.train_' + params['task']

    logger.info('Train file : ' + train_path)

    logger.info('TASK: ' + params['task'])
    logger.info('\nTraining model: ' + model_name)

    train_data = Relations(train_path, reverse=False)

######################################################  INIT #####################################################################
    logger.info('================== START INIT ====================')

    if params['class'] == 'OrderEmb':
        params['epochs_init'] = 0
        params['epochs_init_burn_in'] = 0



    assert params['init_class'] in ['PoincareNIPS', 'EuclNIPS']

    if params['init_class'] == 'PoincareNIPS':
        model = PoincareModel(train_data=train_data,
                              dim=params['dim'],
                              logger=logger,
                              init_range=(-0.0001, 0.0001),
                              lr=params['lr_init'],
                              opt=params['opt'],  # rsgd or exp_map
                              burn_in=params['epochs_init_burn_in'],
                              epsilon=params['epsilon'],
                              seed=params['seed'],
                              num_negative=params['num_negative'],
                              neg_sampl_strategy=params['neg_sampl_strategy_init'],
                              where_not_to_sample=params['where_not_to_sample'],
                              neg_edges_attach=params['neg_edges_attach'],
                              always_v_in_neg=True,
                              neg_sampling_power=params['neg_sampling_power_init'],
                              loss_type='nll',
                              )
    else:
        model = EuclSimpleModel(train_data=train_data,
                                dim=params['dim'],
                                logger=logger,
                                init_range=(-0.0001, 0.0001),
                                lr=params['lr_init'],
                                burn_in=params['epochs_init_burn_in'],
                                seed=params['seed'],
                                num_negative=params['num_negative'],
                                neg_sampl_strategy=params['neg_sampl_strategy_init'],
                                where_not_to_sample=params['where_not_to_sample'],
                                neg_edges_attach=params['neg_edges_attach'],
                                always_v_in_neg=True,
                                neg_sampling_power=params['neg_sampling_power_init'],
                                )


    best_test_f1_init = -1
    best_valid_f1_init = -1
    best_epoch_init = -1
    best_eval_long_res_init = 'NO INIT'
    best_str_init = 'NO INIT'

    for i in range(int(params['epochs_init'] / params['print_every'])):
        model.train(epochs=params['print_every'],
                    batch_size=params['batch_size'],
                    print_every=params['print_every'])
        num_epochs_done = params['print_every'] * (i + 1)

        # Evaluate model
        logger.info(
            '########################### start INIT eval after %d epochs ############################################' % num_epochs_done)
        logger.info('MODEL = %s\n' % (model_name))

        test_f1, valid_f1, eval_results = EVAL_ONE_MODEL(logger=logger,
                                                         full_data_filepath=full_data_filepath,
                                                         task=params['task'],
                                                         model=model,
                                                         ranking_alpha=0,
                                                         validate_alphas=True,
                                                         )

        if valid_f1 > best_valid_f1_init:
            best_valid_f1_init = valid_f1
            best_test_f1_init = test_f1
            best_epoch_init = num_epochs_done
            best_eval_long_res_init = eval_results

        best_str_init = ' f1 INIT test = %.2f; INIT valid = %.2f - after %d epochs. ' % (best_test_f1_init, best_valid_f1_init, best_epoch_init)
        logger.info('\n\n ======> best so far ' + best_str_init)
        logger.info(
            '########################### end INIT eval ##############################################')


    logger.info('\n\n\n ======> best OVERALL ' + best_str_init)
    logger.info('========================== DONE INIT ================================\n\n\n')

######################################################  CONES #####################################################################
    if params['class'] == 'EuclCones':
        cls = EuclConesModel
        opt = 'sgd'
    elif params['class'] == 'HypCones':
        cls = HypConesModel
        opt = params['opt']
    elif params['class'] == 'OrderEmb':
        cls = OrderModel


    if cls == OrderModel:
        model = OrderModel(train_data=train_data,
                           dim=params['dim'],
                           init_range=(-0.1, 0.1),
                           lr=params['lr'],
                           seed=params['seed'],
                           logger=logger,
                           num_negative=params['num_negative'],
                           neg_sampl_strategy=params['neg_sampl_strategy'],
                           where_not_to_sample=params['where_not_to_sample'],
                           neg_edges_attach=params['neg_edges_attach'],
                           neg_sampling_power=0,
                           margin=params['margin'],
                           )
        model.K = -1e12 #### no alpha needed

    else:
        # Use init vecs:
        init_vecs = model.kv.syn0 * params['resc_vecs']
        model = cls(train_data,
                    dim=params['dim'],
                    init_range=(-0.1, 0.1),
                    lr=params['lr'],
                    seed=params['seed'],
                    logger=logger,
                    opt=opt,
                    num_negative=params['num_negative'],
                    neg_sampl_strategy=params['neg_sampl_strategy'],
                    where_not_to_sample=params['where_not_to_sample'],
                    neg_edges_attach=params['neg_edges_attach'],
                    neg_sampling_power=0.0,

                    margin=params['margin'],
                    K=params['K'],
                    epsilon=params['epsilon'],
                    )

        model.kv.syn0 = model._clip_vectors(init_vecs)


    # Train the model
    best_valid_f1 = -1
    best_test_f1 = -1
    best_epoch = -1
    best_eval_long_res = ' ONLY INIT'
    best_str = ' ONLY INIT'

    for i in range(int(params['epochs'] / params['print_every'])):
        model.train(epochs=params['print_every'],
                    batch_size=params['batch_size'],
                    print_every=params['print_every'])

        num_epochs_done = params['epochs_init'] + params['print_every'] * (i + 1)

        # Evaluate model
        logger.info(
            '########################### start CONES eval after %d epochs ############################################' % num_epochs_done)
        logger.info('MODEL = %s\n' % (model_name))

        test_f1, valid_f1, eval_results = EVAL_ONE_MODEL(logger=logger,
                                                         full_data_filepath=full_data_filepath,
                                                         task=params['task'],
                                                         model=model,
                                                         ranking_alpha=model.K,
                                                         validate_alphas=False,
                                                         )
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_test_f1 = test_f1
            best_epoch = num_epochs_done
            best_eval_long_res = eval_results

        best_str = ' f1 CONES test = %.2f; CONES valid = %.2f - after %d epochs.' % (best_test_f1, best_valid_f1, best_epoch)
        logger.info('\n\n ====> best so far ' + best_str)
        logger.info(
            '########################### end CONES eval ##############################################')

    logger.info('\n\n ======> best OVERALL ' + best_str)

    # Save the model
    if params['save']:
        model.save(output_file)


    results_strings = ['best ' + best_str_init + ' ; best ' + best_str]
    results_strings.append(('\n >>>>>>>>> INIT = \n%s \n---------------------\n' +
                           ' >>>>>>>>> CONES = \n%s') % (best_eval_long_res_init, best_eval_long_res))

    return new_params, results_strings




######################### Train and eval all models in parallel ######################
model_files = {}
model_params_list = []
for task in [ '0percent', '10percent', '25percent', '50percent', '90percent']:
    for p in non_default_params:
        # new_params = p.copy()
        new_params = [('task', task)]
        new_params.extend(p.copy())

        model_name = pretty_params(new_params, per_line=len(new_params))
        logger_name = ';'.join(['%s:%s' % (key, value) for (key, value) in new_params])
        model_files[model_name] = os.path.join(models_directory, logger_name[:200])

        model_params_list.append((logger_name, model_name, new_params, model_files[model_name]))

# Train & eval in parallel
results = Parallel(n_jobs=threads) \
    (delayed(train_eval_one_model)(logger_name, model_name, new_params, output_file)
     for (logger_name, model_name, new_params, output_file) in model_params_list)

######################################################################################
