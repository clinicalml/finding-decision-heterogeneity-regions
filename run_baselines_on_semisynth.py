import os
import numpy as np
import sys
import pandas as pd
import torch
import data_processing.semisynth_dataloader as ssdl
import data_processing.semisynth_subsets as sub
from baselines import run_U_learner, run_causal_forest_efficient, run_model_direct, run_tarnet
from run_baseline_models import run_model

LOGIT = 1.5
n_iter = 10

baseline_type = sys.argv[1]
assert baseline_type in {'CausalForest', 'Direct', 'TARNet', 'Iterative'}
if baseline_type == 'TARNet':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    torch.cuda.device(0)
    assert(torch.cuda.is_available())
    torch.manual_seed(42)
num_agents = int(sys.argv[2])
assert num_agents > 1
model_classes = ['LogisticRegression', 'DecisionTree', 'RandomForest']
subset = sys.argv[3]
assert subset in {'drug_possession', 'misdemeanor_under35'}
if subset == 'drug_possession':
    SUB = sub.drug_possession
else:
    SUB = sub.misdemeanor_under35

output_dir = 'semisynth_logit15_' + subset + '_numagents' + str(num_agents) + '_baselines/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
metrics = dict()
if baseline_type == 'CausalForest':
    ulearner_metrics = dict()
for model_class in model_classes:
    metrics[model_class] = {'region_precisions': [], 'region_recalls': [], 'region_aucs': [], 'partition_accs': []}
    if baseline_type == 'CausalForest':
        ulearner_metrics[model_class] = {'region_precisions': [], 'region_recalls': [], 'region_aucs': [], 'partition_accs': []}

for i in range(n_iter):
    dd = ssdl.load_semi_synthetic_compas_data(
            seed=i, logit_adjust=LOGIT,
            num_agents=num_agents, subset_func=SUB,
            verbose=False, cache=False)
    filename = baseline_type + '_iter' + str(i) + '_results.pkl'

    if baseline_type == 'CausalForest':
        ulearner_filename = 'ULearner_iter' + str(i) + '_results.pkl'
        ulearner_results_dict = run_U_learner(
                dd['X'], dd['d'], dd['t'],
                dd['train_idxs'], dd['valid_idxs'], dd['test_idxs'],
                output_dir + ulearner_filename,
                dd['true_region_func'], dd['true_provider_split'],
                beta=dd['true_beta'])

        print('ULearner iter ' + str(i))
        for model_class in model_classes:
            print(model_class)
            print('Region precision: ' + str(ulearner_results_dict[model_class]['region_precision']))
            print('Region recall: ' + str(ulearner_results_dict[model_class]['region_recall']))
            print('Region AUC: ' + str(ulearner_results_dict[model_class]['region_auc']))
            print('Partition accuracy: ' + str(ulearner_results_dict[model_class]['partition_acc']))

            ulearner_metrics[model_class]['region_precisions'].append(ulearner_results_dict[model_class]['region_precision'])
            ulearner_metrics[model_class]['region_recalls'].append(ulearner_results_dict[model_class]['region_recall'])
            ulearner_metrics[model_class]['region_aucs'].append(ulearner_results_dict[model_class]['region_auc'])
            ulearner_metrics[model_class]['partition_accs'].append(ulearner_results_dict[model_class]['partition_acc'])

        oracle_preds = ulearner_results_dict['all_resids_pred']
        results_dict = run_causal_forest(
                dd['X'], dd['d'], dd['t'],
                dd['train_idxs'], dd['valid_idxs'], dd['test_idxs'],
                oracle_preds,
                output_dir + filename,
                dd['true_region_func'], dd['true_provider_split'],
                beta=dd['true_beta'])
    elif baseline_type == 'Direct':
        results_dict = run_model_direct(
                dd['X'], dd['d'], dd['t'],
                dd['train_idxs'], dd['valid_idxs'], dd['test_idxs'],
                output_dir + filename,
                dd['true_region_func'], dd['true_provider_split'],
                beta=dd['true_beta'])
    elif baseline_type == 'TARNet':
        tarnet_dir = baseline_type + '_iter' + str(i) + '/'
        if not os.path.exists(output_dir + tarnet_dir):
            os.makedirs(output_dir + tarnet_dir)
        results_dict = run_tarnet(dd['X'], dd['d'], dd['t'], dd['train_idxs'], dd['valid_idxs'], dd['test_idxs'], 
                                  output_dir + filename, output_dir + tarnet_dir, dd['true_region_func'], 
                                  dd['true_provider_split'], beta=dd['true_beta'])
    else:
        results_dict = dict()
        for model_class in model_classes:
            results_dict[model_class] = run_model(baseline_type, model_class,
                    dd['X'], dd['d'], dd['t'],
                    dd['train_idxs'], dd['valid_idxs'], dd['test_idxs'],
                    output_dir + filename, 
                    dd['true_region_func'], dd['true_provider_split'],
                    beta=dd['true_beta'], n_iter=100,
                    outcome_model_class='LogisticRegression', verbose=False)

    print(baseline_type + ' iter' + str(i))
    for model_class in model_classes:
        print(model_class)
        print('Region precision: ' + str(results_dict[model_class]['region_precision']))
        print('Region recall: ' + str(results_dict[model_class]['region_recall']))
        print('Region AUC: ' + str(results_dict[model_class]['region_auc']))

        metrics[model_class]['region_precisions'].append(results_dict[model_class]['region_precision'])
        metrics[model_class]['region_recalls'].append(results_dict[model_class]['region_recall'])
        metrics[model_class]['region_aucs'].append(results_dict[model_class]['region_auc'])

        if results_dict[model_class]['partition_acc'] is not None:
            print('Partition accuracy: ' + str(results_dict[model_class]['partition_acc']))
            metrics[model_class]['partition_accs'].append(results_dict[model_class]['partition_acc'])

if baseline_type == 'CausalForest':
    print('ULearner mean (std) across ' + str(n_iter) + ' iterations')
    ulearner_df = None
    for model_class in model_classes:
        print(model_class)
        print('Region precision: {0:.4f}'.format(np.mean(ulearner_metrics[model_class]['region_precisions'])) \
              + ' ({0:.4f})'.format(np.std(ulearner_metrics[model_class]['region_precisions'])))
        print('Region recall: {0:.4f}'.format(np.mean(ulearner_metrics[model_class]['region_recalls'])) \
              + ' ({0:.4f})'.format(np.std(ulearner_metrics[model_class]['region_recalls'])))
        print('Region AUC: {0:.4f}'.format(np.mean(ulearner_metrics[model_class]['region_aucs'])) \
              + ' ({0:.4f})'.format(np.std(ulearner_metrics[model_class]['region_aucs'])))
        print('Partition accuracy: {0:.4f}'.format(np.mean(ulearner_metrics[model_class]['partition_accs'])) \
              + ' ({0:.4f})'.format(np.std(ulearner_metrics[model_class]['partition_accs'])))
        class_df = pd.DataFrame({'Model class': [model_class for i in range(n_iter)], 'iter': np.arange(n_iter), \
                                 'Region precision': ulearner_metrics[model_class]['region_precisions'], \
                                 'Region recall': ulearner_metrics[model_class]['region_recalls'], \
                                 'Region AUC': ulearner_metrics[model_class]['region_aucs'], \
                                 'Partition accuracy': ulearner_metrics[model_class]['partition_accs']})
        if ulearner_df is None:
            ulearner_df = class_df
        else:
            ulearner_df = pd.concat((ulearner_df, class_df), ignore_index=True)
    ulearner_df.to_csv(output_dir + 'ULearner_summary_metrics.csv', index=False)

print(baseline_type + ' mean (std) across ' + str(n_iter) + ' iterations')
baseline_df = None
for model_class in model_classes:
    print(model_class)
    print('Region precision: {0:.4f}'.format(np.mean(metrics[model_class]['region_precisions'])) \
          + ' ({0:.4f})'.format(np.std(metrics[model_class]['region_precisions'])))
    print('Region recall: {0:.4f}'.format(np.mean(metrics[model_class]['region_recalls'])) \
          + ' ({0:.4f})'.format(np.std(metrics[model_class]['region_recalls'])))
    print('Region AUC: {0:.4f}'.format(np.mean(metrics[model_class]['region_aucs'])) \
          + ' ({0:.4f})'.format(np.std(metrics[model_class]['region_aucs'])))
    class_df = pd.DataFrame({'Model class': [model_class for i in range(n_iter)], 'iter': np.arange(n_iter), \
                             'Region precision': metrics[model_class]['region_precisions'], \
                             'Region recall': metrics[model_class]['region_recalls'], \
                             'Region AUC': metrics[model_class]['region_aucs']})
    if len(metrics[model_class]['partition_accs']) > 0:
        print('Partition accuracy: {0:.4f}'.format(np.mean(metrics[model_class]['partition_accs'])) \
              + ' ({0:.4f})'.format(np.std(metrics[model_class]['partition_accs'])))
        class_df['Partition accuracy'] = metrics[model_class]['partition_accs']
    if baseline_df is None:
        baseline_df = class_df
    else:
        baseline_df = pd.concat((baseline_df, class_df), ignore_index=True)

baseline_df.to_csv(output_dir + baseline_type + '_summary_metrics.csv', index=False)