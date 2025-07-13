from recbole.sampler import KGSampler
from recbole.data import load_split_dataloaders, create_samplers, get_dataloader, save_split_dataloaders
from recbole.utils import ModelType
from recbole.utils import get_trainer, init_seed, set_color
from logging import getLogger

import copy
import numpy as np
import torch
from tqdm import tqdm
from IPython import embed

def data_preparation_head_tail(config, dataset, pop_item = 'target', co_occurence = False, head_ratio = 0.2):
    """Split the dataset by :attr:`config['eval_args']` and create training, validation and test dataloader.

    Note:
        If we can load split dataloaders by :meth:`load_split_dataloaders`, we will not create new split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    print(f"It split head and item by {pop_item}")
    dataloaders = load_split_dataloaders(config)
    if dataloaders is not None:
        train_data, valid_data, test_data = dataloaders
    else:
        model_type = config['MODEL_TYPE']
        built_datasets = dataset.build()

        train_dataset, valid_dataset, test_dataset = built_datasets
        train_sampler, valid_sampler, test_sampler = create_samplers(config, dataset, built_datasets)

        if model_type != ModelType.KNOWLEDGE:
            train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
        else:
            kg_sampler = KGSampler(dataset, config['train_neg_sample_args']['distribution'])
            train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, kg_sampler, shuffle=True)

        valid_data = get_dataloader(config, 'evaluation')(config, valid_dataset, valid_sampler, shuffle=False)
        test_data = get_dataloader(config, 'evaluation')(config, test_dataset, test_sampler, shuffle=False)
        
        # copy test_dataset for head and tail
        head_test_dataset = copy.deepcopy(test_dataset)
        tail_test_dataset = copy.deepcopy(test_dataset)

        n_items = train_dataset.item_num
        # calculate co-occurrence matrix (consecutive) and popularity array
        consecutive_co_occurrence = np.zeros((n_items, n_items), dtype=np.int32)
        session_co_occurrence = np.zeros((n_items, n_items), dtype=np.int32)
        pop_array = np.zeros(n_items, dtype=np.int32)

        # load training data as numpy array
        session_id = train_data.dataset.inter_feat['session_id'].numpy() # [num_session, ]
        item_seq = train_data.dataset.inter_feat['item_id_list'].numpy() # [num_session, max_session_len]
        item_target = train_data.dataset.inter_feat['item_id'].numpy() # [num_session, ]
        session_len = train_data.dataset.inter_feat['item_length'].numpy() # [num_session, ]

        # extend max_length
        max_length = 1000
        item_seq = np.concatenate([item_seq, np.zeros((item_seq.shape[0], max_length-item_seq.shape[1]), dtype=np.int32)], axis=1)

        before_first_item = 0
        before_length = 0
        remain_items = []

        for i, (idx, seq, target, length) in enumerate(tqdm(zip(session_id, item_seq, item_target, session_len), total=len(session_id), desc='calc pop...' )):
            # check if this session is augmentation of before session
            # case 1: before_first_item == seq[0] and before_last_item == target 
            if before_first_item == seq[0] and before_last_item == target and before_length == length + 1:
                is_augmented = True
            # case 2: next_length == length and next_target == last_item 
            elif i != len(session_id)-1 and session_len[i+1] == length and item_target[i+1] == seq[length-1]:
                remain_items.append(target)
                is_augmented = True
            else:
                is_augmented = False

            before_first_item = seq[0]
            before_last_item = seq[length-1]
            before_length = length

            if is_augmented:
                # remove this session from train_data
                continue
            
            item_seq[i, length] = target
            new_length = length + len(remain_items) + 1
            if len(remain_items) > 0:
                item_seq[i, length+1:length+1+len(remain_items)] = remain_items
                remain_items = []

            session_items = item_seq[i, :new_length]
            session_items = list(set(session_items))

            for j in range(len(session_items)):
                if j != len(session_items)-1:
                    consecutive_co_occurrence[item_seq[i, j], item_seq[i, j+1]] += 1 # current item -> next item
                session_co_occurrence[session_items[j], session_items] += 1 # [all items in session]
            pop_array[session_items] += 1

        item_pop = pop_array
        print(f"percentage of pop_array: {np.percentile(item_pop, [0, 10, 20, 30, 40, 50, 60, 70 ,80, 90, 100])}")


        item_indices = np.argsort(item_pop)
        head_item = item_indices[-int(len(item_indices)*head_ratio):]
        tail_item = item_indices[:-int(len(item_indices)*head_ratio)]
        
        print(f"Number of head_item: {len(head_item)}, Number of tail_item: {len(tail_item)}")
        print(f"Number of Interactions: All {len(train_data.dataset)}, Head {sum(item_pop[head_item])}, Tail {sum(item_pop[tail_item])}")

        # co_occurence == False
        if pop_item == 'target':
            head_indices = [i for i, test_item in enumerate(test_dataset['item_id']) if test_item in head_item]
            tail_indices = [i for i, test_item in enumerate(test_dataset['item_id']) if test_item in tail_item]
        elif pop_item == 'last_item':
            head_indices = []
            tail_indices = []
            for i, (test_seq, seq_len) in enumerate(zip(test_dataset['item_id_list'], test_dataset['item_length'])):
                if test_seq[seq_len-1].item() in head_item:
                    head_indices.append(i)
                elif test_seq[seq_len-1].item() in tail_item:
                    tail_indices.append(i)
        
        head_test_inter_feat = test_dataset[head_indices]
        tail_test_inter_feat = test_dataset[tail_indices]

        head_test_dataset.inter_feat = head_test_inter_feat
        tail_test_dataset.inter_feat = tail_test_inter_feat

        head_test_data = get_dataloader(config, 'evaluation')(config, head_test_dataset, test_sampler, shuffle=False)
        tail_test_data = get_dataloader(config, 'evaluation')(config, tail_test_dataset, test_sampler, shuffle=False)

        if co_occurence and pop_item == 'last_item':
            print(f"use co-occurence between last_item and test_item")
            high_occur_indices = []
            mid_occur_indices = []
            low_occur_indices = []

            all_normalized_co_occurrence = []
            all_transition_prob = []

            for i, (test_seq, seq_len, test_item) in enumerate(zip(test_dataset['item_id_list'], test_dataset['item_length'], test_dataset['item_id'])):
                last_item = test_seq[seq_len-1]

                normalized_co_occurrence = session_co_occurrence[last_item, test_item] / item_pop[last_item]
                all_normalized_co_occurrence.append(normalized_co_occurrence)

                transition_prob = consecutive_co_occurrence[last_item, test_item] / (session_co_occurrence[last_item, test_item]) if session_co_occurrence[last_item, test_item] > 0 else 0
                all_transition_prob.append(transition_prob)

                # # by normalized_co_occurrence
                if normalized_co_occurrence > 0.05:
                    high_occur_indices.append(i)
                # elif normalized_co_occurrence > 0.001:
                #     mid_occur_indices.append(i)
                else:
                    low_occur_indices.append(i)

                # # by transition_prob
                # if transition_prob >= 0.2222:
                #     high_occur_indices.append(i)
                # # elif transition_prob >= 0.01:
                # #     mid_occur_indices.append(i)
                # else:
                #     low_occur_indices.append(i)

            print(f"number of each occurence: high {len(high_occur_indices)}, mid {len(mid_occur_indices)}, low {len(low_occur_indices)}")
            print(f"percentile of normalized co-occurrence: {np.percentile(all_normalized_co_occurrence, [0, 10, 20, 30, 40 ,50, 60, 70, 80, 90, 100])}")
            print(f"percentage of each occurence: high {len(high_occur_indices)/len(test_dataset)}, mid {len(mid_occur_indices)/len(test_dataset)}, low {len(low_occur_indices)/len(test_dataset)}")

            print(f"percentile of transition prob: {np.percentile(all_transition_prob, [10, 20, 30, 40 ,50, 60, 70, 80, 90])}")

            # sorted_numbers = sorted(all_normalized_co_occurrence)
            # result = []
            # percentiles=[33, 66]
            # for p in percentiles:
            #     index = int(len(sorted_numbers) * (p / 100))
            #     result.append(sorted_numbers[index])
            # print(result)

                
            head_high_occur_indices = [i for i in high_occur_indices if i in head_indices]
            tail_high_occur_indices = [i for i in high_occur_indices if i in tail_indices]
            head_mid_occur_indices = [i for i in mid_occur_indices if i in head_indices]
            tail_mid_occur_indices = [i for i in mid_occur_indices if i in tail_indices]
            head_low_occur_indices = [i for i in low_occur_indices if i in head_indices]
            tail_low_occur_indices = [i for i in low_occur_indices if i in tail_indices]

            head_high_occur_inter_feat = test_dataset[head_high_occur_indices]
            tail_high_occur_inter_feat = test_dataset[tail_high_occur_indices]
            head_mid_occur_inter_feat = test_dataset[head_mid_occur_indices]
            tail_mid_occur_inter_feat = test_dataset[tail_mid_occur_indices]
            head_low_occur_inter_feat = test_dataset[head_low_occur_indices]
            tail_low_occur_inter_feat = test_dataset[tail_low_occur_indices]

            # copy test_dataset for head and tail
            head_high_occur_dataset = copy.deepcopy(test_dataset)
            tail_high_occur_dataset = copy.deepcopy(test_dataset)
            head_mid_occur_dataset = copy.deepcopy(test_dataset)
            tail_mid_occur_dataset = copy.deepcopy(test_dataset)
            head_low_occur_dataset = copy.deepcopy(test_dataset)
            tail_low_occur_dataset = copy.deepcopy(test_dataset)

            head_high_occur_dataset.inter_feat = head_high_occur_inter_feat
            tail_high_occur_dataset.inter_feat = tail_high_occur_inter_feat
            head_mid_occur_dataset.inter_feat = head_mid_occur_inter_feat
            tail_mid_occur_dataset.inter_feat = tail_mid_occur_inter_feat
            head_low_occur_dataset.inter_feat = head_low_occur_inter_feat
            tail_low_occur_dataset.inter_feat = tail_low_occur_inter_feat

            head_high_occur_data = get_dataloader(config, 'evaluation')(config, head_high_occur_dataset, test_sampler, shuffle=False)
            tail_high_occur_data = get_dataloader(config, 'evaluation')(config, tail_high_occur_dataset, test_sampler, shuffle=False)
            head_mid_occur_data = get_dataloader(config, 'evaluation')(config, head_mid_occur_dataset, test_sampler, shuffle=False)
            tail_mid_occur_data = get_dataloader(config, 'evaluation')(config, tail_mid_occur_dataset, test_sampler, shuffle=False)
            head_low_occur_data = get_dataloader(config, 'evaluation')(config, head_low_occur_dataset, test_sampler, shuffle=False)
            tail_low_occur_data = get_dataloader(config, 'evaluation')(config, tail_low_occur_dataset, test_sampler, shuffle=False)



        if config['save_dataloaders']:
            save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    logger = getLogger()
    logger.info(
        set_color('[Training]: ', 'pink') + set_color('train_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["train_batch_size"]}]', 'yellow') + set_color(' negative sampling', 'cyan') + ': ' +
        set_color(f'[{config["neg_sampling"]}]', 'yellow')
    )
    logger.info(
        set_color('[Evaluation]: ', 'pink') + set_color('eval_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["eval_batch_size"]}]', 'yellow') + set_color(' eval_args', 'cyan') + ': ' +
        set_color(f'[{config["eval_args"]}]', 'yellow')
    )
    if co_occurence:
        return train_data, valid_data, test_data, head_high_occur_data, tail_high_occur_data, head_mid_occur_data, tail_mid_occur_data, head_low_occur_data, tail_low_occur_data
    else:
        return train_data, valid_data, test_data, head_test_data, tail_test_data

import importlib
from recbole_gnn.utils import _get_customized_dataloader
from recbole.data.utils import data_preparation as recbole_data_preparation

def data_preparation_gnn_head_tail(config, dataset, class_type='SessionGraphDataset'):
    """Split the dataset by :attr:`config['eval_args']` and create training, validation and test dataloader.
    Note:
        If we can load split dataloaders by :meth:`load_split_dataloaders`, we will not create new split dataloaders.
    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    seq_module_path = '.'.join(['recbole_gnn.model.sequential_recommender', config['model'].lower()])
    if importlib.util.find_spec(seq_module_path, __name__) or class_type == 'SessionGraphDataset':
        # Special condition for sequential models of RecBole-Graph
        dataloaders = load_split_dataloaders(config)
        if dataloaders is not None:
            train_data, valid_data, test_data = dataloaders
        else:
            built_datasets = dataset.build()
            train_dataset, valid_dataset, test_dataset = built_datasets
            train_sampler, valid_sampler, test_sampler = create_samplers(config, dataset, built_datasets)

            train_data = _get_customized_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
            valid_data = _get_customized_dataloader(config, 'evaluation')(config, valid_dataset, valid_sampler, shuffle=False)
            test_data = _get_customized_dataloader(config, 'evaluation')(config, test_dataset, test_sampler, shuffle=False)
            if config['save_dataloaders']:
                save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

            head_test_dataset = copy.deepcopy(test_dataset)
            tail_test_dataset = copy.deepcopy(test_dataset)

            # _, item_pop = train_dataset.get_user_item_pop()
            interaction_data = train_dataset.inter_feat  # interaction for training

            column = 'item_id'
            item_pop = torch.unique(interaction_data[column], return_counts=True)  # (arr(item_id), arr(count))
            item_pop_id = item_pop[0].tolist()
            item_pop_cnt = item_pop[1]

            i_pop = torch.ones(train_dataset.item_num).long()
            i_pop[item_pop_id] = item_pop_cnt

            item_pop = i_pop

            item_indices = np.argsort(item_pop)
            head_item = item_indices[-int(len(item_indices)*0.2):]
            tail_item = item_indices[:-int(len(item_indices)*0.2)]

            head_indices = [i for i, test_item in enumerate(test_dataset['item_id']) if test_item in head_item]
            tail_indices = [i for i, test_item in enumerate(test_dataset['item_id']) if test_item in tail_item]

            head_test_inter_feat = test_dataset[head_indices]
            tail_test_inter_feat = test_dataset[tail_indices]

            head_test_dataset.inter_feat = head_test_inter_feat
            tail_test_dataset.inter_feat = tail_test_inter_feat

            head_test_data = _get_customized_dataloader(config, 'evaluation')(config, head_test_dataset, test_sampler, shuffle=False)
            tail_test_data = _get_customized_dataloader(config, 'evaluation')(config, tail_test_dataset, test_sampler, shuffle=False)

            head_test_data.dataset.session_graph_construction()
            tail_test_data.dataset.session_graph_construction()

        logger = getLogger()
        logger.info(
            set_color('[Training]: ', 'pink') + set_color('train_batch_size', 'cyan') + ' = ' +
            set_color(f'[{config["train_batch_size"]}]', 'yellow') + set_color(' negative sampling', 'cyan') + ': ' +
            set_color(f'[{config["neg_sampling"]}]', 'yellow')
        )
        logger.info(
            set_color('[Evaluation]: ', 'pink') + set_color('eval_batch_size', 'cyan') + ' = ' +
            set_color(f'[{config["eval_batch_size"]}]', 'yellow') + set_color(' eval_args', 'cyan') + ': ' +
            set_color(f'[{config["eval_args"]}]', 'yellow')
        )
        return train_data, valid_data, test_data, head_test_data, tail_test_data
    else:
        return recbole_data_preparation(config, dataset)