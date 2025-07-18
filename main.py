import argparse
import time
import copy
import torch
import numpy as np
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_trainer, init_seed, set_color

from collections import OrderedDict

from core_ave import COREave
from core_trm import COREtrm
from slist import SLIST
from swalk import SWalk
from link import LINK

from tqdm import tqdm

from IPython import embed

def run_single_model(args, args_unparsed):
    # configurations initialization
    if args.model == 'core_ave':
        model = COREave
    elif args.model == 'core_trm':
        model = COREtrm
    elif args.model == 'slist':
        model = SLIST
    elif args.model == 'swalk':
        model = SWalk
    elif args.model == 'link':
        model = LINK
    else:
        raise ValueError('Unknown model: {}'.format(args.model))

    if args.all_config == 'none':
        config_file_list = ['props/overall.yaml']
    else:
        config_file_list = [args.all_config]

    if args.config == 'none':
        config_file_list.append(f'props/{args.model}.yaml')
    else:
        config_file_list.append(args.config)

    if args.config2 == 'none':
        pass
    else:
        config_file_list.append(args.config2)


    config = Config(
        model=model,
        dataset=args.dataset, 
        config_file_list=config_file_list,
    )
    
    # revise unparsed arguments
    for i, arg in enumerate(args_unparsed):
        if arg.startswith('--'):
            arg_name = arg[2:]
            arg_value = args_unparsed[i+1]
            if arg_name in config:
                if isinstance(config[arg_name], bool):
                    config[arg_name] = arg_value.lower() == 'true'
                elif isinstance(config[arg_name], float):
                    config[arg_name] = float(arg_value)
                elif isinstance(config[arg_name], int):
                    config[arg_name] = int(arg_value)
                else:
                    config[arg_name] = arg_value

    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    # config logging
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    if args.model == 'core_ave':
        model = COREave(config, train_data.dataset).to(config['device'])
    elif args.model == 'core_trm':
        model = COREtrm(config, train_data.dataset).to(config['device'])
    elif args.model == 'slist':
        model = SLIST(config, train_data.dataset).to(config['device'])
    elif args.model == 'swalk':
        model = SWalk(config, train_data.dataset).to(config['device'])
    elif args.model == 'link':
        model = LINK(config, train_data.dataset).to(config['device'])
    else:
        raise ValueError('model can only be "ave" or "trm" or "item".')
    logger.info(model)
    train_data.dataset.logger = logger

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    # model training
    linear_model_list = ['slis', 'slist', 'slist_kd', 'slist_kd_full', 'swalk', 'link', 'linear_ensemble']
    if args.model in linear_model_list:
        best_valid_score, best_valid_result = None, None
    else:
        try:
            best_valid_score, best_valid_result = trainer.fit(
                train_data, valid_data, saved=True, show_progress=config['show_progress']
            )
        except KeyboardInterrupt:
            logger.info('KeyboardInterrupt, stop training.')
            best_valid_score, best_valid_result = None, None
        except:
            # print detailed error message
            import traceback
            traceback.print_exc()
            best_valid_score, best_valid_result = None, None

    # model evaluation
    load_best_model = True 
    if args.model in linear_model_list:
        load_best_model = False
    try:
        test_result = trainer.evaluate(test_data, load_best_model=load_best_model, show_progress=config['show_progress'])
    except:
        test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
    
    # save teacher matrix
    if 'save_teacher_matrix' in config:
        dense_item_item_matrix = make_teacher_matrix_for_link(args, config, model, train_data)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

def make_teacher_matrix_for_link(args, config, model, train_data):
    start_time = time.time()
    output_folder = f'saved_models_for_embedding/linear_teacher_{args.dataset}_{args.model}'
    # Initialize dense item-item matrix
    dense_item_item_matrix = np.zeros((model.n_items, model.n_items), dtype=np.float32)
    
    # Copy train_data
    single_session_data = copy.deepcopy(train_data)
    
    # change train_data to single item session
    single_session_data.dataset.inter_feat = single_session_data.dataset.inter_feat[:model.n_items]
    assert len(single_session_data.dataset.inter_feat) == model.n_items, f"Size of single_session_data.dataset.inter_feat is not equal to model.n_items ({single_session_data.dataset.inter_feat.shape[0]} != {model.n_items})"
    single_session_data.dataset.inter_feat['item_id_list'][:, 0] = torch.arange(model.n_items, device=config['device'])
    single_session_data.dataset.inter_feat['item_id_list'][:, 1:] = 0
    single_session_data.dataset.inter_feat['item_length'] = torch.ones_like(single_session_data.dataset.inter_feat['item_length'])
    
    model.eval()

    iter_data = tqdm(single_session_data,total=len(single_session_data),ncols=100,desc=set_color(f"Single-session data", 'pink'),)
    
    for batch_idx, interaction in enumerate(iter_data):
        interaction = interaction.to(model.device)

        with torch.no_grad():
            scores = model.full_sort_predict(interaction)
            if args.model == 'core_trm':
                scores = scores * model.temperature # temperature scaling
            if args.model == 'msgifsr' and model.use_logit:
                scores = scores / model.temperature # temperature scaling
        
        # save dense matrix
        item_idx = interaction['item_id_list'][:, 0].cpu().numpy()
        dense_item_item_matrix[item_idx] = scores.cpu().numpy()

    # set padding item to 0
    dense_item_item_matrix[0, :] = 0
    print(f'Elapsed time for extraction teacher matrix: {time.time()-start_time:.2f}s')
    os.makedirs(f'{output_folder}', exist_ok=True)
    np.save(f'{output_folder}/dense_matrix.npy', dense_item_item_matrix)
    print(f'{output_folder}/dense_matrix.npy shape: {dense_item_item_matrix.shape} saved!')
    
    return dense_item_item_matrix

import logging
import colorlog
import os
import re

from colorama import init
from recbole.utils.utils import get_local_time, ensure_dir
log_colors_config = {
    'DEBUG': 'cyan',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}

class RemoveColorFilter(logging.Filter):
    def filter(self, record):
        if record:
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            record.msg = ansi_escape.sub('', str(record.msg))
        return True

def init_logger(config):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Example:
        >>> logger = logging.getLogger(config)
        >>> logger.debug(train_state)
        >>> logger.info(train_result)
    """
    init(autoreset=True)
    LOGROOT = './log/'
    dir_name = os.path.dirname(LOGROOT)
    ensure_dir(dir_name)
    model_name = os.path.join(dir_name, config['model'])
    ensure_dir(model_name)
    dataset_name = os.path.join(model_name, config['dataset'])
    ensure_dir(dataset_name)
    
    if 'folder' in config:
        folder_name = config['folder']
        folder_name = os.path.join(dataset_name, folder_name)
        ensure_dir(folder_name)
        logfilename = folder_name + f"/{get_local_time()}_.log"
    else:
        logfilename = dataset_name + f"/{get_local_time()}_.log"

    logfilepath = logfilename

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(log_color)s%(asctime)-15s %(levelname)s  %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = colorlog.ColoredFormatter(sfmt, sdatefmt, log_colors=log_colors_config)
    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG
    elif config['state'].lower() == 'error':
        level = logging.ERROR
    elif config['state'].lower() == 'warning':
        level = logging.WARNING
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)
    remove_color_filter = RemoveColorFilter()
    fh.addFilter(remove_color_filter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(level=level, handlers=[sh, fh])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='trm', help='ave or trm or item or item2')
    parser.add_argument('--dataset', type=str, default='diginetica', help='diginetica, nowplaying, retailrocket, tmall, yoochoose')
    parser.add_argument('--all_config', type=str, default='none', help='none or path to all_config file')
    parser.add_argument('--config', type=str, default='none', help='none or path to config file')
    parser.add_argument('--config2', type=str, default='none', help='none or path to config file')
    
    args, args_unparsed = parser.parse_known_args()

    run_single_model(args, args_unparsed)