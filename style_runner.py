import yaml
import argparse
import os
from util.logger import Logger
from core.StyleTransferSolver import StyleTransferOpt

if __name__ == '__main__':

    # File for running style transfer class
    # Can be run from the terminal
    # Loading config file, default file exist in config directory

    p_root = os.path.dirname(os.path.abspath(__file__))
    p_save = os.path.join(p_root, 'result')

    # Load config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="config file name", default='style_transfer_default')
    args = parser.parse_args()

    path_config_file = os.path.join(p_root, 'config', args.config_file + '.yaml')
    with open(path_config_file) as f:
        dict_param = yaml.load(f, Loader=yaml.SafeLoader)

    # Run config files
    for t in dict_param:
        log = Logger(save=False, p_save=os.path.join(p_save, t))
        log.title('Starting to work on:', t)
        param = dict_param[t]
        param['test_name'] = t
        param['style_weight'] = float(param['style_weight'])
        param['content_weight'] = float(param['content_weight'])
        param['tv_weight'] = float(param['tv_weight'])
        param['path_style'] = os.path.join(p_root, 'data', 'style', param['style'])
        param['path_content'] = os.path.join(p_root, 'data', 'content',  param['content'])


        style_transfer = StyleTransferOpt(logger=log, scaler=param['scaler'], init_method=param['init_method'],
                                          backbone=param['backbone'], lr=param['lr'], opt=param['opt'],
                                          num_steps=param['num_steps'], style_weight=param['style_weight'],
                                          content_weight=param['content_weight'], tv_weight=param['tv_weight'],
                                          norm=param['norm'], style_layers=param['style_layers'],
                                          style_layers_weights=param['style_layers_weights'],
                                          content_layers=param['content_layers'], style_resize=param['style_resize'])
        style_transfer.style_transfer(path_style=param['path_style'],
                                      path_content=param['path_content'],
                                      path_output=os.path.join(p_save, param['test_name']))