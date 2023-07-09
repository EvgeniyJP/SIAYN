import argparse
import os
import yaml

from util.img import img_denoise, apply_mask, original_color_transform, create_mask, img2gif
from util.logger import Logger

if __name__ == '__main__':
    # File for running postprocessing on generated image
    # Can be run from the terminal
    # Loading config file, default file exist in config directory

    p_root = os.path.dirname(os.path.abspath(__file__))
    p_save = os.path.join(p_root, 'result')

    # Load config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="config file name", default='postprocessing_default')
    args = parser.parse_args()

    path_config_file = os.path.join(p_root, 'config', args.config_file + '.yaml')
    with open(path_config_file) as f:
        dict_param = yaml.load(f, Loader=yaml.SafeLoader)

    for t in dict_param:
        log = Logger(save=False)
        log.title('Starting postprocessing for:', t)

        if dict_param[t]['use_denoise']:
            log.info("Performing denoising..")
            img_denoise(img_path=dict_param[t]['img_path'], p3=dict_param[t]['p3'], p4=dict_param[t]['p4'],
                        p5=dict_param[t]['p5'], p6=dict_param[t]['p5'], save=True)

        if dict_param[t]['use_mask_creation']:
            log.info("Creating mask..")
            create_mask(p_image=os.path.join(p_root, 'data', 'content', dict_param[t]['content']),
                        scaler=dict_param[t]['scaler'], save=True)

        if dict_param[t]['use_mask']:
            log.info("Applying mask..")
            apply_mask(p_content=os.path.join(p_root, 'data', 'content', dict_param[t]['content']),
                       p_styled=dict_param[t]['styled'],
                       p_mask=os.path.join(p_root, 'data', 'content', dict_param[t]['mask']),
                       save=True)

        if dict_param[t]['use_histogram']:
            log.info("Applying histogram matching..")
            original_color_transform(target_path=os.path.join(p_root, 'data', 'style', dict_param[t]['target']),
                                     source_path=dict_param[t]['source'], save=True)

        if dict_param[t]['use_gif_creation']:
            log.info("Creating gif..")
            img2gif(path=dict_param[t]['path'], out_name=dict_param[t]['out_name'], fps=dict_param[t]['fps'])



