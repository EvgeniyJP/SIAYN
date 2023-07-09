import gradio as gr
import numpy as np
import time
from core.StyleTransferSolver import StyleTransferOpt
from util.img import prepare_transforms
from util.logger import Logger
import torch
import os


def style_transfer_wrapper(scaler, init_method, backbone, lr, opt, style_weight, content_weight, tv_weight, norm,
                           style_resize, content_layers, style_layers, num_steps, content_image, style_image, save):

    """
    :param save: If true, generated images will be saved
    :param style_image: Image with that will be transferred
    :param content_image: Content for generation
    :param scaler: Scaler factor for image to be down/up scale, default 1
    :param init_method: Initialization method for generated image, options are: ['random', 'content', 'style']
    default, 'random'
    :param backbone: Backbone that will be used for feature extraction, options are: ['VGG19', 'VGG16'],
    default ='VGG19'
    :param content_layers: List of layers that will be used for content features extraction from backbone
    default =  ['conv_4']
    :param style_layers: List of layers that will be used for style features extraction from backbone,
    default =  ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    :param lr: Learning rate that optimizer will use. default lr=0.01
    :param opt: Optimizer type, the options are ['Adam', 'LFBGS'], default 'LFBGS'
    :param num_steps: Number of steps for optimization, default 2000
    :param style_weight: If list, will perform multi style transfer. Will be used in weighted sum for calculation
     total loss, default 1e5
    :param content_weight: Weight that will be used in sum of all losses for total loss, default 1e3
    :param norm: If true, ImageNet optimization will be performed for input images, default True
    :param tv_weight: If not None, total variation loss will be used with variable as weight in total loss,
     default None
     weight will be used. Default None
    :param style_resize: If true, style image will be resized to be in the same size as content image.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_timer = time.time()
    log = Logger(save=False)
    style_transfer = StyleTransferOpt(logger=log, scaler=scaler, init_method=init_method, backbone=backbone, lr=lr,
                                      opt=opt, num_steps=num_steps, style_weight=style_weight,
                                      content_weight=content_weight, tv_weight=tv_weight, norm=norm,
                                      style_layers=style_layers, style_layers_weights=None,
                                      content_layers=content_layers, style_resize=style_resize)

    im_size = [scaler * content_image.size[0], scaler * content_image.size[1]]
    im_size = (int(im_size[1]), int(im_size[0]))

    t = prepare_transforms(im_size=im_size, norm=norm)
    im_content = t(content_image)

    if style_resize:
        im_style = t(style_image)

    else:
        t = prepare_transforms(im_size=None, norm=norm)
        im_style = t(style_image)

    im_style = im_style.unsqueeze(0).to(device, torch.float)
    im_content = im_content.unsqueeze(0).to(device, torch.float)

    os.makedirs('gui_result', exist_ok=True)
    output_im = style_transfer.style_opt_for_gui(im_content=im_content, im_style=im_style, start_timer=start_timer,
                                                 save=save, path_output='gui_result', output_img_name='temp')

    yield from output_im


def gui_creation():
    """
    Function for creation gradio GUI for basic interface
    :return:
    """
    with gr.Blocks() as demo:
        gr.Markdown("'Style is all you need.'-  Play with some magic" )

        with gr.Tab("Style Transfer"):
            with gr.Row():
                content_image_input = gr.Image(label='Content Image', type='pil')
                style_image_input = gr.Image(label='Style Image',  type='pil')

            with gr.Row():
                backbone_gr = gr.Dropdown(['VGG19', 'VGG16'], value='VGG19', label='Backbone')
                opt_gr = gr.Dropdown(['Adam', 'LFBGS'], value='LFBGS', label='Optimizer')
                init_gr= gr.Dropdown(['random', 'content', 'style'], value='random', label='Initialization method')
                scaler_gr = gr.Slider(minimum=0.1, maximum=2.0, value=1, step=0.05, label='Scale size')
            with gr.Row():
                lr_gr = gr.Number(value=0.1, label='Learning rate',minimum=1e-7, maximum=2)
                style_weight_gr = gr.Number(value=1e5, label='Style weight', minimum=1e-5, maximum=1e7)
                content_weight_gr = gr.Number(value=1e3, label='Content weight', minimum=1e-5, maximum=1e7)
                tv_weight_gr = gr.Number(value=1e-2, label='Total variation weight', minimum=1e-5, maximum=1e7)
                num_steps_gr = gr.Number(value=1000, label='Number of steps', minimum=1, maximum=10000)
                with gr.Row():
                    norm_gr = gr.Checkbox(value=True, label='Image normalization')
                    style_resize_gr = gr.Checkbox(value=True, label='Resize style image')
                    save_gr = gr.Checkbox(value=True, label='Save the results')

            with gr.Row():
                content_layers_gr = gr.CheckboxGroup(choices=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6',
                                                              'conv_7', 'conv_8', 'conv_9', 'conv_10', 'conv_11', 'conv_12',
                                                              'conv_13', 'conv_14', 'conv_15', 'conv_16'],
                                                     label='Content layers', value=['conv_8'])
                style_layers_gr = gr.CheckboxGroup(choices=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6',
                                                              'conv_7', 'conv_8', 'conv_9', 'conv_10', 'conv_11', 'conv_12',
                                                              'conv_13', 'conv_14', 'conv_15', 'conv_16'],
                                                     label='Style layers',
                                                   value=['conv_1', 'conv_3', 'conv_5', 'conv_8', 'conv_11'])

            start_style_transfer_button = gr.Button("Start")
            with gr.Row():
                styled_image_output = gr.Image(label='Styled Image')


        start_style_transfer_button.click(style_transfer_wrapper, inputs=[scaler_gr, init_gr, backbone_gr, lr_gr, opt_gr,
                                                                          style_weight_gr, content_weight_gr, tv_weight_gr,
                                                                          norm_gr, style_resize_gr, content_layers_gr,
                                                                          style_layers_gr, num_steps_gr,
                                                                          content_image_input, style_image_input, save_gr],
                                          outputs=styled_image_output)
    demo.queue()
    demo.launch()

