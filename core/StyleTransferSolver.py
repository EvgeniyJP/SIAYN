import os
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchmetrics import TotalVariation
import time
from util.img import load_image, save_img_tensor, gram_matrix, img2vid, img2gif
from core.Backbone import VGG19, VGG16
from torchvision import transforms


class StyleTransferOpt:
    """
    Class for optimization generated image to have content of target image with style of style target
    """
    def __init__(self, logger, scaler=1.,  device=None, init_method='random', backbone='VGG19', content_layers=None,
                 style_layers=None, lr=0.01, opt='Adam', num_steps = 2000, style_weight=1e5, content_weight=1e3,
                 norm=True, tv_weight=None, style_layers_weights=None, style_resize=True):
        """

        :param logger: Logger from Logger class, helps organize the logs
        :param scaler: Scaler factor for image to be down/up scale, default 1
        :param device: Torch device that will be used for optimization process, default GPU if available
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
        :param style_layers_weights: If not None, for each layer in backbone feature extraction different
         weight will be used. Default None
        :param style_resize: If true, style image will be resized to be in the same size as content image.
        """
        # Initialize the parameters
        self.logger = logger
        self.scaler = scaler
        self.norm = norm
        self.style_resize = style_resize
        self.opt = opt
        self.lr = lr
        self.num_step = num_steps

        # Check init method
        init_options = ['random', 'content', 'style']
        if init_method not in init_options:
            self.logger.error('Wrong init_method, options are:', init_options)
            raise Exception('Wrong init_method, options are:' + str(init_options))
        self.init_method = init_method

        # Check backbone
        backbone_options = ['VGG19', 'VGG16']
        if backbone not in backbone_options:
            self.logger.error('Wrong backbone, options are:', backbone_options)
            raise Exception('Wrong backbone, options are:' + str(backbone_options))
        self.backbone = backbone

        # Check optimizer
        optimizer_options = ['Adam', 'LFBGS']
        if opt not in optimizer_options:
            self.logger.error('Wrong opt, options are:', optimizer_options)
            raise Exception('Wrong opt, options are:' + str(optimizer_options))


        self.style_weight = style_weight
        self.content_weight = content_weight
        self.tv_weight = tv_weight

        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = torch.device(device)

        if not content_layers:
            self.content_layers = ['conv_4']
        else:
            self.content_layers = content_layers

        if style_layers:
            self.style_layers = style_layers
        else:
            self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        if style_layers_weights:
            if len(style_layers_weights) == len(style_layers):
                self.style_layers_weights = style_layers_weights
        else:
            self.style_layers_weights = None

        # Initialize model for feature extraction
        self.model = None
        self.create_model()

        self.logger.info('StyleTransferOpt is ready.. Using device:', self.device)
        self.logger.info('Model for features (backbone) is:', self.backbone)
        self.logger.info("Parameters for stylization are: |Image scaler:", scaler, '|init_method:', init_method,
                         '|LR:', lr, '|Optimizer:', opt, '|num_steps:', num_steps, '|style_weight:', style_weight,
                         '|content_weight:', content_weight,'|tv_weight:', tv_weight, '|Normalize image:', norm,
                         '|Style resize:', style_resize)
        self.logger.info('Layers for content:', self.content_layers, 'Layers for style:', self.style_layers)

        if self.style_layers_weights:
            self.logger.info('Weights for layers style:', self.style_layers_weights)

    def create_model(self):
        """
        Function for initialization of feature extraction model,
        checking type of backbone and build one
        """
        if self.backbone == 'VGG19':
            self.model = VGG19(device=self.device)
            self.logger.info('The model for features (backbone) is created.', self.backbone)
        elif self.backbone == 'VGG16':
            self.model = VGG16(device=self.device)
            self.logger.info('The model for features (backbone) is created.', self.backbone)

    def get_optimizer(self, img):
        """
        Function for initialization optimization that will be used, with parameters like learning rate
        :param img: Generated image that will be optimized
        """
        if self.opt == 'Adam':
            self.logger.info('Optimizer is created.', self.opt)
            return optim.Adam([img], lr=self.lr)
        elif self.opt == 'LFBGS':
            self.logger.info('Optimizer is created.', self.opt)
            return optim.LBFGS([img], max_iter=1, lr=self.lr)

    def init_img(self, im_content, im_style):
        """
        Function for initialization generated image, will initial0ized with random, content or style
        :param im_content: content image
        :param im_style:  style image
        :return: initialized generated image
        """
        # Init output
        if self.init_method == 'random':
            im_output = torch.randn(im_content.data.size(), device=self.device)
            self.logger.info('Output image is initialized, the method is:', self.init_method)
            return im_output

        elif self.init_method == 'content':
            im_output = im_content.clone()
            self.logger.info('Output image is initialized, the method is:', self.init_method)
            return im_output

        elif self.init_method == 'style':
            if self.style_resize:
                im_output = im_style.clone()
            else:
                transform = transforms.Resize((int(im_content.shape[2]), int(im_content.shape[3])))
                image = transform(im_style)
                im_output = image.clone()

            self.logger.info('Output image is initialized, the method is:', self.init_method)
            return im_output

        else:
            im_output = im_content.clone()
            self.logger.info('Output image is initialized, the method is:', self.init_method)
            return im_output

    def calc_loss(self, im_out, style_features, content_features):
        """
        Function for loss calculation
        :param im_out: Generated image
        :param style_features: Style features of original style
        :param content_features: Content features of original content
        :return: 1. total loss of relevant components, 2. Content loss, 3. Style loss, 4. TV loss
        TV - Total variation
        """
        # Extract features of current generated image for style and content
        list_out_style, list_out_content = self.model(im_out)

        # Calculate loss for content
        loss_content = .0
        for content_out, content_in in zip(list_out_content, content_features):
            loss_content += torch.nn.MSELoss(reduction='mean')(content_out[0], content_in[0])
        loss_content /= len(list_out_content)

        # Calculate style loss
        weighted_style_loss = 0.
        if isinstance(self.style_weight, list):

            loss_style = []
            for style_ind in range(len(self.style_weight)):
                current_style_loss = 0.
                style_layer_ind = 0
                for style_out, style_in in zip(list_out_style, style_features[style_ind]):
                    layer_loss = torch.nn.MSELoss(reduction='sum')(gram_matrix(style_out), gram_matrix(style_in))
                    if self.style_layers_weights:
                        layer_loss = self.style_layers_weights[style_layer_ind] * layer_loss
                        current_style_loss += layer_loss
                        style_layer_ind += 1
                    else:
                        current_style_loss += layer_loss
                current_style_loss /= len(list_out_style)
                weighted_style_loss += self.style_weight[style_ind] * current_style_loss
                loss_style.append(current_style_loss)

        else:
            loss_style = .0
            style_layer_ind = 0
            for style_out, style_in in zip(list_out_style, style_features):
                layer_loss = torch.nn.MSELoss(reduction='sum')(gram_matrix(style_out), gram_matrix(style_in))
                if self.style_layers_weights:
                    layer_loss = self.style_layers_weights[style_layer_ind] * layer_loss
                    loss_style += layer_loss
                    style_layer_ind += 1
                else:
                    loss_style += layer_loss
            loss_style /= len(list_out_style)
            weighted_style_loss += self.style_weight * loss_style

        if self.tv_weight:
            tv = TotalVariation().to(self.device)
            tv_loss = tv(im_out)
            total_loss = weighted_style_loss + self.content_weight * loss_content + self.tv_weight * tv_loss
        else:
            tv_loss = torch.tensor([-1])
            total_loss = weighted_style_loss + self.content_weight * loss_content

        return total_loss, loss_content, loss_style, tv_loss

    def style_opt(self, im_content, im_style, start_timer = time.time(), path_output='', output_img_name='', save=False):
        """
        Function for optimization generated image
        :param im_content: content image, for content feature extraction
        :param im_style: style image (or list of images), for style feature extraction
        :param start_timer: Start time, default - now time, for calculation running time
        :param path_output: Output path for data storage
        :param output_img_name: Output name for saving the result
        :param save: If true training images will be saved
        """
        # Init output image
        if isinstance(self.style_weight, list):
            im_output = self.init_img(im_content, im_style[0])
        else:
            im_output = self.init_img(im_content, im_style)

        im_output = Variable(im_output, requires_grad=True)
        optimizer = self.get_optimizer(im_output)

        # Prepare features of content and style, if style image is list prepare all styles images
        if isinstance(self.style_weight, list):
            style_features = []
            for ind_style in range(len(self.style_weight)):
                s, _ = self.model(im_style[ind_style])
                style_features.append(s)
        else:
            style_features, _ = self.model(im_style)
        _, content_features = self.model(im_content)

        # Start the optimization
        if self.opt == 'Adam':
            scheduler = StepLR(optimizer, step_size=6000, gamma=0.1)
            for i in range(self.num_step):
                with torch.no_grad():
                    im_output.clamp(min=0, max=1)

                lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
                if torch.is_grad_enabled():
                    optimizer.zero_grad(set_to_none=True)

                total_loss, content_loss, style_loss, tv_loss = self.calc_loss(im_output, style_features,
                                                                               content_features)
                total_loss.backward()

                if isinstance(self.style_weight, list):
                    style_loss_print = ''
                    for ind_style in range(len(self.style_weight)):
                        style_loss_print += str(round(style_loss[ind_style].item(), 6))
                        style_loss_print += ','
                else:
                    style_loss_print = str(round(style_loss.item(), 6))

                self.logger.info(
                    f'iteration: {i:04}, run time: {time.time() - start_timer:4.0f}, lr: {lr:1.6f}, total loss={total_loss.item():12.2f}, content_loss={content_loss.item():12.3f}, style loss={style_loss_print}, Tv loss={tv_loss.item():12.6f}')

                if save:
                    if i % 50 == 0:
                        save_img_tensor(img=im_output, p_save=path_output, name=str(i).zfill(5) + '.png')

                optimizer.step()
                scheduler.step()

        elif self.opt == 'LFBGS':
            iter_run = [0]
            for i in range(int(self.num_step)):
                def closure():
                    with torch.no_grad():
                        im_output.clamp(min=0, max=1)
                    if torch.is_grad_enabled():
                        optimizer.zero_grad(set_to_none=True)
                    total_loss, content_loss, style_loss, tv_loss = self.calc_loss(im_output, style_features,
                                                                                   content_features)

                    if total_loss.requires_grad:
                        total_loss.backward()
                    with torch.no_grad():
                        if isinstance(self.style_weight, list):
                            style_loss_p = ''
                            for ind_s in range(len(self.style_weight)):
                                style_loss_p += str(round(style_loss[ind_s].item(), 6))
                                style_loss_p += ','
                        else:
                            style_loss_p = str(round(style_loss.item(), 6))

                        self.logger.info(
                            f'iteration: {iter_run[0]:04}, run time: {time.time() - start_timer:4.0f}, total loss={total_loss.item():12.2f}, content_loss={content_loss.item():12.3f}, style loss={style_loss_p}, Tv loss={tv_loss.item():12.6f}')

                    iter_run[0] += 1
                    return total_loss

                if i % 50 == 0:
                    if save:
                        save_img_tensor(img=im_output, p_save=path_output, name=str(i).zfill(5) + '.png')
                optimizer.step(closure)


        self.logger.info('Total train time:', time.time() - start_timer)
        if save:
            save_img_tensor(img=im_output, p_save=path_output, name=output_img_name)
        self.logger.info("End image saved")

    def style_opt_for_gui(self, im_content, im_style, start_timer = time.time(), path_output='', output_img_name='', save=False):
        """
        Function for optimization generated image with yield for gui
        :param im_content: content image, for content feature extraction
        :param im_style: style image (or list of images), for style feature extraction
        :param start_timer: Start time, default - now time, for calculation running time
        :param path_output: Output path for data storage
        :param output_img_name: Output name for saving the result
        :param save: If true training images will be saved
        """

        # Init output image
        if isinstance(self.style_weight, list):
            im_output = self.init_img(im_content, im_style[0])
        else:
            im_output = self.init_img(im_content, im_style)

        im_output = Variable(im_output, requires_grad=True)
        optimizer = self.get_optimizer(im_output)

        # Prepare features of content and style, if style image is list prepare all styles images
        if isinstance(self.style_weight, list):
            style_features = []
            for ind_style in range(len(self.style_weight)):
                s, _ = self.model(im_style[ind_style])
                style_features.append(s)
        else:
            style_features, _ = self.model(im_style)
        _, content_features = self.model(im_content)

        # Start the optimization
        if self.opt == 'Adam':
            scheduler = StepLR(optimizer, step_size=6000, gamma=0.1)
            for i in range(self.num_step):
                with torch.no_grad():
                    im_output.clamp(min=0, max=1)

                lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
                if torch.is_grad_enabled():
                    optimizer.zero_grad(set_to_none=True)

                total_loss, content_loss, style_loss, tv_loss = self.calc_loss(im_output, style_features,
                                                                               content_features)
                total_loss.backward()

                if isinstance(self.style_weight, list):
                    style_loss_print = ''
                    for ind_style in range(len(self.style_weight)):
                        style_loss_print += str(round(style_loss[ind_style].item(), 6))
                        style_loss_print += ','
                else:
                    style_loss_print = str(round(style_loss.item(), 6))

                self.logger.info(
                    f'iteration: {i:04}, run time: {time.time() - start_timer:4.0f}, lr: {lr:1.6f}, total loss={total_loss.item():12.2f}, content_loss={content_loss.item():12.3f}, style loss={style_loss_print}, Tv loss={tv_loss.item():12.6f}')

                if save:
                    out_img = im_output.squeeze(axis=0).detach().to('cpu')
                    dump_img = out_img.clone()
                    dump_img = dump_img.permute(1, 2, 0)
                    if self.norm:
                        dump_img = torch.clamp(
                            dump_img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406]), 0,
                            1)
                    yield dump_img.numpy()

                    if i % 50 == 0:

                        save_img_tensor(img=im_output, p_save=path_output, name=str(i).zfill(5) + '.png')

                optimizer.step()
                scheduler.step()

        elif self.opt == 'LFBGS':
            iter_run = [0]
            for i in range(int(self.num_step)):
                def closure():
                    with torch.no_grad():
                        im_output.clamp(min=0, max=1)
                    if torch.is_grad_enabled():
                        optimizer.zero_grad(set_to_none=True)
                    total_loss, content_loss, style_loss, tv_loss = self.calc_loss(im_output, style_features,
                                                                                   content_features)

                    if total_loss.requires_grad:
                        total_loss.backward()
                    with torch.no_grad():
                        if isinstance(self.style_weight, list):
                            style_loss_p = ''
                            for ind_s in range(len(self.style_weight)):
                                style_loss_p += str(round(style_loss[ind_s].item(), 6))
                                style_loss_p += ','
                        else:
                            style_loss_p = str(round(style_loss.item(), 6))

                        self.logger.info(
                            f'iteration: {iter_run[0]:04}, run time: {time.time() - start_timer:4.0f}, total loss={total_loss.item():12.2f}, content_loss={content_loss.item():12.3f}, style loss={style_loss_p}, Tv loss={tv_loss.item():12.6f}')

                    iter_run[0] += 1
                    return total_loss

                if i % 50 == 0:
                    out_img = im_output.squeeze(axis=0).detach().to('cpu')
                    dump_img = out_img.clone()
                    dump_img = dump_img.permute(1, 2, 0)
                    if self.norm:
                        dump_img = torch.clamp(
                            dump_img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406]), 0,
                            1)
                    yield dump_img.numpy()

                    if save:
                        save_img_tensor(img=im_output, p_save=path_output, name=str(i).zfill(5) + '.png')
                optimizer.step(closure)


        self.logger.info('Total train time:', time.time() - start_timer)
        out_img = im_output.squeeze(axis=0).detach().to('cpu')
        dump_img = out_img.clone()
        dump_img = dump_img.permute(1, 2, 0)
        if self.norm:
            dump_img = torch.clamp(
                dump_img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406]), 0,
                1)
        yield dump_img.numpy()

        if save:
            save_img_tensor(img=im_output, p_save=path_output, name=output_img_name)
        self.logger.info("End image saved")

    def style_transfer(self, path_content, path_style, path_output, save=True):
        """
        Wrapper function for style transfer trainer
        :param path_content: path to content image
        :param path_style: path to style image, of list of path for multi style transfer
        :param path_output: Path to output
        :param save: True if save results in training
        """
        start_timer = time.time()

        self.logger.info('Starting style transfer for content:', path_content, ', style:', path_style)
        self.logger.info("Output path is:", path_output)

        # Output image name, combination of content and style names
        output_img_name = os.path.split(path_content)[1].split('.')[0]
        if isinstance(path_style, list):
            for ind_style in range(len(path_style)):
                output_img_name += '!-!'
                output_img_name += os.path.split(path_style[ind_style])[1].split('.')[0]
        else:
            output_img_name += '!-!'
            output_img_name += os.path.split(path_style)[1].split('.')[0]

        # Output directory creation
        os.makedirs(path_output, exist_ok=True)

        # Load images
        im_content = load_image(path_content, scaler=self.scaler, device=self.device, norm=self.norm)

        if self.style_resize:
            if isinstance(self.style_weight, list):
                im_style = []
                for ind_style in range(len(self.style_weight)):
                    im_style.append(load_image(path_style[ind_style],
                                               resize=(int(im_content.shape[2]), int(im_content.shape[3])),
                                               device=self.device, norm=self.norm))
            else:
                im_style = load_image(path_style, resize=(int(im_content.shape[2]), int(im_content.shape[3])),
                                      device=self.device, norm=self.norm)
        else:
            if isinstance(self.style_weight, list):
                im_style = []
                for ind_style in range(len(self.style_weight)):
                    im_style.append(load_image(path_style[ind_style], device=self.device, norm=self.norm))
            else:
                im_style = load_image(path_style, device=self.device, norm=self.norm)

        self.style_opt(im_content=im_content, im_style=im_style, start_timer=start_timer, path_output=path_output,
                     output_img_name=output_img_name, save=save)
        self.logger.info("Reference images are loaded, size is:", im_content.shape)

