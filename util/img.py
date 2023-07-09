import os
import numpy as np
from torchvision.models.segmentation import FCN_ResNet101_Weights
from torchvision.utils import save_image
from PIL import Image
import torch
from torchvision import transforms, models
import matplotlib.pyplot as plt
import cv2
import glob
import imageio

def prepare_transforms(im_size=None, norm=True):
    """
    Prepare basic set of image torchvision transforms, image resize, to tensor transform and mean, standard deviation
     normalization.
    mean - [0.485, 0.456, 0.406], standard deviation - [0.229, 0.224, 0.225]
    based on Image Net data, used in training of different models like VGG.
    :param im_size: If given, image resize transformation will be added. Default - None
    :param norm: If True, transformation for image normalization will be added. Default is True.
    :return: List of torchvision transforms
    """
    # Create an empty list of transforms
    list_transforms = []

    # If resize parameters given add resize transform
    if im_size:
        list_transforms.append(transforms.Resize(im_size))
    # Add To tensor transform
    list_transforms.append(transforms.ToTensor())

    # If norm is True add normalization transform
    if norm:
        list_transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transform = transforms.Compose(list_transforms)
    return transform

def load_image(img_path, scaler=1., norm=False, device=torch.device("cpu"), resize=None):
    """
    Function for loading image with to tensor transform, option to resize and normalize
    :param img_path: Path to image to be loaded
    :param scaler: Scale variable for upscale or downscale the image
    :param norm: If true normalization from prepare_transforms will be performed.
    :param device: Torch device that image needed to be loaded to.
    :param resize: If not None, will be used to specific size transformation
    :return: Loaded image in tensor format
    """
    # Load image with PIL in RGB format
    image = Image.open(img_path).convert('RGB')

    # Prepare a new image size with scaler value
    im_size = [scaler * image.size[0], scaler * image.size[1]]
    im_size = (int(im_size[1]), int(im_size[0]))

    # if resize value exist use it instead scaler for image size transformation
    if resize:
        im_size = resize

    # prepare all needed torchvision transforms
    transform = prepare_transforms(im_size=im_size, norm=norm)

    # Perform transform on the image
    image = transform(image)
    image = image.unsqueeze(0).to(device, torch.float)

    return image

def show_img_tensor(img_tensor):
    """
    Function for showing image in tensor format
    :param img_tensor: Image in tensor format
    """
    t = transforms.ToPILImage()
    image = img_tensor.cpu().clone()
    image = image.squeeze(0)
    image = t(image)
    plt.figure()
    plt.imshow(image)
    plt.show()

def save_img_tensor(img, p_save, name='default', norm=True):
    """
    Function for image saving
    :param img: Image that needed to be saved
    :param p_save: path to save the image
    :param name: Name of file that image will be saved will, *.* format
    :param norm: If true, reverse transformation to prepare_transforms normalization will be performed
    """

    # Create output directory
    os.makedirs(p_save, exist_ok=True)

    # Transfer image to cpu
    out_img = img.squeeze(axis=0).detach().to('cpu')
    dump_img = out_img.clone()

    # perform normalization if needed
    if norm:
        dump_img = dump_img.permute(1, 2, 0)
        dump_img = torch.clamp(dump_img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406]), 0,1)
        dump_img = dump_img.permute(2, 0, 1)

    # Save the image
    save_image(dump_img, os.path.join(p_save, name + '.png'))

def gram_matrix(x, norm=True):
    """
    Compute Gram matrix for given features maps, more on GRAM matrix can be found in original paper -
    "Image Style Transfer Using Convolutional Neural Networks"
    :param x: Features maps that were extracted from image with backbone model
    :param norm: If true size normalization will be performed, default -True
    :return: Gram matrix
    """
    (b, ch, h, w) = x.size()
    features = x.view(ch, w * h)
    gram = torch.mm(features,features.t())
    if norm:
        gram /= ch * h * w
    return gram

def img2vid(path, out_name='transformation', fps=30):
    """
    Function for creating video from series of images
    :param fps: FPS of putput video, number of images that will be used per second of video
    :param path: path to images or video creation and where the video will be stored.
    :param out_name: Output name for the video
    """
     # add extension to output file name
    out_file_name = out_name + '.mp4'

    # Find all available images in the folder
    img_array = []
    for filename in glob.glob(path + '/*.png'):
        img = cv2.imread(filename)
        img_array.append(img)

    # Add to seconds in the end of video with final image
    # The idea is to give enough time to see the result
    for i in range(fps * 2):
        img_array.append(img_array[-1])

    # Prepare the video
    height, width, layers = img_array[0].shape
    size = (width, height)
    out = cv2.VideoWriter(os.path.join(path, out_file_name), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def img2gif(path, out_name='transformation', fps=5):
    """
    Function for creating GIF from series of images
    :param fps: FPS of putput gif, number of images that will be used per second of gif
    :param path: path to images or video creation and where the gif will be stored.
    :param out_name: Output name for the gif
    """

    # Add extension for output file name
    out_file_name = out_name + '.gif'

    # find all available images for gif creation
    img_array = []
    for filename in glob.glob(path + '/*.png'):
        img = imageio.v2.imread(filename)
        img_array.append(img)

    # prepare and save the gif
    imageio.mimsave(os.path.join(path, out_file_name), img_array, duration=1000/fps, loop=100)

def img_denoise(img_path, p3=10, p4=10, p5=7, p6=15, save=True):
    """
    Function for image denoising
    :param save: If True, image will be saved in the same path as original with _denoised addition to name.
    :param img_path: Path to image that will be denoised
    :param p3: Size in pixels of the template patch that is used to compute weights.
    :param p4: Size in pixels of the window that is used to compute a weighted average for the given pixel.
    :param p5:  Parameter regulating filter strength for luminance component.
    :param p6: Same as above but for color components // Not used in a grayscale image.
    :return: Denoised image
    """

    # Load the image
    img = cv2.imread(img_path)

    # Denoise the image
    dst = cv2.fastNlMeansDenoisingColored(img, None, p3, p4, p5, p6)

    if save:
        # Prepare output name and save
        path_save = os.path.split(img_path)[0]
        name_save = os.path.split(img_path)[1]
        name_save = name_save.split('.')[0] + '_denoised.' + name_save.split('.')[1]
        path_save = os.path.join(path_save, name_save)
        cv2.imwrite(path_save, dst)

    # return denoised image
    return dst

def match_histograms(source, target):
    """
    Match histogram for one chanel
    https://github.com/titu1994/Neural-Style-Transfer/blob/master/color_transfer.py
    :param target: image which histogram will be matched
    :param source: image which histogram will be changed
    :return: source image chanel with matched histogram
    """
    source_shape = source.shape
    source = source.ravel()
    template = target.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(source_shape)

def original_color_transform(target_path, source_path, save=True):

    """
    Function for color transfer, based on histogram matching
    Based on:    https://github.com/titu1994/Neural-Style-Transfer/blob/master/color_transfer.py

    :param target_path: path to image which histogram will be matched
    :param source_path: path to image which histogram will be changed
    :param save: If True, image will be saved in the same path as original with _colored addition to name.
    :return: Histogram matched source image
    """

    target = np.array(Image.open(target_path).convert('RGB'))
    source = np.array(Image.open(source_path).convert('RGB'))

    for channel in range(3):
        source[:, :, channel] = match_histograms(source[:, :, channel], target[:, :, channel])

    source = Image.fromarray(source).convert('RGB')  # Convert to RGB color space

    if save:
        path_save = os.path.split(source_path)[0]
        name_save = os.path.split(source_path)[1]
        name_save = name_save.split('.')[0] + '_colored.' + name_save.split('.')[1]
        path_save = os.path.join(path_save, name_save)

        source.save(path_save)
    return source

def create_mask(p_image, scaler, save=True):
    """
    Function for creating mask of image
    :param p_image: path to image
    :param scaler: Scaler if want to rescale the image
    :param save: If True, image will be saved in the same path as original with _mask addition to name.
    :return: Mask of image
    """
    fcn = models.segmentation.fcn_resnet101(weights=FCN_ResNet101_Weights.DEFAULT).eval()
    image = load_image(img_path=p_image, scaler=scaler, norm=True, device=torch.device("cpu"), resize=None)
    out = fcn(image)['out']
    mask = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    mask[mask>0] = 255

    if save:
        path_save = os.path.split(p_image)[0]
        name_save = os.path.split(p_image)[1]
        name_save = name_save.split('.')[0] + '_mask.' + name_save.split('.')[1]
        path_save = os.path.join(path_save, name_save)
        cv2.imwrite(path_save, mask)

    return  mask

def apply_mask(p_content, p_styled, p_mask, save=True):
    """
        Function for creating mask of image
        :param p_styled: Path to styled image
        :param p_mask: Path to mask
        :param p_content: path to content image
        :param save: If True, image will be saved in the same path as original with _masked addition to name.
        :return: image with mask applied
        """
    # Load mas, content and styled image, resize all of them to content size
    image_content = np.array(Image.open(p_content).convert('RGB'))
    im_shape = (image_content.shape[1], image_content.shape[0])

    mask = cv2.imread(p_mask)
    mask = cv2.resize(mask, dsize=im_shape)

    styled_image = np.array(Image.open(p_styled).convert('RGB').resize((image_content.shape[1], image_content.shape[0])))

    # Apply the mask
    for i in range(image_content.shape[0]):
        for j in range(image_content.shape[1]):
            if any(mask[i, j, :]) > 0.:
                styled_image[i, j, :] = image_content[i, j, :]

    if save:
        path_save = os.path.split(p_styled)[0]
        name_save = os.path.split(p_styled)[1]
        name_save = name_save.split('.')[0] + '_masked.' + name_save.split('.')[1]
        path_save = os.path.join(path_save, name_save)
        Image.fromarray(styled_image).convert('RGB').save(path_save)

    return save_image
