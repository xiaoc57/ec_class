import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import itertools
from networks import Generator
import torch
import math
import matplotlib.pyplot as plt

def get_model(model='PGAN', dataset='celebAHQ-512', use_gpu=True):
    """Returns a pretrained GAN from (https://github.com/facebookresearch/pytorch_GAN_zoo).

    Args:
        model (str): Available values are "PGAN", "DCGAN".
        dataset (str: Available values are "celebAHQ-256", "celebAHQ-512', "DTD", "celeba".
            Ignored if model="DCGAN".
        use_gpu (bool): Whether to use gpu.
    """
    all_models = ['PGAN', 'DCGAN']
    if not model in all_models:
        raise KeyError(
            f"'model' should be in {all_models}."
        )

    pgan_datasets = ['celebAHQ-256', 'celebAHQ-512', 'DTD', 'celeba']
    if model == 'PGAN' and not dataset in pgan_datasets:
        raise KeyError(
            f"If model == 'PGAN', dataset should be in {pgan_datasets}"
        )

    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', model,
                           model_name=dataset, pretrained=True, useGPU=use_gpu)

    return model


def getmodel2():


    device = "cuda"
    num_test_samples = 16

    netG = Generator(1, 100, 32)
    dict_load = torch.load("generator_param.pkl")  # 先从文件中取出字典
    netG.load_state_dict(dict_load)  # 最后加载字典
    # torch.load("generator_param.pkl")
    netG = netG.to(device)
    return netG
    # z = torch.randn(num_test_samples, 100, 1, 1, device=device)
    # size_figure_grid = int(math.sqrt(num_test_samples))
    # title = None

    # if use_fixed:
    #     generated_fake_images = netG(fixed_noise)
    #     path += 'fixed_noise/'
    #     title = 'Fixed Noise'
    # else:

    # generated_fake_images = netG(z)
    # path = "./"
    # # title = 'Variable Noise'
    #
    # fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6, 6))
    # for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
    #     ax[i, j].get_xaxis().set_visible(False)
    #     ax[i, j].get_yaxis().set_visible(False)
    # for k in range(num_test_samples):
    #     i = k // 4
    #     j = k % 4
    #     ax[i, j].cla()
    #     ax[i, j].imshow(generated_fake_images[k].data.cpu().numpy().reshape(28, 28), cmap='Greys')
    # plt.show()
