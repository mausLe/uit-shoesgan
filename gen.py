import os
import random
import pickle
import wget
import json
import gdown
import torch
import numpy as np
from torchvision import transforms, utils
from model import Generator

SEED = 2020
DEVICE = 'cuda'

RESOLUTION = 512
LEVELS = [4, 4, 8]

LATENT_SIZE = 512

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(SEED)

def load_model(model_path='model.pt'):
    if not os.path.exists(model_path):
        # model kaggle 512 transfer
        url_model = 'https://www.kaggleusercontent.com/kf/38674976/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..1Uz3xlS-BzYk04tqvyH3gg.s_ZyPiZ4gxRBPtCMfF8FW4bFGx89xRGJvyms-8RU4yCMJl-Vle4xIQ5mgwRUmROeK0cI-zFGINP9qtdRVXQcf-F4q33kTuLi7P7WH9SusqkjD4Uw6e_CJJM9KqEEUb0YWoOj-U3WRdAb2DfkHjMDf0TTo5o7jZkzLZ4LOK1ETYExL3gyjLMzcuzOyrpA0nq7hqmhBqHBeyii_xza-Urnfv8Vtiap_leCEO0I05KMbSRrKFBd5YLdIpmuILUvb7vfsT3wi3FPmx-BI5uYUujeVqLDY7ttwHLLAdmiHcjIjq-L5m76YblGlwOhA_7fMgcTm7xdcl-OTfuGLhr9EdcgXCirhE9Rnt6O0ynmyxeRCaw4avqrG-V1g6HXzuPuipk_ArQwe_JXCwkEDHo6j0cSxhztaXC-AsLj04Cpzb5n_v-iyxCwwi3YiHaWhBQrzeq0CQthjx1H5CrqPjyVMAgaq1bxptyqq_QS5dgGy9BBPXkAviV7IT722GH68sO8sPckxKuOB8NAZLdZc2Js8BC80sGDsMhgGKX3ZKPS7GRiWAXMoARIL_-MW_hBYf3fIQHDbIFnHAFWPkO_PKAzfV8j6D8cVmrWhvy5UxTfVJER3Dh1LSmqEWpFr22u8rV7ozkX1KT9b8-TuRbX8flyqPkzpA.tgEzI1oRwQpRmfIVzqDDYw/checkpoint/039390.pt'
        wget.download(url_model, out=model_path)
    checkpoint = torch.load(model_path)

    generator = Generator(RESOLUTION, LATENT_SIZE, 8).to(DEVICE)

    generator.load_state_dict(checkpoint['g_ema'])
    generator.eval()
    return generator

def load_latents(latents_path='latents512.pkl'):
    if not os.path.exists(latents_path):
        url_latents = 'https://drive.google.com/uc?id=1Oo_R2un4FO5srZpZowGbDvDzYdo5MaNq&export=download'
        gdown.download(url_latents, output=latents_path, quiet=False)
    with open(latents_path, 'rb') as f:
        latents = pickle.load(f)
    return latents

def load_labels(labels_path='labels.json'):
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    return labels

def load_predefined_styles(styles_path=os.path.join("styles-predefined", "styles-predefined.json")):
    with open(styles_path, 'r') as f:
        styles = json.load(f)
    return styles

MODEL = load_model()
LATENTS = load_latents()
LABELS = load_labels()
STYLES = load_predefined_styles()

@torch.no_grad()
def style_to_image(style1, style2, style3, model=MODEL):
    styles = [style1, style2, style3]

    for i, style in enumerate(styles):
        if len(style) == 0:
            random_latent = torch.randn(1, LATENT_SIZE).to(DEVICE)
            style = model.get_latent(random_latent)
            style = style.cpu().numpy()
        style = np.array(style).reshape(1, LATENT_SIZE)
        style = style.repeat(LEVELS[i], axis=0)
        styles[i] = style

    styles = np.r_[styles[0], styles[1], styles[2]]
    styles = torch.from_numpy(styles)
    styles = styles.unsqueeze(0)
    styles = styles.to(DEVICE)

    sample, _ = model([styles], input_is_latent=True)
    sample = sample.cpu()
    img = transforms.ToPILImage()(sample[0].clamp_(-1, 1).add_(1).div_(2 + 1e-5)).convert('RGB')

    return img

@torch.no_grad()
def gen_images(batch, model=MODEL):
    random_latents = torch.randn(batch, LATENT_SIZE).to(DEVICE)
    samples, latents = model([random_latents], return_latents=True)
    samples = samples.cpu()
    latents = latents.cpu()
    return samples, latents


@torch.no_grad()
def get_random_images(inputs=[-1, -1, 2], model=MODEL):
    results = []
    for x in inputs:
        if x == -1:
            latent_z = torch.randn(1, LATENT_SIZE).to(DEVICE)
            image, latent_w = model([latent_z], return_latents=True)
            latent_w = latent_w[:, 0, :]
        else:
            latent_w = LATENTS[LABELS[x]].reshape(1, LATENT_SIZE)
            latent_w = torch.from_numpy(latent_w).to(DEVICE)

            image, test = model([latent_w], input_is_latent=True, return_latents=True)
        
        image = transforms.ToPILImage()(image[0].cpu().clamp_(-1, 1).add_(1).div_(2 + 1e-5))
        latent_w = latent_w.cpu().numpy().reshape(-1,)
        item = [image, latent_w]
        results.append(item)
    return results

def get_style_from_label(labels):
    name = labels[0]
    results = LABELS[name]
    labels = random.sample(results, len(labels))
    labels = np.array(labels)
    latents = LATENTS[labels]
    return latents

@torch.no_grad()
def get_style_from_random(n_samples, model=MODEL):
    """input:
        - n_samples: number of styles
        output:
        - a numpy array of vector styles with random 
    """
    latent_z = torch.randn(n_samples, LATENT_SIZE).to(DEVICE)
    latent_w = model.get_latent(latent_z)
    return latent_w.cpu().numpy()

def get_style_from_index():
    latents = []
    styles = []
    levels = []
    for x in STYLES:
        latents.append(LATENTS[x["id"]])
        styles.append(x["style"])
        levels.append(x["level"])
    return latents, styles, levels

@torch.no_grad()
def get_images_from_styles(vectors, weight=0.8, model=MODEL, multi_output=False):
    """input:
        - style1, style2, style3 : vector with shape (320,)
        output:
        - image with size 512x512
    """
    style1 = np.array(vectors[0], dtype=np.float32)
    style2 = np.array(vectors[1], dtype=np.float32)
    style3 = np.array(vectors[2], dtype=np.float32)

    vectors = np.array(vectors, dtype=np.float32)
    styles = []
    for i, style in enumerate(vectors):
        style = torch.tensor(style, device=DEVICE)
        style = style.repeat(LEVELS[i], 1)
        styles.append(style)
    styles = torch.cat(styles, 0).unsqueeze(0)
    img, _ = model([styles], input_is_latent=True)
    img = img.cpu()
    img = transforms.ToPILImage()(img[0].clamp_(-1, 1).add_(1).div_(2 + 1e-5)).convert('RGB')
    return img

@torch.no_grad()
def get_images_from_styles_mixing(input_style, mix_style, weight=0.8, model=MODEL, level=1):
    style1 = np.array(input_style[0], dtype=np.float32)
    style2 = np.array(input_style[1], dtype=np.float32)
    style3 = np.array(input_style[2], dtype=np.float32)

    styles = []

    if level <= 3:
        new_style = (input_style[level-1] + weight * mix_style[level-1]) / (weight + 1)
        new_style = np.array(new_style, dtype=np.float32)
        if level == 1:
            style1 = new_style
        elif level == 2:
            style2 = new_style
        elif level == 3:
            style3 = new_style
    else:
        style1 = (input_style[0] + weight * mix_style[0]) / (weight + 1)
        style2 = (input_style[1] + weight * mix_style[1]) / (weight + 1)
    
    for i, style in enumerate([style1, style2, style3]):
        style = torch.tensor(style, device=DEVICE)
        style = style.repeat(LEVELS[i], 1)
        styles.append(style)
    styles = torch.cat(styles, 0).unsqueeze(0)
    img, _ = model([styles], input_is_latent=True)
    img = img.cpu()
    img = transforms.ToPILImage()(img[0].clamp_(-1, 1).add_(1).div_(2 + 1e-5)).convert('RGB')
    return img, [style1, style2, style3]

