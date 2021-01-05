import os
import random
import pickle
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
        #url_model = 'https://www.kaggleusercontent.com/kf/38674976/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..N9dJWnTai5h3BvGK3rot5w.gfyBV_NUdqH5so3h7HfHyZCAE3djnSX2Fw4IK9gwCdsmwpVeyP2fnKhpXJGJ3FNa1UU-H6qT6erLWPSBYR27IoMYIr-svPoXWtXyo8e_kuubaTKGOTVx9OkN4o9lK0vFKUSc6-CYKXMY7vtqGc6MMYyFdwAShAXjARx06t3fDhR74s8X8o7h7HkfXVmtmW08geKNrF1C8Vu55c917DaMKKhGXZbd93MTohaWddLMvtK40TTuX30pamSE7sAKldc22krR8hwu-hfTRrntUYke8Dn75zw2Fvs1g0vGnLjaJY-Nyyi9icyiURV4wofJG4qJiCaPZp5eO6zxY_asFQie9dvqB1VTtDafIZ_yFktETrA730xq1NleL-KcJS2G7-iqjdFO_H2duy626kbQMWdRe6htdOkbNlI5joQUxFNK3fKs5pr4vsP5S1L0PnghwKzqfq2CYf0LpPF6ee1oh6aLe4vfQz7uZCOA8R70uJ2q0e9K5UzWv2ddyCrkvtz4dj7K2t8BhuffWnpw7OEiOTKhx5twfCMyXJwE-Ds9KVFPUk9F7uU8AzzCymPc5KypLzoMqsPd0wHEz1gq_L6MjCJCDf-nTbUpTcW8atvzCatGSMfFYY286sFFTIB4RivFGbXJ.rUE2zW9l5lk-rwRFw5rCzw/checkpoint/039390.pt'
        #url_model = 'https://www.kaggleusercontent.com/kf/38674976/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..OJtjqngFwft4SmyXyia3-A.pS9DBCBXHMeF9rVs9d-bV61AQnn5GeWBjcY0K09bMJYYPdMfOkQphG4PWqmfjJ0ZNI2ButzTFIrWYZ9-9vqMibjcRAKD1t8Z_NY6NeSsbaviGtj6xAk6i2HHuaLeu0EEbTCnRM0gqQyn-LFJh5gNtQ0yNW7vP1gwG24uO1NQJHfMd6NinuVzR7NPurKnFlE7FsN1Lpl6RtFWjp_SMj0I9Ffkqc7j2L0i6AvSwU6bcJ3sq6eb0eFEmNqIN-Pet6diJvecbRXpLdazYUmUR9wTfXV_61JJjOGQvtxc4GNlzFvGV2onnG-EHJgaEw8dSeXEBg4EPl-xIPIvCiOGbyKpvKWIqi9bSr9E0YXJ_M9DDnP_AZ4P8Tung3slNyKPKfMvOoZhzASmC1S6HDV9I9VtCbnjFV6x8Et3B_K_4OuSR4AWbtn3mJJiEZqJ8ZHO_e3dIf1L8fdCGKb4mAId2uFf0JypXbvoxmOrFUeegUf4YTrTgMo-xgKwKSa_9eUYfImu9RcekUukQJTSK06CPUK26_Nt6_MZjkIx79I-aIfO_ucAT8moP9V3qlhLBwdW1XWHpR6GGbSzFcw-Ht-Y3ae6IWBzUFQQhoJSPSWl1O97J6laDkjE447X-PcuFvdo8kuM.7bHDTamk9Li0dx21JnpK8w/checkpoint/039390.pt'
        #wget.download(url_model, out=model_path)
        url_model = "https://drive.google.com/uc?id=1YS8SRLgphALD4DXBMPAtT0Ku_bCzggkd"
        gdown.download(url_model, output=model_path, quiet=False)

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
        style1 = np.array(style1, dtype=np.float32)
        style2 = np.array(style2, dtype=np.float32)
    
    for i, style in enumerate([style1, style2, style3]):
        style = torch.tensor(style, device=DEVICE)
        style = style.repeat(LEVELS[i], 1)
        styles.append(style)
    styles = torch.cat(styles, 0).unsqueeze(0)
    img, _ = model([styles], input_is_latent=True)
    img = img.cpu()
    img = transforms.ToPILImage()(img[0].clamp_(-1, 1).add_(1).div_(2 + 1e-5)).convert('RGB')
    return img, [style1, style2, style3]

