import os
import sys
import math
import random
import streamlit as st

import numpy as np
import pandas as pd
import scipy.io
import torch
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from load_model import load_model_from_opts
from dataset import ImageDataset

# Extract feature from a trained model.

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    inv_idx = inv_idx.to(img.device)
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_features(model, dataloader, device, batchsize):
    img_count = 0
    dummy = next(iter(dataloader))[0].to(device)
    output = model(dummy)
    feature_dim = output.shape[1]
    labels = []

    for idx, data in enumerate(tqdm.tqdm(dataloader)):
        X, y = data
        n, c, h, w = X.size()
        img_count += n
        ff = torch.FloatTensor(n, feature_dim).zero_().to(device)

        for lab in y:
            labels.append(lab)

        for i in range(2):
            if(i == 1):
                X = fliplr(X)
            input_X = Variable(X.to(device))
            outputs = model(input_X)
            ff += outputs

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        if idx == 0:
            features = torch.FloatTensor(len(dataloader.dataset), ff.shape[1])

        start = idx * batchsize
        end = min((idx + 1) * batchsize, len(dataloader.dataset))
        features[start:end, :] = ff
    return features, labels


def extract_feature(model, img, device):
    if len(img.shape) == 3:
        img = torch.unsqueeze(img, 0)
    img = img.to(device)
    feature = model(img).reshape(-1)

    img = fliplr(img)
    flipped_feature = model(img).reshape(-1)
    feature += flipped_feature

    fnorm = torch.norm(feature, p=2)
    return feature.div(fnorm)


def get_scores(query_feature, gallery_features):
    query = query_feature.view(-1, 1)
    score = torch.mm(gallery_features, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    return score


def show_query_result(query_img, gallery_imgs, query_label, gallery_labels, imgs_per_row):
    n_rows = math.ceil((1 + len(gallery_imgs)) / imgs_per_row)
    fig, axes = plt.subplots(n_rows, imgs_per_row, figsize=(12, 15))

    query_trans = transforms.Pad(4, 0)
    good_trans = transforms.Pad(4, (0, 255, 0))
    bad_trans = transforms.Pad(4, (255, 0, 0))

    for idx, img in enumerate([query_img] + gallery_imgs):
        img = img.resize((128, 128))
        if idx == 0:
            img = query_trans(img)
        elif query_label == gallery_labels[idx - 1]:
            img = good_trans(img)
        else:
            img = bad_trans(img)

        ax = axes.flat[idx]
        ax.imshow(img)

    for i in range(len(axes.flat)):
        ax = axes.flat[i]
        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.set_yticks([])
        ax.set_yticks([], minor=True)
        ax.axis("off")

    plt.tight_layout()
    return fig

def visualize(data_dir, query_csv_path, gallery_csv_path, model_opts, checkpoint, batchsize, input_size, num_images, imgs_per_row, use_saved_mat, curr_idx):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    h, w = input_size, input_size

    # load data
    data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    query_df = pd.read_csv(query_csv_path)
    gallery_df = pd.read_csv(gallery_csv_path)
    classes = list(pd.concat([query_df["id"], gallery_df["id"]]).unique())
    use_cam = "cam" in query_df and "cam" in gallery_df

    image_datasets = {
        "query": ImageDataset(data_dir, query_df, "id", classes, transform=data_transforms),
        "gallery": ImageDataset(data_dir, gallery_df, "id", classes, transform=data_transforms),
    }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                                shuffle=False, num_workers=2) for x in ['gallery', 'query']}
    
    # run queries
    if use_saved_mat:
        saved_res = scipy.io.loadmat("pytorch_result.mat")
        gallery_features = torch.Tensor(saved_res["gallery_f"])
        gallery_labels = saved_res["gallery_label"].reshape(-1)
        query_features = torch.Tensor(saved_res["query_f"])
        query_labels = saved_res["query_label"].reshape(-1)
    else:
        model = load_model_from_opts(
            model_opts, checkpoint, remove_classifier=True)
        model.eval()
        model.to(device)

        print("Computing gallery features ...")

        with torch.no_grad():
            gallery_features, gallery_labels = extract_features(model, dataloaders["gallery"], device, batchsize)
            gallery_labels = np.array(gallery_labels)

    dataset = image_datasets["query"]
    queries = list(range(len(dataset)))
    # random.shuffle(queries)
    queries_len = len(queries)

    if use_saved_mat:
        q_feature = query_features[curr_idx]
        y = query_labels[curr_idx]
    else:
        X, y = dataset[curr_idx]
        with torch.no_grad():
            q_feature = extract_feature(model, X, device).cpu()

    if use_cam:
        curr_cam = query_df["cam"].iloc[curr_idx]
        good_gallery_idx = torch.tensor(gallery_df["cam"] != curr_cam).type(torch.bool)
        gallery_orig_idx = np.where(good_gallery_idx)[0]
        gal_features = gallery_features[good_gallery_idx]
    else:
        gallery_orig_idx = np.arange(len(gallery_df))
        gal_features = gallery_features
    gallery_scores = get_scores(q_feature, gal_features)
    idx = np.argsort(gallery_scores)[::-1]

    if use_cam:
        g_labels = gallery_labels[gallery_orig_idx][idx]
    else:
        g_labels = gallery_labels[idx]

    q_img = dataset.get_image(curr_idx)
    g_imgs = [image_datasets["gallery"].get_image(gallery_orig_idx[i])
              for i in idx[:num_images]]
    fig = show_query_result(q_img, g_imgs, y, g_labels, imgs_per_row)

    return fig, queries_len
