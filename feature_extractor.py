import random

import ray
import os
import numpy as np
from PIL import Image
import torch
import imghdr
import argparse
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, ColorJitter, \
    CenterCrop
import clip
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from tqdm import tqdm
import pdb
import json
import pickle as pkl
import pandas as pd
import pytorch_lightning as pl
import re
import torch.nn as nn
from torchvision.utils import save_image
from math import ceil, floor
from collections import OrderedDict
from utils import smart_load_image, smart_open, smart_listdir, smart_pkl_dump, smart_pkl_load, expand_path, expand_path


def imagenet_transform():
    """
    Transformation function
    :return: transform object
    """
    def _convert_image_to_rgb(image):
        # in case there is an Alpha channel
        return image.convert("RGB")

    transform = Compose([
        _convert_image_to_rgb,
        ToTensor(),
        # data from https://github.com/openai/CLIP/blob/main/notebooks/Interacting_with_CLIP.ipynb
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    return transform


def get_all_img_path(dataset_dir):
    """
    Recursively get all image path inside a dataset directory
    """
    all_img_path_list = []
    # for filename in glob.iglob(dataset_dir + '**/*.*', recursive=True):
    for filename in tqdm(smart_listdir(dataset_dir)):
        if filename.lower().endswith("jpg") or filename.lower().endswith("png"):
            file_path = os.path.join(dataset_dir, filename)
            all_img_path_list.append(file_path)

    return all_img_path_list


def get_all_scene_path(dataset_dir):
    """
    Recursively get all scene path inside a dataset directory
    """
    all_scene_path_list = []
    # for filename in glob.iglob(dataset_dir + '**/*.*', recursive=True):
    for filename in tqdm(smart_listdir(dataset_dir)):
        if filename.lower().endswith("json"):
            file_path = os.path.join(dataset_dir, filename)
            all_scene_path_list.append(file_path)

    return all_scene_path_list


def assign_img_per_gpu(num_runner, total_img_number):
    """
    Assign number of images to gpus
    :param num_runner: how many runner (gpu)
    :param total_img_number: image amount
    :return: resource assignment (how many tasks per gpu)
    """
    base_img_per_runner = total_img_number // num_runner
    resource_assignment = [base_img_per_runner] * num_runner
    residual = total_img_number - base_img_per_runner * num_runner
    if residual == 0:
        return resource_assignment
    else:
        i = 0
        while i < residual:
            resource_assignment[i] += 1
            i += 1
        return resource_assignment


def get_img_path_assignment(all_imgs_paths, resource_assignment):
    """
    Assign image paths to gpus
    all_imgs_paths: list of str, contains abs path of imgs
    resource_assignment: list of int, contains image assignment per runner
    return: list of list, replace int in resource_assignment to actual img path
    """
    assert sum(resource_assignment) == len(all_imgs_paths)
    rl = []
    i = 0
    for img_per_runner in resource_assignment:
        temp_path_list = all_imgs_paths[i:i + img_per_runner]
        rl.append(temp_path_list)
        i += img_per_runner
    return rl


class GFEDataset(Dataset):
    """
    Patch-Grid Feature Extractor Dataset
    """
    def __init__(self, img_path_list, clip_input_size):
        self.img_path_list = img_path_list
        self.clip_input_size = clip_input_size

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        Image.MAX_IMAGE_PIXELS = None
        img_path = self.img_path_list[index]
        # bgr image in numpy format
        image_np = smart_load_image(img_path)[:, :, ::-1]
        image = Image.fromarray(np.uint8(image_np))
        transform = imagenet_transform()
        image_after_transform = transform(image)
        full_img_grid = rearrange(image_after_transform, 'c (h s1) (w s2) -> (h w) c s1 s2', s1=self.clip_input_size[0], s2=self.clip_input_size[1])
        full_img_grid = F.interpolate(full_img_grid, self.clip_input_size, mode="bilinear")
        return full_img_grid, img_path


class CCFEDataset(Dataset):
    """
    Patch-CC Feature Extractor Dataset
    """
    def __init__(self, img_path_list, clip_input_size, c_denom):
        self.img_path_list = img_path_list
        self.clip_input_size = clip_input_size
        self.c_denom = c_denom

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        Image.MAX_IMAGE_PIXELS = None
        img_path = self.img_path_list[index]
        # bgr image in numpy format
        image_np = smart_load_image(img_path)[:, :, ::-1]
        image = Image.fromarray(np.uint8(image_np))
        transform = imagenet_transform()
        image_after_transform = transform(image).unsqueeze(0)
        patch_img = img_2patch(image_after_transform, c_denom=self.c_denom, final_size=self.clip_input_size).squeeze(0)
        return patch_img, img_path


class OFEDataset(Dataset):
    """
    Patch-Object Feature Extractor Dataset
    """
    def __init__(self, img_path_list, scene_path_list, clip_input_size, crop_scale):
        self.img_path_list = img_path_list
        self.scene_path_list = scene_path_list
        self.clip_input_size = clip_input_size
        self.crop_scale = crop_scale

    def __len__(self):
        assert len(self.img_path_list) == len(self.scene_path_list)
        return len(self.img_path_list)

    def __getitem__(self, index):
        Image.MAX_IMAGE_PIXELS = None
        img_path = self.img_path_list[index]
        scene_path = self.scene_path_list[index]
        # bgr image in numpy format
        image_np = smart_load_image(img_path)[:, :, ::-1]
        image = Image.fromarray(np.uint8(image_np))
        scene = json.load(smart_open(scene_path, "r"))["objects"]
        bbox_list = []
        bbox_label_list = []
        for object in scene:
            x, y, width, height, shape, material = object["x"], object["y"], object["width"], object["height"], object["shape"], object["material"],
            label = shape
            bbox_list.append([x, y, width, height])
            bbox_label_list.append(label)
        crop_patch, cropped_img_list = img_scene_2crop(image, bbox_list, crop_scale=self.crop_scale, final_size=self.clip_input_size)

        return crop_patch, img_path, bbox_label_list, cropped_img_list


class TFEDataset(Dataset):
    """
    Text Feature Extractor Dataset
    """
    def __init__(self, caption_list, clip_encoder_name, text_feat_dir, device, cls_names=None):
        self.caption_list = caption_list
        self.cls_names = cls_names
        self.clip_encoder_name = clip_encoder_name
        self.text_feat_dir = text_feat_dir
        self.device = device
        if self.clip_encoder_name == "clip-openai-14":
            self.tokenizer = clip.tokenize
        elif self.clip_encoder_name == "slip" or self.clip_encoder_name == "clip":
            from model.slip.tokenizer import SimpleTokenizer
            self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_list)

    def __getitem__(self, index):
        caption = self.caption_list[index]
        if self.clip_encoder_name == "clip-openai-14":
            text_token = self.tokenizer(caption, truncate=True).to(self.device)
        else:
            text_token = self.tokenizer(caption).to(self.device)
        if self.cls_names is not None:
            return text_token, self.cls_names[index]
        else:
            return text_token, caption


@ray.remote
def extract_text_feature(c_name_list, text_feat_save_dir, config, partition_dataset_i2c_dict, partition_dataset_c2i_dict, text_templates=None):
    """

    :param c_name_list: list of cls/caption name
    :param text_feat_save_dir: text feature save dir
    :param config: inference configuration
    :param text_templates: templates for prompt
    :return: None
    """

    pl.seed_everything(0)
    model, img_input_size = get_model(config["model_name"], config["model_ckpt_path"], config["device"])

    if config["text_type"] == "cls":
        prompts = np.array([imagenet_template.format(cls.replace("-", " ").replace("_", " ").replace("\\", " ")) for imagenet_template in text_templates for cls in c_name_list])
        captions = prompts.reshape(-1, len(c_name_list)).T
        dataset = TFEDataset(captions, config["model_name"], text_feat_save_dir, config["device"], c_name_list)

    elif config["text_type"] == "caption":
        caption_img_dict = partition_dataset_c2i_dict
        captions = list(caption_img_dict.keys())
        repeat_caption_counter = 0
        for caption in caption_img_dict:
            img = caption_img_dict[caption]
            if len(img) > 1:
                print(caption, img)
                repeat_caption_counter += 1
        print("repeat caption counter: {}".format(repeat_caption_counter))
        dataset = TFEDataset(captions, config["model_name"], text_feat_save_dir, config["device"])

    else:
        print("text type has to be one of cls or caption")
        exit()

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    text_feature_all = []
    caption_list = []
    for index, batch in enumerate(dataloader):
        # (1, 77), (1, )
        text_token_batch, caption = batch
        caption = caption[0]
        # pdb.set_trace()
        if config["text_type"] == "cls":
            text_token_batch = text_token_batch[0]
            with torch.no_grad():
                caption_text_feat = model.encode_text(text_token_batch).unsqueeze(0).to(torch.float32).cpu().detach()
        elif config["text_type"] == "caption":
            if config["model_name"] == "clip-openai-14":
                text_token_batch = text_token_batch[0]
            with torch.no_grad():
                caption_text_feat = model.encode_text(text_token_batch).to(torch.float32).cpu().detach()
        caption_list.append(caption)
        text_feature_all.append(caption_text_feat)
        print(caption, caption_text_feat.shape)
    text_feature_all = torch.cat(text_feature_all).numpy()
    print(text_feature_all.shape)
    text_feat_path = os.path.join(text_feat_save_dir, "text_feat_{}_{}.pkl".format(config["model_name"], config["text_type"]))
    smart_pkl_dump(text_feat_path, text_feature_all)
    print("dump {} to {}".format(text_feature_all.shape, text_feat_path))
    caption_path = os.path.join(text_feat_save_dir, "captions_{}_{}.pkl".format(config["model_name"], config["text_type"]))
    smart_pkl_dump(caption_path, caption_list)
    print("dump {} to {}".format(len(caption_list), caption_path))


@ray.remote
def extract_grid_feature(img_path_list, img_feature_save_dir, config):
    """
    Extract image features with patch-grid scheme
    :param img_path_list: list of img abs path
    :param img_feature_save_dir: feature save dir
    :param config: inference configuration
    :return: None
    """
    pl.seed_everything(0)
    model, img_input_size = get_model(config["model_name"], config["model_ckpt_path"], config["device"])

    dataset = GFEDataset(img_path_list, img_input_size)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=0)

    print("device: {}".format(config["device"]))

    for index, batch in tqdm(enumerate(dataloader)):
        x, paths = batch
        x = x.to(config["device"])
        b = x.shape[0]
        patch_container = rearrange(x, 'b p c h w -> (b p) c h w')
        grid_feat = model.encode_image(patch_container).to(torch.float32).cpu().detach().numpy()
        grid_feat = rearrange(grid_feat, '(b p) f -> b p f', b=b)
        i = 0
        for path in paths:
            img_name = path.split("/")[-1].split(".")[0]
            grid_image_feature_name = img_name + "_image_grid_{}".format(config["model_name"]).replace("/", "-")
            grid_image_feature_path = os.path.join(img_feature_save_dir, grid_image_feature_name)
            smart_pkl_dump(grid_image_feature_path, grid_feat[i])
            # tmp = pkl_load(grid_image_feature_path)
            i += 1


@ray.remote
def extract_cc_feature(img_path_list, img_feature_save_dir, config, k):
    """
    Extract image features with patch-cc scheme
    :param img_path_list: list of img abs path
    :param img_feature_save_dir: feature save dir
    :param config inference configuration
    :param k: k, for cc@k
    :return: None
    """
    pl.seed_everything(0)
    model, img_input_size = get_model(config["model_name"], config["model_ckpt_path"], config["device"])
    dataset = CCFEDataset(img_path_list, img_input_size, k)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=0)

    print("device: {}".format(config["device"]))

    for index, batch in tqdm(enumerate(dataloader)):
        x, paths = batch

        x = x.to(config["device"])
        b = x.shape[0]

        patch_container = rearrange(x, 'b p c h w -> (b p) c h w')
        with torch.no_grad():
            clip_feat = model.encode_image(patch_container).to(torch.float32).cpu().detach().numpy()
        cc_feat = rearrange(clip_feat, '(b p) f -> b p f', b=b)

        for i, path in enumerate(paths):
            img_name = path.split("/")[-1].split(".")[0]
            patches_image_feature_name = img_name + ".pkl"
            cc_image_feature_path = os.path.join(img_feature_save_dir, patches_image_feature_name)
            smart_pkl_dump(cc_image_feature_path, cc_feat[i])

        print("patch tensor shape: {}".format(cc_feat.shape))


@ray.remote
def extract_obj_feature(img_path_list, scene_path_list, img_feature_save_dir, config, crop_scale):
    """
    Extract image features with patch-obj scheme
    :param img_path_list: list of img abs path
    :param scene_path_list: list of scene file abs path
    :param img_feature_save_dir: img feature save dir
    :param config: inference configuration
    :param crop_scale: scale to enlarge the bbox for cropping image
    :return: None
    """

    pl.seed_everything(0)
    model, img_input_size = get_model(config["model_name"], config["model_ckpt_path"], config["device"])

    def collate_fn(batch):
        cropped_feature_batch = []
        cropped_img_batch = []
        img_path_batch = []
        label_batch = []
        for sample in batch:
            cropped_feature_batch.append(sample[0].unsqueeze(0))
            img_path_batch.append(sample[1])
            label_batch.append(sample[2])
            cropped_img_batch.append(sample[3])
        cropped_feature_batch = torch.cat(cropped_feature_batch)
        return cropped_feature_batch, img_path_batch, label_batch, cropped_img_batch

    dataset = OFEDataset(img_path_list, scene_path_list, img_input_size, crop_scale)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=0, collate_fn=collate_fn)

    for index, batch in tqdm(enumerate(dataloader)):
        # crop_patch, img_path, bbox_label_list, cropped_img_list
        crop_patch, img_path, bbox_label_list, cropped_img_list = batch

        crop_patch = crop_patch.to(config["device"])
        b = crop_patch.shape[0]

        # only support bs = 1 since each image have different number of cropped patches
        assert b == 1

        crop_patch_container = rearrange(crop_patch, 'b p c h w -> (b p) c h w')

        clip_feat = model.encode_image(crop_patch_container).to(torch.float32).cpu().detach().numpy()

        obj_feat = rearrange(clip_feat, '(b p) f -> b p f', b=b)

        i = 0
        for path in img_path:
            crop_img_dir = "cropped_images"
            crop_label_dir = "cropped_labels"
            img_name = path.split("/")[-1].split(".")[0]
            patches_image_feature_name = img_name + "_image_crop-{}_{}".format(crop_scale, config["model_name"]).replace("/", "-")
            obj_image_feature_path = os.path.join(img_feature_save_dir, patches_image_feature_name)
            os.makedirs(os.path.join(img_feature_save_dir, crop_img_dir), exist_ok=True)
            os.makedirs(os.path.join(img_feature_save_dir, crop_label_dir), exist_ok=True)

            # cropped_image
            smart_pkl_dump(obj_image_feature_path, obj_feat[i])
            i += 1


def extract(args):
    """
    Main function for feature extraction
    :param args:
    :return: None
    """
    from datetime import datetime
    toc = datetime.now()

    pl.seed_everything(0)

    if args.ray_mode == "debug_cpu":
        ray.init(local_mode=True)
        args.num_cpu = 0
        args.num_gpu = 0
    elif args.ray_mode == "debug_gpu":
        args.num_cpu = 0
        args.num_gpu = 1
        ray.init(local_mode=True)
    elif args.ray_mode == "local_run":
        args.num_cpu = 8
        args.num_gpu = 1
        ray.init(local_mode=True)
    else:
        ray.init("auto")

    print('runner, cpu, gpu: {}, {}, {}'.format(args.num_runner, args.num_cpu, args.num_gpu))
    ray_resources = ray.available_resources()
    print('available devices: {}'.format(ray_resources))

    modality = args.extract_type
    device = "cuda" if args.num_gpu > 0 else "cpu"

    config = dict()
    config["batch_size"] = args.batch_size
    config["dataset_img_dir"] = args.original_data_partition_img_dir
    config["dataset_scene_dir"] = args.original_data_partition_scene_dir
    config["partition_data_dir"] = args.original_data_partition_dir
    config["model_ckpt_path"] = args.model_ckpt_path
    config["model_name"] = args.model_name
    config["text_type"] = args.text_type
    config["use_oss"] = args.use_oss
    config["partition"] = args.partition
    config["device"] = device

    print("model: {}, modality: {}, device: {}, batch size: {}".format(config["model_name"], modality, config["device"], config["batch_size"]))

    if modality == "text":
        if config["text_type"] == "cls":
            dataset_template = json.load(open(args.template_dict_path))[args.dataset_name.lower()]
            i2c_dict_name = "img_cls_dict_{}.pkl".format(config["partition"])
            c2i_dict_name = "cls_img_dict_{}.pkl".format(config["partition"])
        elif config["text_type"] == "caption":
            dataset_template = None
            i2c_dict_name = "img_caption_dict_{}.pkl".format(config["partition"])
            c2i_dict_name = "caption_img_dict_{}.pkl".format(config["partition"])
        else:
            exit()

        partition_dataset_i2c_dict_path = os.path.join(args.dataset_dir, "mappings", i2c_dict_name)
        partition_dataset_c2i_dict_path = os.path.join(args.dataset_dir, "mappings", c2i_dict_name)

        try:
            partition_dataset_i2c_dict = smart_pkl_load(partition_dataset_i2c_dict_path)
            partition_dataset_c2i_dict = smart_pkl_load(partition_dataset_c2i_dict_path)
            partition_c_names = list(partition_dataset_c2i_dict.keys())
            print("found partition_dataset_i2c_dict from {}".format(partition_dataset_i2c_dict_path))
            print("found partition_dataset_c2i_dict from {}".format(partition_dataset_c2i_dict_path))

        except:
            print("cannot found partition_dataset_i2c_dict from {} ".format(partition_dataset_i2c_dict_path))
            dataset_i2c_dict_path = os.path.join(args.dataset_dir, "mappings", "img_{}_dict_overall.pkl".format(config["text_type"]))
            print("loaded overall dict of {} from {}".format(args.dataset_name, dataset_i2c_dict_path))
            dataset_i2c_dict = smart_pkl_load(dataset_i2c_dict_path)
            partition_all_img_filenames = smart_listdir(config["dataset_img_dir"])
            partition_all_img_ids = [partition_img_filename.split('.')[0] for partition_img_filename in partition_all_img_filenames]
            partition_dataset_i2c_dict = dict()
            partition_dataset_c2i_dict = dict()
            for partition_img_id in partition_all_img_ids:
                if len(partition_img_id) == 1:
                    print("partition_img_id has len 1, check: {}".format(partition_img_id))
                if partition_img_id in dataset_i2c_dict:
                    partition_dataset_i2c_dict[partition_img_id] = dataset_i2c_dict[partition_img_id]
            for img_name in partition_dataset_i2c_dict:
                if len(img_name) == 1:
                    print("img_name has len 1, check: {}".format(img_name))
                captions = partition_dataset_i2c_dict[img_name]
                for caption in captions:
                    if len(caption) == 1:
                        pdb.set_trace()
                        print("caption has len 1, check: {}".format(caption))
                        exit()
                    if caption not in partition_dataset_c2i_dict:
                        partition_dataset_c2i_dict[caption] = [img_name]
                    else:
                        if not config["text_type"] == "cls":
                            print("{} belongs to multiple img".format(caption))
                        partition_dataset_c2i_dict[caption].append(img_name)

            smart_pkl_dump(partition_dataset_i2c_dict_path, partition_dataset_i2c_dict)
            print("dumped partition i2c_dict ({}) to {}".format(len(partition_dataset_i2c_dict), partition_dataset_i2c_dict_path))

            smart_pkl_dump(partition_dataset_c2i_dict_path, partition_dataset_c2i_dict)
            print("dumped partition c2i_dict ({}) to {}".format(len(partition_dataset_c2i_dict), partition_dataset_c2i_dict_path))
            partition_c_names = list(partition_dataset_c2i_dict.keys())

        partition_c_names.sort()

        print("total number of cls/caption for text: {}".format(len(partition_c_names)))
        resource_assignment = assign_img_per_gpu(args.num_runner, len(partition_c_names))
        print("resource assignment for {} runners: {}".format(args.num_runner, resource_assignment))
        c_assignments = get_img_path_assignment(partition_c_names, resource_assignment)
        text_feature_save_dir = os.path.join(args.all_feat_dir, config["model_name"], "text_feat", config["partition"])
        os.makedirs(text_feature_save_dir, exist_ok=True)

        result_status = []
        j = 0
        for c in c_assignments:
            status = extract_text_feature.remote(c, text_feature_save_dir, config, partition_dataset_i2c_dict, partition_dataset_c2i_dict, dataset_template)
            result_status.append(status)
            print("runner: {}".format(j))
            j += 1
        ray.get(result_status)
        tic = datetime.now()
        print(tic - toc)

    elif modality == "image-cc":

        all_imgs_paths = get_all_img_path(args.original_data_partition_img_dir)
        print("total number of images in directory {}: {}".format(config["dataset_img_dir"], len(all_imgs_paths)))
        print("non repeat imgs: {}".format(len(set(all_imgs_paths))))
        resource_assignment = assign_img_per_gpu(args.num_runner, len(all_imgs_paths))
        print("resource assignment for {} runners: {}".format(args.num_runner, resource_assignment))
        img_path_assignments = get_img_path_assignment(all_imgs_paths, resource_assignment)

        # cc@2~15
        # k_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        # k_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        # k_list = [12, 13, 14, 15]
        k_list = [10]
        print("extract patch feature with patch factor(s) {}".format(k_list))

        for k in k_list:
            print("patch_factor: {}".format(k))
            img_feature_save_dir = os.path.join(args.all_feat_dir, args.model_name, "patch_feat", "cc_{}".format(k), args.partition)
            os.makedirs(img_feature_save_dir, exist_ok=True)
            print("image patch feature save dir: {}".format(img_feature_save_dir))
            result_status = []
            j = 0
            for img_path_list in img_path_assignments:
                status = extract_cc_feature.options(num_cpus=args.num_cpu, num_gpus=args.num_gpu).remote(
                    img_path_list, img_feature_save_dir, config, k
                )
                result_status.append(status)
                print("runner: {}".format(j))
                j += 1
            ray.get(result_status)

            tic = datetime.now()
            print(tic - toc)

    elif modality == "image-full":
        k = 10
        source_dir = os.path.join(args.all_feat_dir, args.model_name, "patch_feat", "cc_{}".format(k), args.partition)
        target_dir = os.path.join(args.all_feat_dir, args.model_name, "vanilla_feat", args.partition)
        get_vanilla_feat_from_patch_feat(source_dir, target_dir)

    elif modality == "image-grid":
        all_imgs_paths = get_all_img_path(args.dataset_img_dir)
        print("total number of images in directory {}: {}".format(args.dataset_img_dir, len(all_imgs_paths)))
        print("non repeat imgs: {}".format(len(set(all_imgs_paths))))
        resource_assignment = assign_img_per_gpu(args.num_runner, len(all_imgs_paths))
        print("resource assignment for {} runners: {}".format(args.num_runner, resource_assignment))
        img_path_assignments = get_img_path_assignment(all_imgs_paths, resource_assignment)

        img_feature_save_dir = os.path.join(args.partition_data_dir, "image_feat", "grid")
        # os.makedirs(img_feature_save_dir, exist_ok=True)
        print("image feature save dir: {}".format(img_feature_save_dir))
        result_status = []
        j = 0
        for img_path_list in img_path_assignments:
            status = extract_grid_feature.options(num_cpus=args.num_cpu, num_gpus=args.num_gpu).remote(
                img_path_list, img_feature_save_dir, config
            )
            result_status.append(status)
            print("runner: {}".format(j))
            j += 1
        ray.get(result_status)

        tic = datetime.now()
        print(tic - toc)

    elif modality == "image-obj":
        all_imgs_paths = get_all_img_path(args.dataset_img_dir)
        all_scenes_paths = get_all_scene_path(args.dataset_scene_dir)
        all_imgs_paths.sort()
        all_scenes_paths.sort()
        assert len(all_imgs_paths) == len(all_scenes_paths)
        print("total number of images in directory {}: {}".format(args.dataset_img_dir, len(all_imgs_paths)))
        print("non repeat imgs: {}".format(len(set(all_imgs_paths))))
        resource_assignment = assign_img_per_gpu(args.num_runner, len(all_imgs_paths))
        print("resource assignment for {} runners: {}".format(args.num_runner, resource_assignment))
        img_path_assignments = get_img_path_assignment(all_imgs_paths, resource_assignment)
        scene_path_assignments = get_img_path_assignment(all_scenes_paths, resource_assignment)

        # crop scale @ 1~4
        # crop_scale_list = [1, 2, 3, 4]
        crop_scale_list = [1]
        for crop_scale in crop_scale_list:
            print("crop scale: {}".format(crop_scale))
            img_feature_save_dir = os.path.join(args.partition_data_dir, "image_feat", "crop_scale_{}".format(crop_scale))
            # os.makedirs(img_feature_save_dir, exist_ok=True)
            print("image crop feature save dir: {}".format(img_feature_save_dir))
            result_status = []
            j = 0
            while j < len(img_path_assignments):
                img_path_list = img_path_assignments[j]
                scene_path_list = scene_path_assignments[j]
                assert len(img_path_list) == len(scene_path_list)
                status = extract_obj_feature.options(num_cpus=args.num_cpu, num_gpus=args.num_gpu).remote(img_path_list, scene_path_list, img_feature_save_dir, config, crop_scale)
                result_status.append(status)
                print("runner: {}".format(j))
                j += 1
            ray.get(result_status)

            tic = datetime.now()
            print(tic - toc)


def img_scene_2crop(img, bbox_list, crop_scale, final_size,  concat_last=True):
    """
    Get cropped feature and img from scene files
    :param img: image file
    :param bbox_list: all bbox (x, y, w, h) in this image
    :param crop_scale: crop scale, default to be 1 or 3
    :param final_size: clip input size
    :param concat_last: present in list or torch tensor
    :return: image feature in patch-obj scheme, cropped_img_list (actual img crops)
    """

    def get_image_crop(img, bbox, box_scale_ratio):
        """
        Gets image crop.
        ref: https://github.com/google-research/meta-dataset/blob/ca81edbf5093ec5ea1a1f5a4b31ec4078825f44b/meta_dataset/
        dataset_conversion/dataset_to_records.py#L1429
        """

        def scale_box(bbox, scale_ratio):
            x, y, w, h = bbox
            x = x - 0.5 * w * (scale_ratio - 1.0)
            y = y - 0.5 * h * (scale_ratio - 1.0)
            w = w * scale_ratio
            h = h * scale_ratio
            return [x, y, w, h]

        image_w, image_h = img.width, img.height,
        x, y, w, h = scale_box(bbox, box_scale_ratio)
        # Convert half-integer to full-integer representation.
        # The Python Imaging Library uses a Cartesian pixel coordinate system,
        # with (0,0) in the upper left corner. Note that the coordinates refer
        # to the implied pixel corners; the centre of a pixel addressed as
        # (0, 0) actually lies at (0.5, 0.5). Since COCO uses the later
        # convention and we use PIL to crop the image, we need to convert from
        # half-integer to full-integer representation.
        xmin = max(int(round(x - 0.5)), 0)
        ymin = max(int(round(y - 0.5)), 0)
        xmax = min(int(round(x + w - 0.5)) + 1, image_w)
        ymax = min(int(round(y + h - 0.5)) + 1, image_h)
        image_crop = img.crop((xmin, ymin, xmax, ymax))
        crop_width, crop_height = image_crop.size
        if crop_width <= 0 or crop_height <= 0:
            raise ValueError('crops are not valid.')

        return image_crop

    crop_container = []
    cropped_img_list = []
    for bbox in bbox_list:
        img_crop = get_image_crop(img, bbox, crop_scale)
        transform = imagenet_transform()
        image_after_transform = transform(img_crop).unsqueeze(0)
        img_crop_resize = F.interpolate(image_after_transform, final_size, mode="bilinear")
        crop_container.append(img_crop_resize)
        cropped_img_list.append(img_crop)
    if concat_last:
        crop_container = torch.cat(crop_container)

    return crop_container, cropped_img_list


def img_2patch(img, c_denom=10, final_size=(224, 224), concat_last=True):
    """
    Get image patches with patch-cc scheme
    (bs, 3, h, w) -> (bs, p, 3, h_p, w_p)
    :param img: img torch tensor
    :return: torch tensor with patch-cc scheme
    """
    b, ch, h, w = img.shape
    img_224 = F.interpolate(img, final_size, mode="bilinear").unsqueeze(1)
    if h % c_denom != 0:
        resize_h = (h // c_denom + 1) * c_denom
    else:
        resize_h = h

    if w % c_denom != 0:
        resize_w = (w // c_denom + 1) * c_denom
    else:
        resize_w = w
    img_resize = F.interpolate(img, (resize_h, resize_w), mode="bilinear")

    patch_container = []
    patch_container.append(img_224)
    n = 2
    a_h = resize_h
    a_w = resize_w
    c_h = a_h // c_denom
    c_w = a_w // c_denom
    kernel_h = a_h
    kernel_w = a_w
    patch_size_list = [1]

    # print("h w c_h c_w: {} {} {} {}".format(h, w, c_h, c_w))
    # print("resize h resize w: {} {}".format(resize_h, resize_w))

    while kernel_h - 2 * c_h >= 0 and kernel_w - 2 * c_w + 1 >= 0:
        kernel_h = (a_h - (n - 1) * c_h)
        kernel_w = (a_w - (n - 1) * c_w)

        stride_h = floor(kernel_h - (kernel_h - c_h) / c_h)
        stride_w = floor(kernel_w - (kernel_w - c_w) / c_w)

        img_patches = return_sliding_windows(img_resize, kernel_h, kernel_w, stride_h, stride_w)
        img_patches = rearrange(img_patches, 'b p (c h w) -> (b p) c h w', c=ch, h=kernel_h, w=kernel_w)
        img_patches_final_size = F.interpolate(img_patches, final_size, mode="bilinear")
        img_patches_final_size = rearrange(img_patches_final_size, '(b p) c h w -> b p c h w', b=b)
        patch_container.append(img_patches_final_size)
        # print("level: {}, kernel h: {}, kernel w: {}, stride h: {}, stride w: {}".format(n, kernel_h, kernel_w, stride_h, stride_w))
        patch_size_list.append(img_patches_final_size.shape[1])
        n += 1
    index_array, vl = index_of_last_apperance(patch_size_list)
    patch_container_deduplicate = np.array(patch_container, dtype=object)[index_array].tolist()
    # print(c_denom, vl, sum(vl))
    if concat_last:
        patch_container_deduplicate = torch.cat(patch_container_deduplicate, 1)

    return patch_container_deduplicate


def index_of_last_apperance(patch_size_list):
    rd = dict()
    for i, ele in enumerate(patch_size_list):
        if ele not in rd:
            rd[ele] = [i]
        else:
            rd[ele].append(i)
    rl = []
    vl = []
    for key in rd:
        rl.append(max(rd[key]))
        vl.append(key)
    return rl, vl


def get_path_img(img, h_range, w_range):
    patch_img = img[:, : ,h_range[0]:h_range[1],w_range[0]:w_range[1]]
    patch_img = rearrange(patch_img, 'b c h w -> b (c h w)')
    patch_img = torch.unsqueeze(patch_img, -1)
    return patch_img


def return_sliding_windows(img, kernel_h, kernel_w, stride_h, stride_w):
    result = []
    b, c, h, w = img.shape
    for i in range(0, h, stride_h):
        for j in range(0, w, stride_w):
            if i + kernel_h < h:
                if j+kernel_w < w:
                    result.append(get_path_img(img, [i, i+kernel_h], [j, j+kernel_w]))
                else:
                    result.append(get_path_img(img, [i, i+kernel_h], [w-kernel_w, w]))
                    break
            else:
                if j+kernel_w < w:
                    result.append(get_path_img(img, [h-kernel_h, h], [j,j+kernel_w]))
                else:
                    result.append(get_path_img(img, [h-kernel_h, h], [w-kernel_w, w]))
                    result = torch.cat(result, 2)
                    return result.permute(0, 2, 1)


def main_patch_generation():
    """
    Save patch-cc samples
    :return: None
    """
    # 1:2
    img_path = "/home/zhangzilun/Desktop/Data-Gen/cmag_paper-master/data/shapenet51_10k/test/images/1be987c137d37f0b7c15f7bdb6fa82dd_Bowl_iso.png"

    transform_sample = imagenet_transform()
    img = Image.open(img_path)
    img = transform_sample(img).unsqueeze(0)

    # cc@10
    for i in range(10, 11):
        patch_container = img_2patch(img, c_denom=i, concat_last=False)
        # print()

    dir_name = "sample_patch_img_{}".format(img_path.split("/")[-1].split(".")[0])
    os.makedirs(dir_name, exist_ok=True)
    for i, patch_level in enumerate(patch_container):
        for j, batch_sample in enumerate(patch_level):
            for k, patch in enumerate(batch_sample):
                save_image(patch, '{}/sample_img_patch_level{}_batch{}_patch{}.png'.format(dir_name, i, j, k))


def extract_single_image_cc_feat():
    model_name = "slip"
    ckpt_path = "./model/checkpoints/slip_vitb16_100ep.pt"
    device = "cuda"
    model, img_input_size = get_model(model_name, ckpt_path, device)
    transform = imagenet_transform()

    Image.MAX_IMAGE_PIXELS = None
    img_path = "./000000000139.jpg"
    # bgr image in numpy format
    image_np = smart_load_image(img_path)
    image = Image.fromarray(np.uint8(image_np))
    image_after_transform = transform(image).unsqueeze(0)
    patch_img = img_2patch(image_after_transform, c_denom=10, final_size=img_input_size).squeeze(0)
    patch_img = patch_img.unsqueeze(0).to(device)

    image_diopen = Image.open(img_path)
    image_diopen_after_transform = transform(image_diopen).unsqueeze(0)
    patch_img_diopen = img_2patch(image_diopen_after_transform, c_denom=10, final_size=img_input_size).squeeze(0)
    patch_img_diopen = patch_img_diopen.unsqueeze(0).to(device)

    with torch.no_grad():
        b = patch_img.shape[0]
        patch_container = rearrange(patch_img, 'b p c h w -> (b p) c h w')
        clip_feat = model.encode_image(patch_container).to(torch.float32).cpu().detach().numpy()
        cc_feat = rearrange(clip_feat, '(b p) f -> b p f', b=b)

        patch_container_diopen = rearrange(patch_img_diopen, 'b p c h w -> (b p) c h w')
        clip_feat_diopen = model.encode_image(patch_container_diopen).to(torch.float32).cpu().detach().numpy()
        cc_feat_diopen = rearrange(clip_feat_diopen, '(b p) f -> b p f', b=b)

    pdb.set_trace()


def get_vanilla_feat_from_patch_feat(source_dir, target_dir):
    """
    Obtain full (vanilla) clip feature from image feature in patch-cc scheme (entire image, first layer of patch-cc scheme)
    :param source_dir: patch feature dir
    :param target_dir: vanilla feature dir
    :return: None
    """
    # sample_train_b2_new_000499_image_patches-10_ViT-B-32.npy
    all_patch_feat_name = smart_listdir(source_dir)
    os.makedirs(target_dir, exist_ok=True)
    for patch_feat_name in all_patch_feat_name:
        patch_feat_path = os.path.join(source_dir, patch_feat_name)
        patch_feat = smart_pkl_load(patch_feat_path)
        full_clip_feat = patch_feat[0, :]
        vanilla_feat_path = os.path.join(target_dir, patch_feat_name)
        smart_pkl_dump(vanilla_feat_path, full_clip_feat.reshape(1, -1))


def get_model(model_name, ckpt_path, device):

    if model_name == "slip" or model_name == "clip":
        from model.slip import slip_models
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v

        # create model
        old_args = ckpt['args']
        print("=> creating model: {}".format(old_args.model))
        model = getattr(slip_models, old_args.model)(rand_embed=False,
                                                ssl_mlp_dim=old_args.ssl_mlp_dim, ssl_emb_dim=old_args.ssl_emb_dim)
        model.to(device)
        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint from'{}' (epoch {})".format(ckpt_path, ckpt['epoch']))
        img_input_size = (224, 224)

    elif model_name == "clip-openai-32":
        import clip
        clip_encoder_name = "ViT-B/32"
        model, _ = clip.load(clip_encoder_name, device, jit=True)
        img_input_size = (224, 224)

    elif model_name == "clip-openai-16":
        import clip
        clip_encoder_name = "ViT-B/16"
        model, _ = clip.load(clip_encoder_name, device, jit=True)
        img_input_size = (224, 224)

    elif model_name == "clip-openai-14":
        import clip
        clip_encoder_name = "ViT-L/14@336px"
        model, _ = clip.load(clip_encoder_name, device, jit=True)
        img_input_size = (336, 336)

    else:
        print("select a valid model name (clip, slip)")
        exit()

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model, img_input_size


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_runner', type=int, default=1,
                        help='number of gpu per trail')
    # #gpu per runner
    parser.add_argument('--num_gpu', type=int, default=1,
                        help='number of gpu per trail')
    # #cpu per runner
    parser.add_argument('--num_cpu', type=int, default=4,
                        help='number of cpu per trail')

    parser.add_argument('--ray_mode', type=str, default="local_run", help='local mode or auto mode')

    parser.add_argument('--batch_size', type=int, default=1, help='batch size')

    parser.add_argument('--bmk_root', type=str,
                        default="mmr/bmk",
                        help='')

    parser.add_argument('--use_oss', type=bool,
                        default=False,
                        help='if data is on oss')

    parser.add_argument('--partition', type=str,
                        default="test",
                        help='train, val or test')

    parser.add_argument('--extract_type', type=str,
                        default="text",
                        help='image-cc, image-grid, image-full, image-obj, text')

    parser.add_argument('--label_dict_path', type=str,
                        default="./config/labels.json",
                        help='dataset label dict')

    parser.add_argument('--template_dict_path', type=str,
                        default="./config/templates.json",
                        help='dataset template dict')

    parser.add_argument('--dataset_name', type=str,
                        default="flickr30k",
                        help='dataset name')

    parser.add_argument('--model_name', type=str,
                        default="slip",
                        help='model name')

    parser.add_argument('--model_ckpt_path', type=str,
                        default="./model/checkpoints/slip_vitb16_100ep.pt",
                        help='model checkpoint name')

    parser.add_argument('--text_type', type=str,
                        default="caption",
                        help='text data is caption or cls')

    args = parser.parse_args()
    args.model_name = args.model_name.replace("/", "-")

    args.dataset_dir = os.path.join(args.bmk_root, args.dataset_name)
    args.original_data_dir = os.path.join(args.dataset_dir, "original_data")
    args.original_data_partition_dir = os.path.join(args.original_data_dir, args.partition)
    args.original_data_partition_img_dir = os.path.join(args.original_data_partition_dir, "images")
    args.original_data_partition_scene_dir = os.path.join(args.original_data_partition_dir, "scenes")

    args.all_feat_dir = os.path.join(args.dataset_dir, "feat")

    args.original_data_dir = expand_path(args.original_data_dir, args.use_oss)
    args.original_data_partition_dir = expand_path(args.original_data_partition_dir, args.use_oss)
    args.original_data_partition_img_dir = expand_path(args.original_data_partition_img_dir, args.use_oss)
    args.original_data_partition_scene_dir = expand_path(args.original_data_partition_scene_dir, args.use_oss)
    args.model_ckpt_path = expand_path(args.model_ckpt_path, args.use_oss)
    args.all_feat_dir = expand_path(args.all_feat_dir, args.use_oss)

    extract(args)


if __name__ == "__main__":
    main()
    # main_patch_generation()
    # extract_single_image_cc_feat()
