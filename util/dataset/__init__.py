import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from util.dataset.visual_dataset import contrastive_pretrain_dataset, pretrain_dataset
from util.dataset.visual_dataset import binary_incidentonem
from util.dataset.visual_dataset import crisis_image_benchmarks
from util.dataset.randaugment import RandomAugment
from util.dataset.utils import GaussianBlur
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def create_dataset(dataset, config):
    # augmentations
    # jinyu: add augmentation
    # todo: add augmentation with augreg
    # construct data loader

    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=config['image_res'], scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])


    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'pretrain':
        train_dataset  = pretrain_dataset(config['train_file'], train_transform)
        val_dataset = pretrain_dataset(config['val_file'], test_transform)
        return train_dataset, val_dataset

    # doataloader for downstream tasks
    if dataset == 'disaster_types':
        train_dataset = crisis_image_benchmarks(config['disaster_types']['train_file'], train_transform)
        val_dataset = crisis_image_benchmarks(config['disaster_types']['val_file'], test_transform)
        test_dataset = crisis_image_benchmarks(config['disaster_types']['test_file'], test_transform)
        return train_dataset, val_dataset, test_dataset

    if dataset == 'informativeness':
        train_dataset =  crisis_image_benchmarks(config['informativeness']['train_file'], train_transform)
        val_dataset =  crisis_image_benchmarks(config['informativeness']['val_file'], test_transform)
        test_dataset =  crisis_image_benchmarks(config['informativeness']['test_file'], test_transform)
        return train_dataset, val_dataset, test_dataset
    if dataset == 'humanitarian':
        train_dataset =  crisis_image_benchmarks(config['humanitarian']['train_file'], train_transform)
        val_dataset =  crisis_image_benchmarks(config['humanitarian']['val_file'], test_transform)
        test_dataset =  crisis_image_benchmarks(config['humanitarian']['test_file'], test_transform)
        return train_dataset, val_dataset, test_dataset

    if dataset == 'damage_severity':
        train_dataset =  crisis_image_benchmarks(config['damage_severity']['train_file'], train_transform)
        val_dataset =  crisis_image_benchmarks(config['damage_severity']['val_file'], test_transform)
        test_dataset =  crisis_image_benchmarks(config['damage_severity']['test_file'], test_transform)
        return train_dataset, val_dataset, test_dataset


    # if dataset == 'damage_severity':
    #     return crisis_image_benchmarks(config['damage_severity'], RandomAugment(2, 5))

def create_binary_dataset(dataset, config, task):
    # augmentations
    # jinyu: add augmentation
    # todo: add augmentation with augreg
    # construct data loader

    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=config['image_res'], scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])


    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'incidents':
        train_dataset = binary_incidentonem(config['incidentonem_train'], train_transform, "incidents_binary", task)
        val_dataset = binary_incidentonem(config['incidentonem_val'], test_transform, "incidents_binary", task)
        return train_dataset, val_dataset

    if dataset == 'places':
        train_dataset = binary_incidentonem(config['incidentonem_train'], train_transform, "places_binary", task)
        val_dataset = binary_incidentonem(config['incidentonem_val'], test_transform, "places_binary", task)
        return train_dataset, val_dataset


def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list, dim=0), question_list, answer_list, torch.Tensor(weight_list), n


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders