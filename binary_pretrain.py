import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist


from util import utils
from util.dataset import create_binary_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
import wandb
from torch.utils.data import random_split
from modeling.crisis_vit import CrisisVitPretrainBinary
from torch.utils.data import RandomSampler


def train_one_epoch(model, data_loader, optimizer, epoch, warmup_steps,
                    device, scheduler, config, args, task):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (image, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        optimizer.zero_grad()

        image = image.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        loss= model(image, labels=labels)

        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

            # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: float("{:.4f}".format(meter.global_avg)) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluation(model, data_loader, device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    for images, labels in metric_logger.log_every(data_loader, print_freq, header):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)


        prediction = model(images, labels, train=False)

        _, pred_class = prediction.max(1)
        accuracy = (labels == pred_class).sum() / labels.size(0) * 100

        metric_logger.meters['acc'].update(accuracy.item(), n=images.size(0))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: float("{:.4f}".format(meter.global_avg)) for k, meter in metric_logger.meters.items()}



def train_one_task(start_epoch, max_epoch, warmup_steps,
                    model, model_without_ddp,
                    lr_scheduler, optimizer,
                   train_loader, val_loader,
                   device, config, args, category, task):


    print("Start training")
    start_time = time.time()
    best = 0

    model.create_binary_cls_head()

    for epoch in range(start_epoch, max_epoch):

        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        train_stats = train_one_epoch(model, train_loader, optimizer, epoch, warmup_steps,
                                      device, lr_scheduler, config, args, task)

        val_stats = evaluation(model, val_loader, device)


        if utils.is_main_process():
            if args.evaluate:
                log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                             'epoch': epoch,
                             }

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_stats.items()},
                             'epoch': epoch,
                             }

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if float(val_stats['acc']) > best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_best_{category}_{task}.pth'))
                    best = float(val_stats['acc'])


        print(f'Best val_acc: {best}')

        if args.evaluate:
            break

        if args.wandb_enable and utils.is_main_process():
            wandb.log(train_stats)
            wandb.log(val_stats)


        lr_scheduler.step(epoch + warmup_steps + 1)
        if args.distributed:
            dist.barrier()

    if utils.is_main_process():
        save_obj = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'config': config,
            'epoch': epoch,
        }
        torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_last_{category}_{task}.pth'))


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return best

def inilise_optimization(args, config, device, model):


    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    start_epoch = 0


    return  optimizer, lr_scheduler, start_epoch

def create_one_category_dataloader(args, config, category, task):
    print("Creating dataset for task: ", category, task)
    datasets = create_binary_dataset(category, config, task)
    # train_dataset, val_dataset = random_split(dataset, [ 0.9,  0.1])
    # datasets = [train_dataset, val_dataset]

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True], num_tasks, global_rank)
    else:
        samplers = [RandomSampler(dataset) for dataset in datasets]

    train_loader, val_loader = create_loader(datasets, samplers,
                                                 batch_size=[config['batch_size_train'],
                                                     config['batch_size_test']],
                                                 num_workers=[2, 2],
                                                 is_trains=[True, False],
                                                 collate_fns=[None, None]
                                                 )

    return train_loader, val_loader

def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    if args.seed is not None:
        print("Using seed: ", (int(args.seed) + utils.get_rank()))
        seed = int(args.seed) + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    else:
        print("Do not use seed")

    cudnn.benchmark = True


    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    if utils.is_main_process() and args.wandb_enable:
        wandb_config = {
            'lr': config['optimizer']['lr'],
            'epochs': config['schedular']['epochs'],
            'batch_size': config['batch_size_train'],
        }
        wandb.init(project=args.project,
                   tags=["IncidentOneM", args.tag],
                   config=wandb_config, save_code=True  # , mode="offline"
                   )
        wandb.config.update(args)
        wandb.save(os.path.join(args.output_dir, 'config.yaml'))
        wandb.save('main_pretrain.py')
        wandb.save('modeling/crisis_vit.py')

    model = CrisisVitPretrainBinary(args.visual_encoder, config)
    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module



    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        if args.resume:
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1

        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        print('load checkpoint from %s' % args.checkpoint)
        # if config['distill'] is True:
        #     model.copy_params()

    incident_label_dict = {0: 'airplane accident', 1: 'bicycle accident', 2: 'blocked',
     3: 'burned', 4: 'bus accident', 5: 'car accident', 6: 'collapsed', 7: 'damaged',
     8: 'derecho', 9: 'dirty contamined', 10: 'drought', 11: 'dust devil',
     12: 'dust sand storm', 13: 'earthquake', 14: 'fire whirl',
     15: 'flooded', 16: 'fog', 17: 'hailstorm',
     18: 'heavy rainfall', 19: 'ice storm', 20: 'landslide',
     21: 'motorcycle accident', 22: 'mudslide mudflow', 23: 'nuclear explosion',
     24: 'oil spill', 25: 'on fire', 26: 'rockslide rockfall',
     27: 'ship boat accident', 28: 'sinkhole', 29: 'snow covered',
     30: 'snowslide avalanche', 31: 'storm surge', 32: 'thunderstorm',
     33: 'tornado', 34: 'traffic jam', 35: 'train accident',
     36: 'tropical cyclone', 37: 'truck accident', 38: 'under construction',
     39: 'van accident', 40: 'volcanic eruption', 41: 'wildfire',
     42: 'with smoke'}
    places_label_dict = {0: 'badlands',
     1: 'beach',
     2: 'bridge',
     3: 'building facade',
     4: 'building outdoor',
     5: 'cabin outdoor',
     6: 'coast',
     7: 'construction site',
     8: 'dam',
     9: 'desert',
     10: 'desert road',
     11: 'downtown',
     12: 'excavation',
     13: 'farm',
     14: 'field',
     15: 'fire station',
     16: 'forest',
     17: 'forest road',
     18: 'gas station',
     19: 'glacier',
     20: 'highway',
     21: 'house',
     22: 'industrial area',
     23: 'junkyard',
     24: 'lake natural',
     25: 'landfill',
     26: 'lighthouse',
     27: 'mountain',
     28: 'nuclear power plant',
     29: 'ocean',
     30: 'oil rig',
     31: 'park',
     32: 'parking lot',
     33: 'pier',
     34: 'port',
     35: 'power line',
     36: 'railroad track',
     37: 'religious building',
     38: 'residential neighborhood',
     39: 'river',
     40: 'sky',
     41: 'skyscraper',
     42: 'slum',
     43: 'snowfield',
     44: 'sports field',
     45: 'street',
     46: 'valley',
     47: 'village',
     48: 'volcano'}

    # training for the incident task:
    print("Training for the incident task")
    best_incident_acc = {}

    for task in range(args.start_task, 43):
        print("Start training for task: ", incident_label_dict[task])
        optimizer, lr_scheduler, start_epoch = inilise_optimization(args, config, device, model)
        train_loader, val_loader = create_one_category_dataloader(args, config, 'incidents', task)
        best_incident_acc[incident_label_dict[task]] = train_one_task(start_epoch, max_epoch,
                                                                      warmup_steps,
                                                                    model, model_without_ddp,
                                                                    lr_scheduler, optimizer,
                                                                    train_loader, val_loader,
                                                                    device, config, args,
                                                                    category='incident', task=task)
        torch.cuda.empty_cache()


    # training for the places task
    print("Training for the places task")
    best_places_acc = {}
    for task in range(49):
        print("Start training for task: ", places_label_dict[task])
        optimizer, lr_scheduler, start_epoch = inilise_optimization(args, config, device, model)
        train_loader, val_loader = create_one_category_dataloader(args, config, 'places', task)
        best_places_acc[places_label_dict[task]] = train_one_task(start_epoch, max_epoch, warmup_steps,
                       model, model_without_ddp,
                       lr_scheduler, optimizer,
                       train_loader, val_loader,
                       device, config, args,
                       category='places', task=task)
        torch.cuda.empty_cache()

    print("Best incident acc: ", best_incident_acc)
    print("Best places acc: ", best_places_acc)
    #### Model ####


    if utils.is_main_process() and args.wandb_enable:
          wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/pretrain.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='output/binary_pretrain/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=None)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--start_task', default=0, type=int)


    # parser.add_argument('--task', default='incidence', type=str)
    parser.add_argument('--visual_encoder', default='MAE')
    parser.add_argument('--visual_encoder_weight', default='exchange/mae_finetuned_vit_base.pth')
    parser.add_argument('--image_res', default=224, type=int, help='The resolution of the image')

    # wandb
    parser.add_argument('--project', type=str, default='IncidentOneM_pretrain_binary',
                         help='project name for wandb')
    parser.add_argument('--wandb_enable', action='store_true',
                        help='enable wandb to record')
    parser.add_argument('--tag', type=str, default='MAE',
                         help='Tag for wandb')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)