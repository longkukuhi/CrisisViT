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
from util.dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
import wandb

from modeling.crisis_vit import CrisisVit, CrisisCNN
from torch.utils.data import RandomSampler
from timm.models.layers import trunc_normal_
from modeling.mae.pos_embed import interpolate_pos_embed

def vl_train(model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, config, args):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    if args.task == 'multi':
        metric_logger.add_meter('loss_incidents', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_places', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    elif args.task == 'incidence':
        metric_logger.add_meter('loss_incidents', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    elif args.task == 'place':
        metric_logger.add_meter('loss_places', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    else:
        metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (image, incidents_labels, places_labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        optimizer.zero_grad()

        image = image.to(device, non_blocking=True)
        # print(incidents_labels)
        incidents_labels = incidents_labels.to(device, non_blocking=True)
        places_labels = places_labels.to(device, non_blocking=True)

        if args.task == 'multi':
            loss_incidents, loss_places = model(image, incidents_labels, places_labels, task=args.task)
            loss = loss_incidents + loss_places

        elif args.task == 'incidence':
            loss_incidents = model(image, labels_inci=incidents_labels, task=args.task)
            loss = loss_incidents

        elif args.task == 'place':
            loss_places = model(image, labels_plac=places_labels, task=args.task)
            loss = loss_places

        else:
            loss = model(image,incidents_labels, places_labels, task=args.task)


        loss.backward()
        optimizer.step()


        if args.task == 'multi':
            metric_logger.update(loss_incidents=loss_incidents.item())
            metric_logger.update(loss_places=loss_places.item())
        elif args.task == 'incidence':
            metric_logger.update(loss_incidents=loss_incidents.item())
        elif args.task == 'place':
            metric_logger.update(loss_places=loss_places.item())
        else:
            metric_logger.update(loss_mlm=loss.item())

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

            # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: float("{:.4f}".format(meter.global_avg)) for k, meter in metric_logger.meters.items()}

def vl_train_contrast(model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    if config['mlm'] is True:
        metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (image, image_aug, text)  in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        optimizer.zero_grad()

        image = image.to(device, non_blocking=True)
        image_aug = image_aug.to(device, non_blocking=True)

        text_input = tokenizer(text, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device)

        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        if config['mlm'] is True:
            loss_itc, loss_itm, loss_mlm = model(text=text_input, image=image, image_aug=image_aug,device=device, alpha=alpha, freeze_att_score=config['freeze_att_score'])
            loss = loss_itc + loss_itm + loss_mlm

        else:
            loss_itc, loss_itm = model(text=text_input, image=image, image_aug=image_aug, device=device, alpha = alpha, freeze_att_score=config['freeze_att_score'])
            loss = loss_itc + loss_itm


        loss.backward()
        optimizer.step()

        metric_logger.update(loss_itc=loss_itc.item())
        metric_logger.update(loss_itm=loss_itm.item())
        if config['mlm'] is True:
            metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

        if i % 3000 == 0:
            date_time = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            print(date_time)

            # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    date_time = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    print("Time:", date_time, "Averaged stats:", metric_logger.global_avg())
    return {k: float("{:.4f}".format(meter.global_avg)) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluation(model, data_loader, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    for image, incidents_labels, places_labels in metric_logger.log_every(data_loader, print_freq, header):

        image = image.to(device, non_blocking=True)
        incidents_labels = incidents_labels.to(device, non_blocking=True)
        places_labels = places_labels.to(device, non_blocking=True)

        if args.task == 'multi':
            preds_inci, preds_plac = model(image, incidents_labels, places_labels, task=args.task,
                                                                    train=False)
            _, pred_inci_class = preds_inci.max(1)
            _, pred_plac_class = preds_plac.max(1)
            acc_inci = (pred_inci_class == incidents_labels).sum() / incidents_labels.size(0) * 100
            acc_plac = (pred_plac_class == places_labels).sum() / places_labels.size(0) * 100

            metric_logger.meters['acc_inci'].update(acc_inci.item(), n=image.size(0))
            metric_logger.meters['acc_plac'].update(acc_plac.item(), n=image.size(0))

        elif args.task == 'incidence':
            preds_inci = model(image, labels_inci=incidents_labels, task=args.task,
                               train=False)
            _, pred_inci_class = preds_inci.max(1)

            acc_inci = (pred_inci_class == incidents_labels).sum() / incidents_labels.size(0) * 100
            metric_logger.meters['acc_inci'].update(acc_inci.item(), n=image.size(0))

        elif args.task == 'place':
            preds_plac = model(image, labels_plac=places_labels, task=args.task,
                               train=False)
            _, pred_plac_class = preds_plac.max(1)

            acc_plac = (pred_plac_class == places_labels).sum() / places_labels.size(0) * 100
            metric_logger.meters['acc_plac'].update(acc_plac.item(), n=image.size(0))




    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: float("{:.4f}".format(meter.global_avg)) for k, meter in metric_logger.meters.items()}



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

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']


    #### Dataset ####
    print("Creating dataset")
    if config['distill'] is True:
        datasets = create_dataset('pretrain_momentum', config)
    else:
        datasets = create_dataset('pretrain', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True,False], num_tasks, global_rank)
        # enable wandb
    else:
        # samplers = RandomSampler(datasets)
        samplers = [RandomSampler(dataset) for dataset in datasets]


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

    train_loader, val_loader = create_loader(datasets, samplers,
                                                 batch_size=[config['batch_size_train'],
                                                     config['batch_size_test']],
                                                 num_workers=[2, 2],
                                                 is_trains=[True, False],
                                                 collate_fns=[None, None]
                                                 )



    #### Model ####
    print("Creating model")
    if args.visual_encoder == 'mae':
        model = CrisisVit(args.visual_encoder, config)
    elif args.visual_encoder == 'resnet34' or args.visual_encoder == 'resnet101d':
        model = CrisisCNN(args.visual_encoder, config, args.task)

    model = model.to(device)

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1

        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        print('load checkpoint from %s' % args.checkpoint)
        # if config['distill'] is True:
        #     model.copy_params()

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in list(checkpoint_model.keys()):
            checkpoint_model['mae_model.'+k] = checkpoint_model.pop(k)

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # if args.global_pool:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        # else:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.img_encoder.head.weight, std=2e-5)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):

        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        if  config['distill'] is True:
            train_stats = vl_train_contrast(model, train_loader, optimizer, epoch, warmup_steps,
                                   device, lr_scheduler, config, args)
            val_stats = evaluation(model, val_loader, device, args)
        else:
            train_stats = vl_train(model, train_loader, optimizer, epoch, warmup_steps,
                                   device, lr_scheduler, config, args)
            val_stats = evaluation(model, val_loader, device, args)

        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch,
                         }

            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth' % epoch))

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if args.wandb_enable:
                wandb.log(log_stats)
        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if utils.is_main_process() and args.wandb_enable:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/mae.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='output/place/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=None)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)

    # parser.add_argument('--multi_task', default=False, type=bool)
    parser.add_argument('--task', default='incidence', type=str)
    parser.add_argument('--visual_encoder', default='MAE')
    parser.add_argument('--visual_encoder_weight', default='exchange/mae_finetuned_vit_base.pth')
    parser.add_argument('--image_res', default=224, type=int, help='The resolution of the image')

    # wandb
    parser.add_argument('--project', type=str, default='IncidentOneM_task_pretrain',
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