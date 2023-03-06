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

from modeling.crisis_vit import CrisisVitDs
from torch.utils.data import RandomSampler


def train_one_epoch(model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, config, args, task):
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
        # print(incidents_labels)
        labels = labels.to(device, non_blocking=True)

        loss= model(image, labels=labels, task=task)

        loss.backward()
        optimizer.step()

        # if i % 3000 == 0 and args.multi_task:
        #     _, pred_inci_class = preds_inci.max(1)
        #     acc_inci = (incidents_labels == pred_inci_class).sum() / incidents_labels.size(0) * 100
        #     _, pred_plac_class = preds_plac.max(1)
        #     acc_plac = (places_labels == pred_plac_class).sum() / places_labels.size(0) * 100
        #     print(f'Pred_inci_acc: {acc_inci}, Pred_plac_acc: {acc_plac}')


        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

            # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: float("{:.4f}".format(meter.global_avg)) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluation(model, data_loader, device, task):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    for images, labels in metric_logger.log_every(data_loader, print_freq, header):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)


        prediction = model(images, labels, task=task, train=False)

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
                   train_loader, val_loader, test_loader,
                   device, config, args, task):

    # arg_opt = utils.AttrDict(config['optimizer'])
    # optimizer = create_optimizer(arg_opt, model)
    # arg_sche = utils.AttrDict(config['schedular'])
    # lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    print("Start training")
    start_time = time.time()
    best = 0
    best_test = 0

    for epoch in range(start_epoch, max_epoch):

        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        train_stats = train_one_epoch(model, train_loader, optimizer, epoch, warmup_steps,
                                      device, lr_scheduler, config, args, task)

        val_stats = evaluation(model, val_loader, device, task)
        test_stats = evaluation(model, test_loader, device, task)

        if utils.is_main_process():
            if args.evaluate:
                log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                             }

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
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
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    best = float(val_stats['acc'])

                if float(test_stats['acc']) > best_test:
                    best_test = float(test_stats['acc'])

        print(f'Best val_acc: {best}')
        print(f'Best test_acc: {best_test}')
        if args.evaluate:
            break

        if args.wandb_enable and utils.is_main_process():
            wandb.log(train_stats)
            wandb.log(val_stats)
            wandb.log(test_stats)

        lr_scheduler.step(epoch + warmup_steps + 1)
        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def inilise_model(args, config, device):
    model = CrisisVitDs(args.visual_encoder, config)
    model = model.to(device)

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    start_epoch = 0

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        for k in list(state_dict.keys()):
            state_dict[k.replace('mae_model', 'visual_encoder')] = state_dict.pop(k)

        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

        # if config['distill'] is True:
        #     model.copy_params()

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    return model, model_without_ddp, optimizer, lr_scheduler, start_epoch


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


    #### Dataset ####
    print("Creating dataset")
    # dt_train_dataset, dt_val_dataset, dt_test_dataset = create_dataset('disaster_types', config)
    # info_train_dataset, info_val_dataset, info_test_dataset = create_dataset('informativeness', config)
    # human_train_dataset, human_val_dataset, human_test_dataset = create_dataset('humanitarian', config)
    # damage_train_dataset, damage_val_dataset, damage_test_dataset = create_dataset('damage_severity', config)
    dt_datasets = create_dataset('disaster_types', config)
    info_datasets = create_dataset('informativeness', config)
    human_datasets = create_dataset('humanitarian', config)
    damage_datasets = create_dataset('damage_severity', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        dt_samplers = create_sampler(dt_datasets, [True, False, False], num_tasks, global_rank)
        info_samplers = create_sampler(info_datasets, [True, False, False], num_tasks, global_rank)
        human_samplers = create_sampler(human_datasets, [True, False, False], num_tasks, global_rank)
        damage_samplers = create_sampler(damage_datasets, [True, False, False], num_tasks, global_rank)
    else:
        # samplers = RandomSampler(datasets)
        dt_samplers = [RandomSampler(dt_datasets[0]), RandomSampler(dt_datasets[1]), RandomSampler(dt_datasets[2])]
        info_samplers = [RandomSampler(info_datasets[0]), RandomSampler(info_datasets[1]), RandomSampler(info_datasets[2])]
        human_samplers = [RandomSampler(human_datasets[0]), RandomSampler(human_datasets[1]), RandomSampler(human_datasets[2])]
        damage_samplers = [RandomSampler(damage_datasets[0]), RandomSampler(damage_datasets[1]), RandomSampler(damage_datasets[2])]

        # info_samplers = [RandomSampler(info_train_dataset)]
        # human_samplers = [RandomSampler(human_train_dataset)]
        # damage_samplers = [RandomSampler(damage_train_dataset)]

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


    dt_train_loader, dt_val_loader, dt_test_loader = \
        create_loader(dt_datasets, dt_samplers,
                      batch_size=[config['batch_size_train']] + [
                          config['batch_size_test']] * 2,
                      num_workers=[2, 2, 2],
                      is_trains=[True, False, False],
                      collate_fns=[None, None, None]
                      )

    info_train_loader, info_val_loader, info_test_loader = \
        create_loader(info_datasets, info_samplers,
                      batch_size=[config['batch_size_train']] + [
                          config['batch_size_test']] * 2,
                      num_workers=[2, 2, 2],
                      is_trains=[True, False, False],
                      collate_fns=[None, None, None]
                      )


    human_train_loader, human_val_loader, human_test_loader = \
        create_loader(human_datasets, human_samplers,
                        batch_size=[config['batch_size_train']] + [
                            config['batch_size_test']] * 2,
                        num_workers=[2, 2, 2],
                        is_trains=[True, False, False],
                        collate_fns=[None, None, None]
                        )

    damage_train_loader, damage_val_loader, damage_test_loader = \
            create_loader(damage_datasets, damage_samplers,
                        batch_size=[config['batch_size_train']] + [
                            config['batch_size_test']] * 2,
                        num_workers=[2, 2, 2],
                        is_trains=[True, False, False],
                        collate_fns=[None, None, None]
                        )


    #### Model ####

    model, model_without_ddp, optimizer, lr_scheduler, start_epoch = inilise_model(args, config, device)
    print("Start training for disaster_types ")
    train_one_task(start_epoch, max_epoch, warmup_steps,
                   model, model_without_ddp,
                   lr_scheduler, optimizer,
                   dt_train_loader, dt_val_loader, dt_test_loader,
                  device,  config, args, task='dt')

    torch.cuda.empty_cache()

    model, model_without_ddp, optimizer, lr_scheduler, start_epoch = inilise_model(args, config, device)
    print("Start training for informativeness")
    train_one_task(start_epoch, max_epoch, warmup_steps,
                     model, model_without_ddp,
                   lr_scheduler, optimizer,
                     info_train_loader, info_val_loader, info_test_loader,
                    device,  config, args, task='info')

    torch.cuda.empty_cache()
    model, model_without_ddp, optimizer, lr_scheduler, start_epoch = inilise_model(args, config, device)
    print("Start training for humanitarian")
    train_one_task(start_epoch, max_epoch, warmup_steps,
                     model, model_without_ddp,
                   lr_scheduler, optimizer,
                     human_train_loader, human_val_loader, human_test_loader,
                    device,  config, args, task='human')

    torch.cuda.empty_cache()
    model, model_without_ddp, optimizer, lr_scheduler, start_epoch = inilise_model(args, config, device)
    print("Start training for damage_severity")
    train_one_task(start_epoch, max_epoch, warmup_steps,
                     model, model_without_ddp,
                    lr_scheduler, optimizer,
                     damage_train_loader, damage_val_loader, damage_test_loader,
                    device,  config, args, task='damage')

    if utils.is_main_process() and args.wandb_enable:
          wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/downstream.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='output/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=None)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)


    # parser.add_argument('--task', default='incidence', type=str)
    parser.add_argument('--visual_encoder', default='MAE')
    parser.add_argument('--visual_encoder_weight', default='exchange/mae_finetuned_vit_base.pth')
    parser.add_argument('--image_res', default=224, type=int, help='The resolution of the image')

    # wandb
    parser.add_argument('--project', type=str, default='IncidentOneM_downstreams',
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