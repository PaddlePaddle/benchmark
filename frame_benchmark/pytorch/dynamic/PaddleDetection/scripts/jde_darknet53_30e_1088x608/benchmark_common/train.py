import argparse
import os
import json
import time
from time import gmtime, strftime
import test
from models import *
from shutil import copyfile
from utils.datasets import JointDataset, collate_fn
from utils.utils import *
from utils.log import logger
from torchvision.transforms import transforms as T

import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count


def train(
        cfg,
        data_cfg,
        weights_from="",
        weights_to="",
        save_every=10,
        img_size=(1088, 608),
        resume=False,
        epochs=100,
        batch_size=16,
        accumulated_batches=1,
        freeze_backbone=False,
        opt=None):
    # The function starts

    timme = strftime("%Y-%d-%m %H:%M:%S", gmtime())
    timme = timme[5:-3].replace('-', '_')
    timme = timme.replace(' ', '_')
    timme = timme.replace(':', '_')
    weights_to = osp.join(weights_to, 'run' + timme)
    mkdir_if_missing(weights_to)
    if resume:
        latest_resume = osp.join(weights_from, 'latest.pt')

    torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print("rank = ", rank)
    print("local_rank = ", local_rank)

    torch.cuda.set_device(rank % torch.cuda.device_count())

    print("local, local_rank", rank % torch.cuda.device_count(), local_rank)
    # 分布式通讯协议使用NCCL
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size = ", world_size)
    if world_size > 1:
        print('------world_size = {}'.format(os.getenv("WORLD_SIZE")))
        dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    # Configure run
    f = open(data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    f.close()

    transforms = T.Compose([T.ToTensor()])
    # Get dataloader
    dataset = JointDataset(dataset_root, trainset_paths, img_size, augment=True, transforms=transforms)

    dist_sampler = None
    if world_size > 1:
        dist_sampler = DistributedSampler(dataset, shuffle=True)

    print("batch_size = ", batch_size)

    dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size=batch_size, shuffle=False, sampler = dist_sampler,
                                             num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    # Initialize model
    model = Darknet(cfg, dataset.nID)

    model = model.to(device)

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    if resume:
        checkpoint = torch.load(latest_resume, map_location='cpu')

        # Load weights to resume from
        model.load_state_dict(checkpoint['model'])
        model.cuda().train()

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr, momentum=.9)

        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])

        del checkpoint  # current, saved

    else:
        # Initialize model with backbone (optional)
        if cfg.endswith('yolov3.cfg'):
            load_darknet_weights(model, osp.join(weights_from, 'darknet53.conv.74'))
            cutoff = 75
        elif cfg.endswith('yolov3-tiny.cfg'):
            load_darknet_weights(model, osp.join(weights_from, 'yolov3-tiny.conv.15'))
            cutoff = 15

        model.cuda().train()

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr, momentum=.9,
                                    weight_decay=1e-4)
    
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    #model = torch.nn.DataParallel(model)
    # Set scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[int(0.5 * opt.epochs), int(0.75 * opt.epochs)],
                                                     gamma=0.1)

    # An important trick for detection: freeze bn during fine-tuning
    if not opt.unfreeze_bn:
        for i, (name, p) in enumerate(model.named_parameters()):
            p.requires_grad = False if 'batch_norm' in name else True

    batch_time = AverageMeter()
    data_time = AverageMeter()
    # model_info(model)

    t0 = time.time()
    for epoch in range(epochs):
        epoch += start_epoch

        if world_size > 1:
            dist_sampler.set_epoch(epoch)

        logger.info(('%8s%12s' + '%10s' * 6) % (
            'Epoch', 'Batch', 'box', 'conf', 'id', 'total', 'nTargets', 'time'))

        # Freeze darknet53.conv.74 for first epoch
        if freeze_backbone and (epoch < 2):
            for i, (name, p) in enumerate(model.named_parameters()):
                if int(name.split('.')[2]) < cutoff:  # if layer < 75
                    p.requires_grad = False if (epoch == 0) else True

        ui = -1
        rloss = defaultdict(float)  # running loss
        optimizer.zero_grad()
        iter_tic = time.time() ###
        module = getattr(model, "module", model)
        for i, (imgs, targets, _, _, targets_len) in enumerate(dataloader):
            data_time.update(time.time() - iter_tic) ###
            if sum([len(x) for x in targets]) < 1:  # if no targets continue
                continue

            # SGD burn-in
            burnin = min(1000, len(dataloader))
            if (epoch == 0) & (i <= burnin):
                lr = opt.lr * (i / burnin) ** 4
                for g in optimizer.param_groups:
                    g['lr'] = lr

            # Compute loss, compute gradient, update parameters
            loss, components = model(imgs.cuda(), targets.cuda(), targets_len.cuda())
            components = torch.mean(components.view(-1, 5), dim=0)
            loss = torch.mean(loss)
            loss.backward()

            # accumulate gradient for x batches before optimizing
            if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()

            # Running epoch-means of tracked metrics
            ui += 1

            for ii, key in enumerate(module.loss_names):
                rloss[key] = (rloss[key] * ui + components[ii]) / (ui + 1)

            # batch_time.update(time.time() - t0) #
            batch_time.update(time.time() - iter_tic) ###

            # rloss indicates running loss values with mean updated at every epoch
            '''
            s = ('%8s%12s' + '%10.3g' * 6) % (
                '%g/%g' % (epoch, epochs - 1),
                '%g/%g' % (i, len(dataloader) - 1),
                rloss['box'], rloss['conf'],
                rloss['id'], rloss['loss'],
                rloss['nT'], time.time() - t0)
            '''
            # print( "Batch: {} time: {:.5f}".format(i, time.time()-t0))
            # s = 'Epoch: {} Batch: {}/{} avg_batch_cost: {:.5f} loss_bbox: {:.3f} loss_conf: {:.3f} loss_id: {:.3f} loss_total: {:.3f} time: {:.3f}s nT: {}\n'.format(epoch, i, len(dataloader)-1, avg_batch_cost, rloss['box'], rloss['conf'], rloss['id'], rloss['loss'], avg_batch_cost, int(rloss['nT']))
            s = 'Epoch: {} Batch: {}/{} loss_bbox: {:.3f} loss_conf: {:.3f} loss_id: {:.3f} loss_total: {:.3f} nT: {} datatime: {:.3f}s batch time: {:.3f}s ips: {:.3f}'.format(epoch, i, len(dataloader)-1, rloss['box'], rloss['conf'], rloss['id'], rloss['loss'], int(rloss['nT']), data_time.avg, batch_time.avg, batch_size/batch_time.avg)
            # t0 = time.time()
            if i % opt.print_interval == 0:
                logger.info(s)
            
            iter_tic = time.time() ###

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'model': module.state_dict(),
                      'optimizer': optimizer.state_dict()}

        copyfile(cfg, weights_to + '/yolo3.cfg')
        copyfile(data_cfg, weights_to + '/ccmcpe.json')

        latest = osp.join(weights_to, 'latest.pt')
        torch.save(checkpoint, latest)
        if epoch % save_every == 0 and epoch != 0:
            # making the checkpoint lite
            checkpoint["optimizer"] = []
            torch.save(checkpoint, osp.join(weights_to, "weights_epoch_" + str(epoch) + ".pt"))

        # Calculate mAP
        '''
        # no need to eval 
        if epoch % opt.test_interval == 0:
            with torch.no_grad():
                mAP, R, P = test.test(cfg, data_cfg, weights=latest, batch_size=batch_size,
                                      print_interval=40)
                test.test_emb(cfg, data_cfg, weights=latest, batch_size=batch_size,
                              print_interval=40)
        '''
        # Call scheduler.step() after opimizer.step() with pytorch > 1.1.0
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='size of each image batch')
    parser.add_argument('--accumulated-batches', type=int, default=1, help='number of batches before optimizer step')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights-from', type=str, default='weights/',
                        help='Path for getting the trained model for resuming training (Should only be used with '
                             '--resume)')
    parser.add_argument('--weights-to', type=str, default='weights/',
                        help='Store the trained weights after resuming training session. It will create a new folder '
                             'with timestamp in the given path')
    parser.add_argument('--save-model-after', type=int, default=10,
                        help='Save a checkpoint of model at given interval of epochs')
    parser.add_argument('--data-cfg', type=str, default='cfg/ccmcpe.json', help='coco.data file path')
    parser.add_argument('--img-size', type=int, default=[1088, 608], nargs='+', help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--print-interval', type=int, default=5, help='print interval')
    parser.add_argument('--test-interval', type=int, default=9, help='test interval')
    parser.add_argument('--lr', type=float, default=1e-2, help='init lr')
    parser.add_argument('--unfreeze-bn', action='store_true', help='unfreeze bn')
    opt = parser.parse_args()

    init_seeds()

    train(
        opt.cfg,
        opt.data_cfg,
        weights_from=opt.weights_from,
        weights_to=opt.weights_to,
        save_every=opt.save_model_after,
        img_size=opt.img_size,
        resume=opt.resume,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulated_batches=opt.accumulated_batches,
        opt=opt,
    )
