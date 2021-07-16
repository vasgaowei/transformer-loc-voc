"""
normal: fine-tune learning rate 0.1

parameters: --config_file ../configs/CUB/cub_vgg16_cam.yaml BASIC.GPU_ID [0]
"""
import os
import sys
import datetime
import pprint

import _init_paths
from config.default import cfg_from_list, cfg_from_file, update_config
from config.default import config as cfg
from core.engine import creat_data_loader, str_gpus, \
    AverageMeter, accuracy, list2acc, adjust_learning_rate_normal
from core.functions import prepare_env
from utils import mkdir, Logger
from cams_deit import resize_cam, get_bboxes
from core.cls_eval import AveragePrecisionMeter

import torch
from torch.utils.tensorboard import SummaryWriter

from models.vgg import vgg16_cam
from timm.models import create_model as create_deit_model
from timm.optim import create_optimizer
import numpy as np

def creat_model(cfg, args):
    print('==> Preparing networks for baseline...')
    # use gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = str_gpus(cfg.BASIC.GPU_ID)
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "CUDA is not available"
    # model and optimizer
    model = create_deit_model(
            cfg.MODEL.ARCH,
            pretrained=True,
            num_classes=cfg.DATA.NUM_CLASSES,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
        )
    print(model)
    optimizer = create_optimizer(args, model)

    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).to(device)
    # loss
    cls_criterion = torch.nn.BCEWithLogitsLoss().to(device)
    print('Preparing networks done!')
    return device, model, optimizer, cls_criterion


def main():
    args = update_config()

    # create checkpoint directory
    cfg.BASIC.SAVE_DIR = os.path.join('ckpt', cfg.DATA.DATASET, '{}_CAM-NORMAL_SEED{}_CAM-THR{}_BS{}_{}'.format(
        cfg.MODEL.ARCH, cfg.BASIC.SEED, cfg.MODEL.CAM_THR, cfg.TRAIN.BATCH_SIZE, cfg.BASIC.TIME))
    cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    log_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'log'); mkdir(log_dir)
    ckpt_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'ckpt'); mkdir(ckpt_dir)
    cache_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'corloc_cache'); mkdir(ckpt_dir)
    log_file = os.path.join(cfg.BASIC.SAVE_DIR, 'Log_' + cfg.BASIC.TIME + '.txt')
    # prepare running environment for the whole project
    prepare_env(cfg)

    # start loging
    sys.stdout = Logger(log_file)
    pprint.pprint(cfg)
    writer = SummaryWriter(log_dir)

    train_loader, val_loader = creat_data_loader(cfg, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATA.DATADIR))
    device, model, optimizer, cls_criterion = creat_model(cfg, args)

    best_gtknown = 0
    best_top1_loc = 0
    best_corloc = 0
    update_train_step = 0
    update_val_step = 0
    for epoch in range(1, cfg.SOLVER.NUM_EPOCHS+1):
        adjust_learning_rate_normal(optimizer, epoch, cfg)
        update_train_step, loss_train = \
            train_one_epoch(train_loader, model, device, cls_criterion,
                            optimizer, epoch, writer, cfg, update_train_step)

        torch.save({
            "epoch": epoch,
            'state_dict': model.state_dict(),
            'best_map': best_gtknown
        }, os.path.join(ckpt_dir, 'model_epoch{}.pth'.format(epoch)))


        update_val_step, loss_val, corloc = val_loc_one_epoch(val_loader, model, device, cls_criterion, epoch, writer, cfg, update_val_step, cache_dir)

        torch.save({
            "epoch": epoch,
            'state_dict': model.state_dict(),
            'best_map': best_gtknown
        }, os.path.join(ckpt_dir, 'model_epoch{}.pth'.format(epoch)))

        if corloc > best_corloc:
            best_corloc = corloc
            torch.save({
                "epoch": epoch,
                'state_dict': model.state_dict(),
                'best_map': best_gtknown
            }, os.path.join(ckpt_dir, 'model_best.pth'))


        print("Best CorLoc_LOC: {}".format(best_corloc))
        print(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))


def train_one_epoch(train_loader, model, device, criterion, optimizer, epoch,
                    writer, cfg, update_train_step):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    ap_meter = AveragePrecisionMeter()
    ap_meter.reset()

    model.train()
    for i, (input, target) in enumerate(train_loader):
        # update iteration steps
        update_train_step += 1

        target = target.to(device)
        input = input.to(device)

        cls_logits = model(input, return_cam=False)

        loss = criterion(cls_logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), input.size(0))
        ap_meter.add(cls_logits.data, target.data)

        writer.add_scalar('loss_iter/train', loss.item(), update_train_step)


        if i % cfg.BASIC.DISP_FREQ == 0 or i == len(train_loader)-1:
            if isinstance(criterion, list):
                print(('Train Epoch: [{0}][{1}/{2}],lr: {lr:.5f}\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'MAP {map.value().numpy().mean():.3f}\t'.format(
                    epoch, i + 1, len(train_loader), loss=losses, map=ap_meter, lr=optimizer.param_groups[-1]['lr'])))
            else:
                print(('Train Epoch: [{0}][{1}/{2}],lr: {lr:.5f}\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'MAP {map.val:.3f}\t'.format(
                    epoch, i + 1, len(train_loader), loss=losses, map=ap_meter, lr=optimizer.param_groups[-1]['lr'])))
    ap = ap_meter.value().numpy()
    print('the classification AP is ')
    for index, cls in enumerate(range(cfg.DATA.NUM_CLASSES)):
        print(('AP for {} = {:.4f}'.format(cls, ap[index])))
    print('__________________')
    map = 100 * ap.mean()
    print('the mAP is {:.4f}'.format(map))
    if isinstance(criterion, list):
        return update_train_step, [losses.avg, ], [top1.avg, ], [top5.avg, ]
    return update_train_step, losses.avg


def val_loc_one_epoch(val_loader, model, device, criterion,epoch, writer, cfg, update_val_step, cache_dir):

    losses = AverageMeter()
    ap_meter = AveragePrecisionMeter()
    ap_meter.reset()
    num_images = len(val_loader.dataset)
    img_id = -1

    loc_all_boxes = [[[] for _ in range(num_images)]
               for _ in range(cfg.DATA.NUM_CLASSES)]


    with torch.no_grad():
        model.eval()
        for i, (input, target, gt_box, image_sizes, image_names, image_path_ids) in enumerate(val_loader):
            # update iteration steps
            update_val_step += 1

            target = target.to(device)
            input = input.to(device)

            # Get localization maps
            # cams: B X C X 14 X 14
            cls_logits, cams = model(input, return_cam=True)
            loss = criterion(cls_logits, target)

            losses.update(loss.item(), input.size(0))
            ap_meter.add(cls_logits.data, target.data)
            writer.add_scalar('loss_iter/val', loss.item(), update_val_step)

            for batch_i in range(cams.shape[0]):
                image_size_i = image_sizes[batch_i]
                image_name_i = image_names[batch_i]
                image_path_i = image_path_ids[batch_i]

                image_score_i = cls_logits[batch_i].data.cpu().numpy()
                img_id += 1
                for class_i in range(cfg.DATA.NUM_CLASSES):
                    cam_i = cams[batch_i, [class_i], :, :]
                    cam_i = torch.mean(cam_i, dim=0, keepdim=True)
                    cam_i = cam_i.detach().cpu().numpy().transpose(1, 2, 0)
                    # Resize and Normalize CAM
                    #cam_i = resize_cam(cam_i, size=(cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE))
                    cam_i = resize_cam(cam_i, size=image_size_i)
                    # Estimate BBOX
                    bbox = get_bboxes(cam_i, cam_thr=cfg.MODEL.CAM_THR)
                    bbox_score = np.hstack((np.array(bbox).reshape(1, -1), image_score_i[class_i].reshape(1, -1)))
                    #bbox_score = np.hstack(bbox_score, np.array([image_path_i]).reshape(1, -1))
                    loc_all_boxes[class_i][img_id] = bbox_score.copy()

            if i % cfg.BASIC.DISP_FREQ == 0 or i == len(val_loader)-1:
                print('Val Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i+1, len(val_loader), loss=losses))
                print('MAP {map.val:.3f}\n'.format(
                    map=ap_meter
                ))
        print(img_id)
        print('Evaluating ClsMap')
        ap = ap_meter.value().numpy()
        for index, cls in enumerate(range(cfg.DATA.NUM_CLASSES)):
            print(('AP for {} = {:.4f}'.format(cls, ap[index])))
        print('__________________')
        map = 100 * ap.mean()
        print('the mAP is {:.4f}'.format(map))
        print('Evaluating CorLoc')
        mAP = val_loader.dataset.evaluate_discovery(loc_all_boxes, cache_dir)

    return update_val_step, losses.avg, mAP
if __name__ == "__main__":
    main()
