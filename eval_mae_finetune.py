import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import json
import os
import random
import os.path as osp
from PIL import Image
from pathlib import Path
from timm.models import create_model
import utils
import modeling_pretrain
from datasets import CelebA, DataAugmentationForMAE
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from masking_generator import RandomMaskingGenerator
from tqdm import tqdm
from facenet import FaceNet152


def get_args():
    parser = argparse.ArgumentParser('MAE visualization reconstruction script')
    parser.add_argument('--img_path', default='files/ILSVRC2012_val_00031649.JPEG', type=str, help='input image path')
    parser.add_argument('--save_path', default='output/', type=str, help='save image path')
    parser.add_argument('--pretrain_model_path', default='checkpoints/mae_pretrain.pth', type=str, help='checkpoint path of pre-trained model')
    parser.add_argument('--finetune_model_path', default='output/checkpoint-best.pth', type=str, help='checkpoint path of fine-tuned model')

    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='pretrain_mae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    return model


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@torch.no_grad()
def calc_Itop1(I, rec_data, gt_label, scale_fnI):
    if scale_fnI is not None:
        high_rec_data = scale_fnI(rec_data)
    else: high_rec_data = rec_data
    
    _, pred_outputs, _ = I(high_rec_data)
    pred_class_idx = torch.argmax(pred_outputs, dim=1)
    correct_top1 = pred_class_idx[pred_class_idx==gt_label].shape[0]
    correct_top1 /= rec_data.shape[0]

    pred_class_top5 = torch.topk(pred_outputs, k=5, dim=-1).indices
    gt_class_idx = gt_label.unsqueeze(-1).repeat(1, 5)
    correct_top5 = int(torch.sum((torch.sum((gt_class_idx == pred_class_top5), dim=1) > 0), dim=0))
    correct_top5 /= rec_data.shape[0]
    
    return correct_top1, correct_top5


def main(args):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    g = torch.Generator()
    g.manual_seed(0)

    print(args)
    os.makedirs(args.save_path, exist_ok=True)

    device = torch.device(args.device)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    pretrain_checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')
    model.load_state_dict(pretrain_checkpoint['model'])
    finetune_checkpoint = torch.load('output/pretrain/checkpoint-299.pth', map_location='cpu')['model']
    finetune_checkpoint_cleaned = finetune_checkpoint.copy()
    for key in finetune_checkpoint.keys():
        if not key.startswith('encoder'):
            del finetune_checkpoint_cleaned[key]
    utils.load_state_dict(model, finetune_checkpoint_cleaned)

    # finetune_checkpoint = torch.load(args.finetune_model_path, map_location='cpu')
    # # loading the encoder as the fine-tuned model
    # finetune_checkpoint = finetune_checkpoint['model']
    # for k in ['head.weight', 'head.bias']:
    #     if k in finetune_checkpoint:
    #         print(f"Removing key {k} from finetuned checkpoint")
    #         del finetune_checkpoint[k]

    # utils.load_state_dict(model.encoder, finetune_checkpoint)
    model.eval()
    print('model loaded successfully')

    I = FaceNet152(num_classes=1000).to(device)
    I.load_state_dict(torch.load(osp.join('checkpoints', f'ir152-pri.pth'))['state_dict'])
    I.eval()

    priset = CelebA(split='pri')
    priloader = DataLoader(priset, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker, num_workers=4, generator=g)
    
    img, _ = next(iter(priloader))
    masked_position_generator = RandomMaskingGenerator(args.window_size, args.mask_ratio)

    top1_total = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(priloader)
        for img, gt_label in pbar:
            img = img.to(device)
            gt_label = gt_label.to(device)
            bool_masked_pos = masked_position_generator()
            bool_masked_pos = torch.from_numpy(bool_masked_pos).unsqueeze(0).repeat(args.batch_size, 1)
            bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).to(torch.bool)
            outputs = model(img, bool_masked_pos)

            ori_img = img.clone()
            img_squeeze = rearrange(ori_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size[0], p2=patch_size[0])
            img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
            img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
            for i in range(args.batch_size):
                img_patch[i][bool_masked_pos[i]] = outputs[i]

            #make mask
            mask = torch.ones_like(img_patch)
            mask[bool_masked_pos] = 0
            mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
            mask = rearrange(mask, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=14, w=14)

            #save reconstruction img
            rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
            rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(dim=-2, keepdim=True)
            rec_img = rearrange(rec_img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=14, w=14)

            top1, top5 = calc_Itop1(I, rec_img, gt_label, utils.high2low112)
            top1_total += (top1 * rec_img.shape[0])
            total += rec_img.shape[0]
            acc = top1_total / total

            pbar.set_description('Acc: {:.5f}'.format(acc))



if __name__ == '__main__':
    opts = get_args()
    main(opts)
