"""
双模态测试/评估脚本

支持：
- 整图推理
- 滑窗推理（大图）
- 多尺度测试（TTA）
- 测试时翻转增强
- 可视化保存
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import cv2

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import DualModalWaveletMoENet, SimpleDualModalNet
from geoseg.datasets import VaihingenDualModalDataset, get_test_transform
from tools import Config, SegmentationMetric


def parse_args():
    parser = argparse.ArgumentParser(description='Test Dual-Modal Segmentation Model')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--work-dir', help='working directory for outputs')
    parser.add_argument('--save-pred', action='store_true', help='save prediction masks')
    parser.add_argument('--save-vis', action='store_true', help='save visualization')
    parser.add_argument('--gpu-id', type=int, default=0, help='gpu id')

    args = parser.parse_args()
    return args


def build_model(cfg, checkpoint_path):
    """构建模型并加载权重"""
    model_cfg = cfg.model.copy()
    model_type = model_cfg.pop('type')

    # 禁用预训练加载（我们要加载checkpoint）
    model_cfg['pretrained'] = False

    if model_type == 'DualModalWaveletMoENet':
        model = DualModalWaveletMoENet(**model_cfg)
    elif model_type == 'SimpleDualModalNet':
        model = SimpleDualModalNet(**model_cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # 移除DDP的module.前缀（如果有）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.cuda()
    model.eval()

    print(f"Model loaded from: {checkpoint_path}")

    return model


def inference_whole(model, img, dsm):
    """
    整图推理

    Args:
        model: 模型
        img: (1, 3, H, W)
        dsm: (1, 1, H, W)
    Returns:
        pred: (1, H, W) 预测类别
    """
    with torch.no_grad():
        output, _ = model(img, dsm)
        pred = output.argmax(dim=1)

    return pred


def inference_sliding_window(model, img, dsm, crop_size=512, stride=256):
    """
    滑窗推理（用于大图）

    Args:
        model: 模型
        img: (1, 3, H, W)
        dsm: (1, 1, H, W)
        crop_size: 滑窗大小
        stride: 滑动步长
    Returns:
        pred: (1, H, W)
    """
    B, C, H, W = img.shape
    num_classes = model.num_classes if hasattr(model, 'num_classes') else 6

    # 输出累积
    output_sum = torch.zeros(B, num_classes, H, W).cuda()
    count_map = torch.zeros(B, 1, H, W).cuda()

    with torch.no_grad():
        for y in range(0, H - crop_size + 1, stride):
            for x in range(0, W - crop_size + 1, stride):
                # 裁剪
                img_crop = img[:, :, y:y+crop_size, x:x+crop_size]
                dsm_crop = dsm[:, :, y:y+crop_size, x:x+crop_size]

                # 推理
                output_crop, _ = model(img_crop, dsm_crop)

                # 累积
                output_sum[:, :, y:y+crop_size, x:x+crop_size] += output_crop
                count_map[:, :, y:y+crop_size, x:x+crop_size] += 1

        # 处理边界
        # 右边界
        if W > crop_size and (W - crop_size) % stride != 0:
            x = W - crop_size
            for y in range(0, H - crop_size + 1, stride):
                img_crop = img[:, :, y:y+crop_size, x:x+crop_size]
                dsm_crop = dsm[:, :, y:y+crop_size, x:x+crop_size]
                output_crop, _ = model(img_crop, dsm_crop)
                output_sum[:, :, y:y+crop_size, x:x+crop_size] += output_crop
                count_map[:, :, y:y+crop_size, x:x+crop_size] += 1

        # 下边界
        if H > crop_size and (H - crop_size) % stride != 0:
            y = H - crop_size
            for x in range(0, W - crop_size + 1, stride):
                img_crop = img[:, :, y:y+crop_size, x:x+crop_size]
                dsm_crop = dsm[:, :, y:y+crop_size, x:x+crop_size]
                output_crop, _ = model(img_crop, dsm_crop)
                output_sum[:, :, y:y+crop_size, x:x+crop_size] += output_crop
                count_map[:, :, y:y+crop_size, x:x+crop_size] += 1

        # 右下角
        if H > crop_size and W > crop_size:
            if (W - crop_size) % stride != 0 and (H - crop_size) % stride != 0:
                y = H - crop_size
                x = W - crop_size
                img_crop = img[:, :, y:y+crop_size, x:x+crop_size]
                dsm_crop = dsm[:, :, y:y+crop_size, x:x+crop_size]
                output_crop, _ = model(img_crop, dsm_crop)
                output_sum[:, :, y:y+crop_size, x:x+crop_size] += output_crop
                count_map[:, :, y:y+crop_size, x:x+crop_size] += 1

    # 平均
    output_avg = output_sum / count_map
    pred = output_avg.argmax(dim=1)

    return pred


def inference_multi_scale(model, img, dsm, scales=[0.75, 1.0, 1.25]):
    """
    多尺度测试（TTA）

    Args:
        model: 模型
        img: (1, 3, H, W)
        dsm: (1, 1, H, W)
        scales: 测试尺度列表
    Returns:
        pred: (1, H, W)
    """
    B, C, H, W = img.shape
    num_classes = model.num_classes if hasattr(model, 'num_classes') else 6

    output_sum = torch.zeros(B, num_classes, H, W).cuda()

    with torch.no_grad():
        for scale in scales:
            # Resize
            new_h, new_w = int(H * scale), int(W * scale)
            img_scaled = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
            dsm_scaled = F.interpolate(dsm, size=(new_h, new_w), mode='bilinear', align_corners=False)

            # 推理
            output_scaled, _ = model(img_scaled, dsm_scaled)

            # Resize回原尺寸
            output_scaled = F.interpolate(output_scaled, size=(H, W), mode='bilinear', align_corners=False)

            output_sum += output_scaled

    # 平均
    output_avg = output_sum / len(scales)
    pred = output_avg.argmax(dim=1)

    return pred


def colorize_mask(mask, palette):
    """
    将类别mask转为彩色可视化

    Args:
        mask: (H, W) numpy array
        palette: list of [R, G, B]
    Returns:
        color_mask: (H, W, 3) uint8
    """
    H, W = mask.shape
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)

    for label, color in enumerate(palette):
        color_mask[mask == label] = color

    return color_mask


@torch.no_grad()
def test(model, dataloader, cfg, args):
    """测试"""
    metric = SegmentationMetric(cfg.num_classes, cfg.ignore_index)

    # Vaihingen调色板
    palette = [
        [255, 255, 255],  # impervious_surface - 白色
        [0, 0, 255],      # building - 蓝色
        [0, 255, 255],    # low_vegetation - 青色
        [0, 255, 0],      # tree - 绿色
        [255, 255, 0],    # car - 黄色
        [255, 0, 0]       # clutter - 红色
    ]

    # 输出目录
    if args.save_pred or args.save_vis:
        pred_dir = os.path.join(args.work_dir, 'predictions')
        vis_dir = os.path.join(args.work_dir, 'visualizations')
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

    # 测试配置
    test_cfg = cfg.get('test_cfg', {})
    mode = test_cfg.get('mode', 'whole')
    multi_scale = test_cfg.get('multi_scale', False)
    flip = test_cfg.get('flip', False)

    print(f"\nTesting with mode: {mode}")
    if multi_scale:
        print(f"Multi-scale: {test_cfg.get('scales', [0.75, 1.0, 1.25])}")
    if flip:
        print("Test-time flip augmentation: enabled")

    for batch in tqdm(dataloader, desc='Testing'):
        img = batch['img'].cuda()
        dsm = batch['dsm'].cuda()
        target = batch['gt_semantic_seg'].cuda()
        img_name = batch['img_name'][0]

        # 推理
        if mode == 'whole':
            if multi_scale:
                pred = inference_multi_scale(model, img, dsm, test_cfg.get('scales', [0.75, 1.0, 1.25]))
            else:
                pred = inference_whole(model, img, dsm)
        elif mode == 'sliding_window':
            pred = inference_sliding_window(
                model, img, dsm,
                test_cfg.get('crop_size', 512),
                test_cfg.get('stride', 256)
            )
        else:
            raise ValueError(f"Unknown test mode: {mode}")

        # 测试时翻转增强
        if flip:
            # 水平翻转
            img_flip = torch.flip(img, dims=[3])
            dsm_flip = torch.flip(dsm, dims=[3])
            pred_flip = inference_whole(model, img_flip, dsm_flip)
            pred_flip = torch.flip(pred_flip, dims=[2])

            # 合并（投票）
            pred = torch.stack([pred, pred_flip], dim=0).mode(dim=0)[0]

        # 更新指标
        metric.update(pred, target)

        # 保存预测
        if args.save_pred:
            pred_np = pred.cpu().numpy().squeeze()
            pred_path = os.path.join(pred_dir, f"{img_name}.png")
            cv2.imwrite(pred_path, pred_np.astype(np.uint8))

        # 保存可视化
        if args.save_vis:
            pred_np = pred.cpu().numpy().squeeze()
            target_np = target.cpu().numpy().squeeze()

            # 彩色化
            pred_color = colorize_mask(pred_np, palette)
            target_color = colorize_mask(target_np, palette)

            # RGB原图
            img_np = img.cpu().numpy().squeeze().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)

            # 拼接
            vis = np.concatenate([img_np, target_color, pred_color], axis=1)

            vis_path = os.path.join(vis_dir, f"{img_name}_vis.png")
            cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # 打印评估结果
    print("\n" + "="*60)
    print("Test Results")
    print("="*60)

    metrics = metric.summary(class_names=cfg.get('classes'))

    return metrics


def main():
    args = parse_args()

    # 加载配置
    cfg = Config.fromfile(args.config)

    # 工作目录
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = os.path.join(
            os.path.dirname(args.checkpoint),
            'test_results'
        )

    os.makedirs(cfg.work_dir, exist_ok=True)

    # 设置GPU
    torch.cuda.set_device(args.gpu_id)

    # 构建模型
    print("Loading model...")
    model = build_model(cfg, args.checkpoint)

    # 构建测试数据集
    print("Building test dataset...")
    transform = get_test_transform(**cfg.test_transform)
    test_dataset = VaihingenDualModalDataset(
        data_root=cfg.data_root,
        img_dir=cfg.test_img_dir,
        dsm_dir=cfg.test_dsm_dir,
        ann_dir=cfg.test_ann_dir,
        transform=transform,
        mode='val',  # 有标签的测试集
        ignore_index=cfg.ignore_index
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 测试
    metrics = test(model, test_loader, cfg, args)

    # 保存结果
    results_file = os.path.join(cfg.work_dir, 'test_results.txt')
    with open(results_file, 'w') as f:
        f.write("Test Results\n")
        f.write("="*60 + "\n")
        f.write(f"mIoU: {metrics['mIoU']:.4f} ({metrics['mIoU']*100:.2f}%)\n")
        f.write(f"F1:   {metrics['F1']:.4f} ({metrics['F1']*100:.2f}%)\n")
        f.write(f"OA:   {metrics['OA']:.4f} ({metrics['OA']*100:.2f}%)\n\n")

        f.write("Per-class IoU:\n")
        for i, class_name in enumerate(cfg.get('classes', [])):
            f.write(f"  {class_name}: {metrics['IoU_per_class'][i]:.4f}\n")

    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()