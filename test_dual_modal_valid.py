"""
VALID数据集测试脚本

测试/评估双模态语义分割模型（RGB + Depth）
硬编码权重路径，可视化保存到GeoSeg目录
"""
import os
import sys
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
from geoseg.datasets.dual_modal_valid import VALIDDualModalDataset
from geoseg.datasets.transform import get_test_transform
from tools import Config, SegmentationMetric


# ==================== 硬编码配置 ====================
DATA_ROOT = '/home/lenovo/fsdownload/VALID'
# 权重路径（绝对路径）
# CHECKPOINT_PATH = '/home/lenovo/fsdownload/SFFNet-main/GeoSeg/log_VALID_mit_b2/checkpoint/best_model.pth'
CHECKPOINT_PATH = '/home/lenovo/fsdownload/DualModalWaveletMoE/work_dirs/log_VALID_mit_b2/checkpoint/best_model.pth'
# 可视化保存目录（在GeoSeg下）
# VIS_ROOT = '/home/lenovo/fsdownload/SFFNet-main/GeoSeg/eval_visualization'
VIS_ROOT = '/home/lenovo/fsdownload/DualModalWaveletMoE/eval_visualization'

PRED_COLOR_DIR = os.path.join(VIS_ROOT, 'prediction_color')
SUMMARY_DIR = os.path.join(VIS_ROOT, 'summary')

# 创建目录
os.makedirs(VIS_ROOT, exist_ok=True)
os.makedirs(PRED_COLOR_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)


# Vaihingen调色板
PALETTE = [
    [55, 181, 57],    # 0: background
    [89, 121, 72],    # 1: tree
    [190, 225, 64],   # 2: otherplant
    [206, 190, 59],   # 3: road
    [11, 236, 9],     # 4: pavement
    [153, 108, 6],    # 5: land
    [135, 169, 180],  # 6: water
    [115, 176, 195],  # 7: pool
    [112, 105, 191],  # 8: ice
    [0, 53, 65],      # 9: stones
    [211, 80, 208],   # 10: pierrubble
    [35, 196, 244],   # 11: bridge
    [81, 13, 36],     # 12: sign
    [196, 30, 8],     # 13: smallvehicle
    [205, 120, 161],  # 14: largevehicle
    [220, 163, 49],   # 15: building
    [146, 52, 70],    # 16: animal
    [242, 107, 146],  # 17: person
    [121, 67, 28],    # 18: chair
    [68, 218, 116],   # 19: fence
    [29, 26, 199],    # 20: garbagebin
    [54, 72, 205],    # 21: otherlowobstacle
    [226, 149, 143],  # 22: powerline
    [151, 126, 171],  # 23: trafficlight
    [103, 252, 157],  # 24: busstop
    [102, 16, 239],   # 25: otherhighobstacle
    [189, 135, 188],  # 26: lamp
    [161, 171, 27],   # 27: tunel
    [124, 21, 123],   # 28: ship
    [19, 132, 69],    # 29: plane
    [86, 254, 214]    # 30: harbor
]

CLASS_NAMES = [
    'background', 'tree', 'otherplant', 'road', 'pavement',
    'land', 'water', 'pool', 'ice', 'stones',
    'pierrubble', 'bridge', 'sign', 'smallvehicle', 'largevehicle',
    'building', 'animal', 'person', 'chair', 'fence',
    'garbagebin', 'otherlowobstacle', 'powerline', 'trafficlight', 'busstop',
    'otherhighobstacle', 'lamp', 'tunel', 'ship', 'plane',
    'harbor'
]


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


def create_summary_image(rgb, gt_color, pred_color, gap=20):
    """
    创建汇总图像：RGB | GT | Pred（带间隔）

    Args:
        rgb: (H, W, 3) RGB图像
        gt_color: (H, W, 3) GT彩色图
        pred_color: (H, W, 3) 预测彩色图
        gap: 图像间隔像素
    Returns:
        summary: (H, W_total, 3) 汇总图像
    """
    H, W = rgb.shape[:2]

    # 创建白色间隔
    gap_image = np.ones((H, gap, 3), dtype=np.uint8) * 255

    # 水平拼接：RGB | gap | GT | gap | Pred
    summary = np.concatenate([
        rgb,
        gap_image,
        gt_color,
        gap_image,
        pred_color
    ], axis=1)

    return summary


def build_model(checkpoint_path):
    """构建模型并加载权重"""
    # 加载配置
    cfg_path = project_root / 'config' / 'valid' / 'dual_modal_wavelet_moe.py'
    cfg = Config.fromfile(str(cfg_path))

    # 构建模型
    model = DualModalWaveletMoENet(
        rgb_channels=3,
        aux_channels=1,
        num_classes=cfg.num_classes,
        decode_channels=96,
        pretrained=False,
        wave='haar'
    )

    # 加载权重
    print(f"加载权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # 移除DDP的module.前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.cuda()
    model.eval()

    return model, cfg


@torch.no_grad()
def test(model, dataloader, cfg):
    """测试并生成可视化"""
    metric = SegmentationMetric(cfg.num_classes, cfg.ignore_index)

    print("\n开始测试...")
    for batch in tqdm(dataloader, desc='Testing'):
        img = batch['img'].cuda(non_blocking=True)
        depth = batch['depth'].cuda(non_blocking=True)
        target = batch['gt_semantic_seg'].cuda(non_blocking=True)
        img_name = batch['img_name'][0]

        # 前向传播
        output, _ = model(img, depth)
        pred = output.argmax(dim=1)

        # 更新指标
        metric.update(pred, target)

        # ===== 生成可视化 =====
        # 转为numpy
        pred_np = pred.cpu().numpy().squeeze()  # (H, W)
        target_np = target.cpu().numpy().squeeze()  # (H, W)
        img_np = img.cpu().numpy().squeeze().transpose(1, 2, 0)  # (H, W, 3)
        img_np = (img_np * 255).astype(np.uint8)

        # 1. 彩色化预测结果
        pred_color = colorize_mask(pred_np, PALETTE)

        # 保存预测可视化
        pred_color_path = os.path.join(PRED_COLOR_DIR, f"{img_name}_pred.png")
        cv2.imwrite(pred_color_path, cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

        # 2. 彩色化GT
        gt_color = colorize_mask(target_np, PALETTE)

        # 3. 创建汇总图像（RGB | GT | Pred，带间隔）
        summary_image = create_summary_image(img_np, gt_color, pred_color, gap=20)

        # 保存汇总图像
        summary_path = os.path.join(SUMMARY_DIR, f"{img_name}_summary.png")
        cv2.imwrite(summary_path, cv2.cvtColor(summary_image, cv2.COLOR_RGB2BGR))

    # ===== 打印评估结果 =====
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)

    metrics = metric.get_all_metrics()

    print(f"\n整体指标:")
    print(f"  mIoU:  {metrics['mIoU']:.4f} ({metrics['mIoU']*100:.2f}%)")
    print(f"  F1:    {metrics['F1']:.4f} ({metrics['F1']*100:.2f}%)")
    print(f"  OA:    {metrics['OA']:.4f} ({metrics['OA']*100:.2f}%)")

    print(f"\n各类别IoU:")
    print(f"{'类别':<25} {'IoU':>10}")
    print("-" * 37)

    for i, class_name in enumerate(CLASS_NAMES):
        iou = metrics['IoU_per_class'][i]
        print(f"{class_name:<25} {iou:>9.4f}")

    print("=" * 60 + "\n")

    # 保存结果到文本文件
    results_file = os.path.join(VIS_ROOT, 'test_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("VALID数据集测试结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"权重: {CHECKPOINT_PATH}\n")
        f.write(f"mIoU: {metrics['mIoU']:.4f} ({metrics['mIoU']*100:.2f}%)\n")
        f.write(f"F1:   {metrics['F1']:.4f} ({metrics['F1']*100:.2f}%)\n")
        f.write(f"OA:   {metrics['OA']:.4f} ({metrics['OA']*100:.2f}%)\n\n")

        f.write("各类别IoU:\n")
        for i, class_name in enumerate(CLASS_NAMES):
            f.write(f"  {class_name}: {metrics['IoU_per_class'][i]:.4f}\n")

    print(f"结果已保存到: {results_file}")

    return metrics


def main():
    print("=" * 60)
    print("VALID数据集测试")
    print("=" * 60)
    print(f"数据集路径: {DATA_ROOT}")
    print(f"权重路径: {CHECKPOINT_PATH}")
    print(f"可视化保存: {VIS_ROOT}")
    print("=" * 60)

    # 检查权重文件
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n错误：权重文件不存在: {CHECKPOINT_PATH}")
        print("请确认训练已完成，或修改CHECKPOINT_PATH")
        return

    # 构建模型
    model, cfg = build_model(CHECKPOINT_PATH)

    # 构建数据集
    print("\n加载测试数据集...")
    test_transform = get_test_transform(**cfg.test_transform)

    test_dataset = VALIDDualModalDataset(
        data_root=DATA_ROOT,
        mode='test',
        transform=test_transform,
        ignore_index=cfg.ignore_index
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"测试样本数: {len(test_dataset)}")

    # 测试
    metrics = test(model, test_loader, cfg)

    # 完成
    print("\n测试完成！")
    print(f"\n可视化文件保存位置:")
    print(f"  预测结果: {PRED_COLOR_DIR}")
    print(f"  汇总图像: {SUMMARY_DIR}")
    print(f"  评估结果: {os.path.join(VIS_ROOT, 'test_results.txt')}")


if __name__ == '__main__':
    main()