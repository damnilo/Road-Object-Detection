import os
import random
import argparse
from datetime import datetime

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from src.config.configs import FINE_GRID_SIZE, GRID_SIZE, NUM_CLASSES
from data.bdd100k_dataset import BDD100KDataset
from src.data.splitter import sequence_aware_split
from src.models.detector import Detector
from src.models.loss import DetectionLoss
from src.training.eval import evaluate
from src.training.logger import TrainingLogger
from src.visualization.targets import save_augmented_target_visualizations


def maybe_compile_model(model):
    if os.name == 'nt' and os.environ.get('ENABLE_TORCH_COMPILE') != '1':
        print('Warning: skipping torch.compile() on Windows; set ENABLE_TORCH_COMPILE=1 to opt in.')
        return model

    try:
        return torch.compile(model)
    except Exception as exc:
        print(f'Warning: skipping torch.compile() because it is unavailable for this environment: {exc}')
        return model


def build_run_log_path(base_log_path):
    if not os.path.exists(base_log_path):
        return base_log_path

    root, ext = os.path.splitext(base_log_path)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    candidate = f'{root}_{timestamp}{ext}'
    if not os.path.exists(candidate):
        return candidate

    run_index = 1
    while True:
        candidate = f'{root}_{timestamp}_run{run_index:02d}{ext}'
        if not os.path.exists(candidate):
            return candidate
        run_index += 1


def select_indices(total_count, sample_count, seed):
    indices = list(range(total_count))
    random.Random(seed).shuffle(indices)
    return indices[:min(sample_count, total_count)]


def main(epochs=40, batch_size=8, seed=42, resume_path=None, overfit_samples=0,
         visualize_targets=0, target_visualization_dir='diagnostics/targets', log_file=None,
         use_weighted_sampling=True):
    os.environ.setdefault('OMP_NUM_THREADS', str(os.cpu_count() // 2 or 1))
    os.environ.setdefault('MKL_NUM_THREADS', str(os.cpu_count() // 2 or 1))

    torch.set_num_threads(os.cpu_count() // 2 or 1)
    torch.set_num_interop_threads(os.cpu_count() // 2 or 1)
    os.makedirs('checkpoints', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    overfit_mode = overfit_samples > 0
    base_dataset = BDD100KDataset('dataset/images/train', 'dataset/labels/train', augment=not overfit_mode)

    if overfit_mode:
        indices = list(range(len(base_dataset)))
        random.shuffle(indices)
        selected_indices = indices[:min(overfit_samples, len(indices))]
        train_dataset = Subset(base_dataset, selected_indices)
        val_dataset = Subset(
            BDD100KDataset('dataset/images/train', 'dataset/labels/train', augment=False),
            selected_indices,
        )
        class_counts = base_dataset.class_counts(selected_indices)
    else:
        train_idx, val_idx = sequence_aware_split(base_dataset, seed=seed)
        train_dataset = Subset(base_dataset, train_idx)
        val_dataset = Subset(
            BDD100KDataset('dataset/images/train', 'dataset/labels/train', augment=False),
            val_idx,
        )
        class_counts = base_dataset.class_counts(train_idx)

    print(f"train_dataset size: {len(train_dataset)}, val_dataset size: {len(val_dataset)}")
        
    loader_generator = torch.Generator().manual_seed(seed)

    if overfit_mode:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  generator=loader_generator,
                                  collate_fn=BDD100KDataset.detection_collate)
    elif use_weighted_sampling:
        train_weights = base_dataset.sample_weights(train_idx, class_counts=class_counts)
        sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_idx),
            replacement=True,
            generator=loader_generator,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                  sampler=sampler,
                                  collate_fn=BDD100KDataset.detection_collate)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  generator=loader_generator,
                                  collate_fn=BDD100KDataset.detection_collate)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=BDD100KDataset.detection_collate)

    model = Detector(num_classes=NUM_CLASSES).to(device)
    model = maybe_compile_model(model)
    criterion = DetectionLoss(num_classes=NUM_CLASSES, class_counts=class_counts).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-5)
    best_map = float('-inf')
    start_epoch = 0
    log_path = log_file or 'logs/training_log.csv'
    append_log = False
    patience = 0

    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        if 'model_state_dict' not in checkpoint:
            raise ValueError("Resume checkpoint must contain model and optimizer state")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_map = checkpoint['best_map']
        start_epoch = checkpoint['epoch']
        log_path = checkpoint.get('log_file', log_path)
        append_log = os.path.exists(log_path)

    if not resume_path:
        log_path = build_run_log_path(log_path)

    logger = TrainingLogger(log_dir=log_path, num_classes=NUM_CLASSES, append=append_log)

    if visualize_targets > 0:
        save_augmented_target_visualizations(
            image_dir='dataset/images/train',
            label_dir='dataset/labels/train',
            output_dir=target_visualization_dir,
            sample_count=visualize_targets,
            seed=seed,
            augment=not overfit_mode,
        )

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        epoch_stats = {'coord': 0.0, 'obj': 0.0, 'noobj': 0.0, 'class': 0.0}

        for batch_idx, (images, targets, _, _) in enumerate(train_loader):
            images = images.to(device)
            targets = {scale: target.to(device) for scale, target in targets.items()}
            optimizer.zero_grad()
            loss, loss_stats = criterion(model(images), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            for key in epoch_stats:
                epoch_stats[key] += loss_stats[key]

            if (batch_idx + 1) % 100 == 0:
                print(f'Batch {batch_idx + 1}/{len(train_loader)}: loss={loss.item():.4f}, Avg Loss: {total_loss / (batch_idx + 1):.4f}')

        if not len(train_loader):
            raise RuntimeError("Training dataset is empty")

        avg_loss = total_loss / len(train_loader)
        avg_stats = {key: value / len(train_loader) for key, value in epoch_stats.items()}
        mean_ap, aps = evaluate(
            model, val_loader, grid_sizes={'fine': FINE_GRID_SIZE, 'coarse': GRID_SIZE}, num_classes=NUM_CLASSES,
            device=device, conf_thresh=0.001,
        )
        logger.log_epoch(epoch + 1, avg_loss, avg_stats, mean_ap,
                         optimizer.param_groups[0]['lr'], aps)
        print(f'Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, mAP={mean_ap:.4f}')

        if mean_ap > best_map:
            patience = 0
            best_map = mean_ap
            torch.save(model.state_dict(), 'checkpoints/best_detector_weights.pth')
        else:
            patience += 1
            if patience >= 7:
                print("Early stopping triggered due to no improvement in mAP for 7 consecutive epochs.")
                break

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_map': best_map,
            'log_file': logger.log_dir,
        }, 'checkpoints/latest.pth')

        scheduler.step()

    torch.save(model.state_dict(), 'checkpoints/detector_weights.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the grid-based object detector.')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', help='Path to a checkpoints/latest.pth file to resume.')
    parser.add_argument('--overfit-samples', type=int, default=0,
                        help='Train on a small fixed subset with augmentation disabled.')
    parser.add_argument('--visualize-targets', type=int, default=0,
                        help='Save this many augmented target visualizations before training.')
    parser.add_argument('--target-visualization-dir', default='diagnostics/targets',
                        help='Directory for augmented target visualizations.')
    parser.add_argument('--log-file', default=None,
                        help='Base CSV path for the training log. Fresh runs still get unique files.')
    parser.add_argument('--weighted-sampling', dest='use_weighted_sampling', action='store_true',
                        help='Use class-aware image sampling during normal training (default).')
    parser.add_argument('--no-weighted-sampling', dest='use_weighted_sampling', action='store_false',
                        help='Disable class-aware image sampling and use plain shuffling instead.')
    parser.set_defaults(use_weighted_sampling=True)
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.seed, args.resume, args.overfit_samples,
         args.visualize_targets, args.target_visualization_dir, args.log_file,
         args.use_weighted_sampling)
