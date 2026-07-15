import os
import random
import argparse

import torch
from torch.utils.data import DataLoader, Subset

from src.config.configs import GRID_SIZE, NUM_CLASSES
from src.data.kitti_dataset import KITTIDataset
from src.models.detector import Detector
from src.models.loss import DetectionLoss
from src.training.eval import evaluate
from src.training.logger import TrainingLogger


def main(epochs=25, batch_size=4, seed=42, resume_path=None):
    torch.set_num_threads(os.cpu_count() or 1)
    os.makedirs('checkpoints', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    base_dataset = KITTIDataset('dataset/images/train', 'dataset/labels/train', augment=True)
    indices = list(range(len(base_dataset)))
    random.shuffle(indices)
    split = int(0.8 * len(indices))

    train_dataset = Subset(base_dataset, indices[:split])
    val_dataset = Subset(
        KITTIDataset('dataset/images/train', 'dataset/labels/train', augment=False),
        indices[split:],
    )
    class_counts = base_dataset.class_counts(indices[:split])
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              generator=loader_generator,
                              collate_fn=KITTIDataset.detection_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=KITTIDataset.detection_collate)

    model = Detector(num_classes=NUM_CLASSES).to(device)
    criterion = DetectionLoss(num_classes=NUM_CLASSES, class_counts=class_counts).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    logger = TrainingLogger(log_dir='logs/training_log.csv', num_classes=NUM_CLASSES)
    best_map = float('-inf')
    start_epoch = 0

    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        if 'model_state_dict' not in checkpoint:
            raise ValueError("Resume checkpoint must contain model and optimizer state")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_map = checkpoint['best_map']
        start_epoch = checkpoint['epoch']

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        epoch_stats = {'coord': 0.0, 'obj': 0.0, 'noobj': 0.0, 'class': 0.0}

        for batch_idx, (images, targets, _, _) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss, loss_stats = criterion(model(images), targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            for key in epoch_stats:
                epoch_stats[key] += loss_stats[key]

            if (batch_idx + 1) % 100 == 0:
                print(f'Batch {batch_idx + 1}/{len(train_loader)}: loss={loss.item():.4f}, Avg Loss: {total_loss / (batch_idx + 1):.4f}')

        scheduler.step()
        if not len(train_loader):
            raise RuntimeError("Training dataset is empty")

        avg_loss = total_loss / len(train_loader)
        avg_stats = {key: value / len(train_loader) for key, value in epoch_stats.items()}
        mean_ap, aps = evaluate(
            model, val_loader, grid_size=GRID_SIZE, num_classes=NUM_CLASSES,
            device=device, conf_thresh=0.001,
        )
        logger.log_epoch(epoch + 1, avg_loss, avg_stats, mean_ap,
                         optimizer.param_groups[0]['lr'], aps)
        print(f'Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, mAP={mean_ap:.4f}')

        if mean_ap > best_map:
            best_map = mean_ap
            torch.save(model.state_dict(), 'checkpoints/best_detector_weights.pth')

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_map': best_map,
        }, 'checkpoints/latest.pth')

    torch.save(model.state_dict(), 'checkpoints/detector_weights.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the grid-based object detector.')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', help='Path to a checkpoints/latest.pth file to resume.')
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.seed, args.resume)
