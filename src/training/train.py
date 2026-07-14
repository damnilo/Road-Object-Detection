import os
import torch
import random
import time
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from src.config.configs import NUM_CLASSES
from src.models.loss import DetectionLoss
from src.models.detector import Detector
from src.data.kitti_dataset import KITTIDataset
from src.training.eval import evaluate

torch.set_num_threads(os.cpu_count())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_counts = [23026, 2359, 871, 3566, 186, 1275, 394, 765]

base_dataset = KITTIDataset(image_dir='dataset/images/train', label_dir='dataset/labels/train', augment=True)
print(f"Loaded dataset with {len(base_dataset)} samples.")

indices = list(range(len(base_dataset)))
random.seed(42)
random.shuffle(indices)

split = int(0.8 * len(indices))
train_indices = indices[:split]
val_indices = indices[split:]

train_dataset = Subset(base_dataset, train_indices)
val_dataset = Subset(KITTIDataset(image_dir='dataset/images/train', label_dir='dataset/labels/train', augment=False), val_indices)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=KITTIDataset.detection_collate)
print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=KITTIDataset.detection_collate)
print("Loaded Validation DataLoader")

model = Detector(num_classes=NUM_CLASSES).to(device)
criterion = DetectionLoss(num_classes=NUM_CLASSES, class_counts=class_counts).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
print("Initialized model, loss function, and optimizer.")

EPOCHS = 25
best_map = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, targets, _, _) in enumerate(train_loader):

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss, loss_stats = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f"Loss [{loss.item():.4f}], Avg Loss: {total_loss / (batch_idx + 1):.4f}," 
                  f"Batch [{batch_idx+1}/{len(train_loader)}]",
                  f"coord: {loss_stats['coord']:.4f}, obj: {loss_stats['obj']:.4f}, "
                  f"noobj: {loss_stats['noobj']:.4f}, class: {loss_stats['class']:.4f}")
            
    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    
    mean_ap, aps = evaluate(model, val_loader, grid_size=7, num_classes=NUM_CLASSES, device=device)

    per_class = ",  ".join([f"{idx}: {cls:.4f}" for idx, cls in aps.items()])
    print(f"Validation mAP: {mean_ap:.4f}, APs: {per_class}")

    if mean_ap > best_map:
        best_map = mean_ap
        torch.save(model.state_dict(), 'checkpoints/best_detector_weights.pth')
        print(f"New best model saved with mAP: {best_map:.4f}")

torch.save(model.state_dict(), 'checkpoints/detector_weights.pth')