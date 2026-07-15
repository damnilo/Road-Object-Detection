import os
import csv

class TrainingLogger:

    def __init__(self, log_dir='logs/training_log.csv', num_classes=None):
        self.log_dir = log_dir
        self.num_classes = num_classes
        os.makedirs(os.path.dirname(log_dir), exist_ok=True)

        self.field_names = [
            'epoch', 'train_loss', 'coord_loss', 'obj_loss',
            'noobj_loss', 'class_loss', 'val_map', 'lr'
        ]

        if num_classes is not None:
            self.field_names += [f'class_{i}_ap' for i in range(num_classes)]

        if os.path.exists(log_dir):
            with open(log_dir, newline='') as f:
                existing_fields = next(csv.reader(f), [])
            if existing_fields != self.field_names:
                root, ext = os.path.splitext(log_dir)
                self.log_dir = f'{root}_v2{ext}'

        if not os.path.exists(self.log_dir):
            with open(self.log_dir, 'w') as f:
                writer  = csv.DictWriter(f, fieldnames=self.field_names)
                writer.writeheader()

    def log_epoch(self, epoch, train_loss, loss_stats, val_map, lr, aps):
        log_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'coord_loss': loss_stats['coord'],
            'obj_loss': loss_stats['obj'],
            'noobj_loss': loss_stats['noobj'],
            'class_loss': loss_stats['class'],
            'val_map': val_map,
            'lr': lr
        }

        if self.num_classes is not None:
            for i in range(self.num_classes):
                log_data[f'class_{i}_ap'] = aps.get(i, float('nan'))

        with open(self.log_dir, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.field_names)
            writer.writerow(log_data)
