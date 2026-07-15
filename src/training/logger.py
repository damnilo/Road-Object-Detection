import os
import csv


def _next_available_log_path(log_dir):
    if not os.path.exists(log_dir):
        return log_dir

    root, ext = os.path.splitext(log_dir)
    run_index = 1
    while True:
        candidate = f'{root}_run{run_index:02d}{ext}'
        if not os.path.exists(candidate):
            return candidate
        run_index += 1

class TrainingLogger:

    def __init__(self, log_dir='logs/training_log.csv', num_classes=None, append=False):
        self.log_dir = log_dir if append else _next_available_log_path(log_dir)
        self.num_classes = num_classes
        os.makedirs(os.path.dirname(self.log_dir) or '.', exist_ok=True)

        self.field_names = [
            'epoch', 'train_loss', 'coord_loss', 'obj_loss',
            'noobj_loss', 'class_loss', 'val_map', 'lr'
        ]

        if num_classes is not None:
            self.field_names += [f'class_{i}_ap' for i in range(num_classes)]

        if not os.path.exists(self.log_dir):
            with open(self.log_dir, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.field_names)
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
