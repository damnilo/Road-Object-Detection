import csv
import matplotlib.pyplot as plt

def plot_training_log(log_dir='logs/training_log.csv'):
    epochs, train_loss, val_map, coord, obj, noobj, class_loss = [], [], [], [], [], [], []

    with open(log_dir, 'r') as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            val_map.append(float(row['val_map']))
            coord.append(float(row['coord_loss']))
            obj.append(float(row['obj_loss']))
            noobj.append(float(row['noobj_loss']))
            class_loss.append(float(row['class_loss']))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_loss, label='Train Loss')
    axes[0].plot(epochs, coord, label='Coord Loss', linestyle='--')
    axes[0].plot(epochs, obj, label='Obj Loss', linestyle='--')
    axes[0].plot(epochs, noobj, label='NoObj Loss', linestyle='--')
    axes[0].plot(epochs, class_loss, label='Class Loss', linestyle='--')
    axes[0].set_title('Training Losses')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(epochs, val_map, label='Validation mAP', color='orange')
    axes[1].set_title('Validation mAP')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('mAP')

    plt.tight_layout()
    plt.savefig('logs/training_plot.png')
    plt.show()

if __name__ == "__main__":
    plot_training_log()