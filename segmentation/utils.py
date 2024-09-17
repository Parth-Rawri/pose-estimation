import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def denormalize_rgb(rgb):
    """
    In:
        rgb: Tensor [3, height, width].
    Out:
        rgb: Tensor [3, height, width].
    Purpose:
        Denormalize an RGB image.
    """
    mean_rgb = [0.722, 0.751, 0.807]
    std_rgb = [0.171, 0.179, 0.197]
    for i in range(rgb.shape[0]):
        rgb[i] = np.maximum(rgb[i] * std_rgb[i] + mean_rgb[i], 0)
    return rgb


def show_rgb(rgb_img):
    """
    In:
        rgb_img: Numpy array [height, width, 3].
    Out:
        None.
    Purpose:
        Visualize an RGB image.
    """
    plt.figure()
    plt.imshow(rgb_img)
    plt.show()


def put_palette(obj_id):
    """
    In:
        obj_id: int.
    Out:
        None.
    Purpose:
        Fetch the mask color of specific object.
    """
    mypalette = np.array(
        [[0, 0, 0],
         [255, 0, 0],
         [0, 255, 0],
         [0, 0, 255],
         [255, 255, 0],
         [255, 0, 255],
         ],
        dtype=np.uint8,
    )
    return mypalette[obj_id]


def mask2rgb(mask):
    """
    In:
        mask: Numpy array [height, width].
    Out:
        None.
    Purpose:
        Convert a mask to RGB image for visualization.
    """
    v_put_palette = np.vectorize(put_palette, signature='(n)->(n,3)')
    return v_put_palette(mask.flatten()).reshape(mask.shape[0], mask.shape[1], 3)


def show_mask(mask):
    """
    In:
        mask: Numpy array [height, width].
    Out:
        None.
    Purpose:
        Visualize a mask.
    """
    show_rgb(mask2rgb(mask))


def check_dataset(dataset):
    """
    In:
        dataset: RGBDataset instance in this homework.
    Out:
        None.
    Purpose:
        Test dataset by visualizing a random sample.
    """

    print("dataset size:", len(dataset))
    sample = dataset[np.random.randint(len(dataset))]
    rgb = sample['input'].numpy()
    print("input shape:", rgb.shape)
    show_rgb(denormalize_rgb(rgb).transpose(1, 2, 0))
    if dataset.has_gt is True:
        mask = sample['target'].numpy()
        print("target shape:", mask.shape)
        show_mask(mask)


def check_dataloader(dataloader):
    """
    In:
        dataloader: Dataloader instance.
    Out:
        None.
    Purpose:
        Test dataloader by visualizing a batch.
    """
    print("dataset size:", len(dataloader.dataset))
    dataiter = iter(dataloader)
    sample = dataiter.next()
    rgb = sample['input'].numpy()
    print("input shape:", rgb.shape)
    if dataloader.dataset.has_gt is True:
        mask = sample['target'].numpy()
        print("target shape:", mask.shape)
    for i in range(rgb.shape[0]):
        show_rgb(denormalize_rgb(rgb[i]).transpose(1, 2, 0))
        if dataloader.dataset.has_gt is True:
            show_mask(mask[i])


def save_checkpoint(model, epoch, val_miou, path='checkpoint.pth.tar'):
    """
    Save the model's state dictionary.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'model_miou': val_miou,
    }, path)
    print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(model, path='checkpoint.pth.tar'):
    """
    Load a saved model checkpoint.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['epoch'], checkpoint['model_miou']


def plot_learning_curve(train_loss_list, train_miou_list, val_loss_list, val_miou_list, path='learning_curve.png'):
    """
    Plot and save the training and validation loss/mIoU curves.
    """
    epochs = np.arange(1, len(train_loss_list) + 1)
    plt.figure()
    plt.plot(epochs, train_loss_list, label="train_loss", color='navy')
    plt.plot(epochs, train_miou_list, label="train_mIoU", color='teal')
    plt.plot(epochs, val_loss_list, label="val_loss", color='orange')
    plt.plot(epochs, val_miou_list, label="val_mIoU", color='gold')
    plt.legend(loc='upper left')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU / Loss')
    plt.grid(True)
    plt.savefig(path, bbox_inches='tight')
    plt.show()


def save_predictions(model, device, dataloader, dataset_dir):
    """
    Save model predictions to the specified directory.
    """
    pred_dir = Path(dataset_dir) / 'pred'
    pred_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving predicted masks to {pred_dir}")

    model.to(device).eval()
    with torch.no_grad():
        for batch_id, batch in enumerate(dataloader):
            inputs = batch['input'].to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)

            for i in range(preds.size(0)):
                scene_id = batch_id * dataloader.batch_size + i
                mask = preds[i].cpu().numpy()
                mask_path = pred_dir / f"{scene_id}_pred.png"
                show_mask(mask)
                cv2.imwrite(str(mask_path), mask.astype(np.uint8))


def iou(prediction, target):
    """
    Compute IoU for each class excluding the background.
    """
    _, pred = torch.max(prediction, dim=1)
    batch_size = prediction.size(0)
    class_num = prediction.size(1)
    batch_ious = []

    for batch_id in range(batch_size):
        class_ious = []
        for class_id in range(1, class_num):  # Skip background class
            mask_pred = (pred[batch_id] == class_id).int()
            mask_target = (target[batch_id] == class_id).int()
            if mask_target.sum() == 0:  # Skip if target is not present
                continue
            intersection = (mask_pred * mask_target).sum().item()
            union = mask_pred.sum().item() + mask_target.sum().item() - intersection
            class_ious.append(float(intersection) / float(union) if union > 0 else 0.0)
        batch_ious.append(np.mean(class_ious) if class_ious else 0.0)
    
    return batch_ious

