from app_utils import train_utils
from tqdm import tqdm
from time import gmtime, strftime
from glob import glob
from models.weseg import MaxMinMIL
from models.weseg_classifiers import instantiate_model
import time
import torch
import os
import numpy as np


def to_dataloader(dataset, for_training, num_workers):
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=for_training, num_workers=num_workers)

def early_stopping(val_losses, patience):
    """ Return (True, min achieved val loss) if no val losses is under the minimal achieved val loss for patience
        epochs, otherwise (False, None) """
    # Do not stop until enough epochs have been made
    if len(val_losses) < patience:
        return False, None

    best_val_loss = np.min(val_losses)
    if not np.any(val_losses[-patience:] <= best_val_loss):
        return True, best_val_loss
    return False, None

def load_checkpoint(cfg, cur, mil_model, optimizer):
    """Loads the model checkpoint if exists."""

    results_dir = os.path.join(cfg.data.results_dir, 'fold_{}'.format(cur))
    if os.path.exists(results_dir):
        checkpoint_files = [f for f in os.listdir(results_dir) if f.endswith('.pt')]
        if checkpoint_files:
            latest_checkpoint = None
            max_epoch = -1
            for checkpoint_file in checkpoint_files:
                epoch_str = checkpoint_file.split('_')[2][5:]  # Extract the epoch part
                try:
                    epoch = int(epoch_str)
                    if epoch > max_epoch:
                        max_epoch = epoch
                        latest_checkpoint = checkpoint_file
                except ValueError:
                    continue  # Ignore invalid files
            if latest_checkpoint:
                print("Latest checkpoint: ", latest_checkpoint)
                checkpoint_path = os.path.join(results_dir, latest_checkpoint)
                checkpoint = torch.load(checkpoint_path)
                mil_model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print("Resuming training from epoch: ", start_epoch)
                return True, start_epoch
    return False, 0


def save_checkpoint(results_dir, prefix_time, cur, epoch, val_loss, mil_model, optimizer):
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(
        results_dir,
        '{}_fold{}_epoch{}_val_loss{:.3f}.pt'.format(
            prefix_time, cur, epoch, val_loss
        )
    )

    torch.save({
        'epoch': epoch,
        'model_state_dict': mil_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,  # Save the validation loss
    }, save_path)
    print("Saved model: ", save_path)


def train_model(tracker, cur, cfg, datasets):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    since = time.time() 

    train_split, val_split, test_split = datasets

    # Instantiate the model 

    model_dict = {"model_type": cfg.model.underlying_model_type, 
                      "pretrained": cfg.model.underlying_model_pretrained,
                      "n_classes": cfg.settings.n_classes,}
    
    instance_classifier, _ = instantiate_model(**model_dict)
    mil_model = MaxMinMIL(instance_classifier)

    mil_model.to(device)
    train_utils.print_network(mil_model)

    optimizer = train_utils.get_optim(mil_model, cfg)
    train_dataloader = to_dataloader(train_split, True, cfg.settings.num_workers)
    val_dataloader = to_dataloader(val_split, False, cfg.settings.num_workers) if len(val_split) else None
    test_dataloader = to_dataloader(test_split, False, cfg.settings.num_workers) if len(test_split) else None

    results_dir = os.path.join(cfg.data.results_dir, 'fold_{}'.format(cur))

    if not cfg.testing.only_testing:
        start_epoch = 0

        resume, start_epoch = load_checkpoint(cfg, cur, mil_model, optimizer)

        prefix_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        val_losses = []
        for epoch in range(start_epoch, cfg.settings.max_epochs):
            if tracker is not None:
                tracker.log({"epoch": epoch + 1})

            train_loss = perform_epoch(mil_model, True, epoch, train_dataloader,
                                                    optimizer, 'training', results_dir,
                                                    tracker, cfg) 
            if val_dataloader:
                with torch.no_grad():
                    val_loss = perform_epoch(mil_model, False, epoch, val_dataloader,
                                                            optimizer, 'validation', results_dir,
                                                            tracker, cfg) 
                    
                # Early stopping
                val_losses.append(val_loss)
                do_stop, best_value = early_stopping(val_losses, patience=cfg.early_stopping.patience)
                if do_stop:
                    print('Early stopping triggered: stopping training after no improvement on val set for '
                        '%d epochs with value %.3f' % (cfg.early_stopping.patience, best_value))
                    break

                save_checkpoint(results_dir, prefix_time, cur, epoch, val_loss, mil_model, optimizer)

        print('Total training time %s' % (time.time() - since))

    else:
        test_model(cfg, results_dir, cur, mil_model, optimizer, test_dataloader, tracker)


def perform_epoch(model, is_training, epoch, dataloader, optimizer, set_name, results_dir, tracker, cfg):
    model.train() if is_training else model.eval()

    print("Performing epoch: ", epoch)
    epoch_loss = []

    for batch_idx, (slide_instances, slide_label, slide_id) in enumerate(tqdm(dataloader)):
        if is_training:
            optimizer.zero_grad()

        slide_instances = slide_instances.cuda()
        slide_label = slide_label.cuda()
        # Forward pass
        instances_predictions, computed_instances_labels = model(slide_instances, slide_label)
        loss = model.loss(instances_predictions, computed_instances_labels)

        epoch_loss.append(loss.item() / slide_instances.shape[0])
        if is_training:
            loss.backward()
            optimizer.step()
        
        if set_name == 'test':
            np_predictions = instances_predictions.flatten().cpu().numpy()
            predictions_path = os.path.join(results_dir, f'predictions_{dataloader.dataset.current_range[1]}', slide_id[0] + '.npy')
            os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
            np.save(predictions_path, np_predictions)
        
    mean_epoch_loss = np.mean(epoch_loss)
    std_epoch_loss = np.std(epoch_loss)

    if tracker is not None:
        tracker.update({"epoch": epoch, f"{set_name}/loss_mean": mean_epoch_loss})
        tracker.update({"epoch": epoch, f"{set_name}/loss_std": std_epoch_loss})

    log_msg = f"Epoch: {epoch:2d}/{cfg.settings.max_epochs}  \
            loss={mean_epoch_loss:.4f}+/-{std_epoch_loss:.4f}" if is_training \
            else f"{' ' * 100} {set_name}  loss={mean_epoch_loss:.4f}+/-{std_epoch_loss:.4f}"
    print(log_msg)

    return mean_epoch_loss


def test_model(cfg, results_dir, cur, mil_model, optimizer, test_dataloader, tracker):
    folder_names = glob(os.path.join(results_dir, 'predictions*'))
    
    if folder_names:
        max_number = max(int(folder.split('_')[-1]) for folder in folder_names)
        k = int((max_number / 500) - 1)
        start_idx = k * cfg.settings.chunk_size
    else:
        start_idx = 0
        k = 0
    
    limits = cfg.settings.limits
    up_lim = limits[int(cur)]
    chunk_size = cfg.settings.chunk_size

    for start_idx, end_idx in test_dataloader.dataset.chunk_indices(up_lim, chunk_size):
        if start_idx < k * chunk_size:
            continue
        print(f"Testing on instances {start_idx} to {end_idx}")
        test_dataloader.dataset.set_range(start_idx, end_idx)
        checkpoint = torch.load(cfg.testing.ckpt_path)
        mil_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        with torch.no_grad():
            perform_epoch(mil_model, False, -1, test_dataloader,
                                                    optimizer, 'test', results_dir,
                                                    tracker, cfg) 
