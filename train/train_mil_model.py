import time
import torch
import numpy as np
import os
import wandb
import pandas as pd
from app_utils import train_utils
from models.abmil import ABMIL, ABMIL_Instance
from models.clam import CLAM_SB, CLAM_Instance
from models.meanpool import MeanPoolInstance
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count
    
def define_model_dictionary(cfg):
    """Creates a dictionary of model parameters based on configuration."""
    model_dict = {
        "dropout": cfg.model.drop_out,
        "n_classes": cfg.settings.n_classes,
        "size_arg": cfg.model.size,
        "dp_rate": cfg.model.dp_rate,
    }
    
    if cfg.model.type in ['abmil', 'clam']:
        model_dict["gate"] = cfg.model.gated_attention
    if cfg.model.type == 'clam':
        model_dict["k_sample"] = cfg.model.B
  
    return model_dict

def define_model(cfg, device):
    """Creates a model based on configuration settings."""

    model_dict = define_model_dictionary(cfg)

    if cfg.model.type == 'abmil':
        if cfg.model.based == 'instance':
            return ABMIL_Instance(**model_dict)
        else:
            return ABMIL(**model_dict)
    elif cfg.model.type == 'clam':
        instance_loss_fn = train_utils.define_loss(device, cfg.loss.inst_loss, 2)
        if cfg.model.based == 'instance':
            return CLAM_Instance(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            return CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
    elif cfg.model.type == 'meanpool':
        return MeanPoolInstance(**model_dict)
    

def load_checkpoint(cfg, cur, model, optimizer):
    """Loads the model checkpoint if exists."""

    checkpoint_path = os.path.join(cfg.data.results_dir, f"s_{cur}_checkpoint.pt")
    if os.path.exists(checkpoint_path):
        print(f"Resume from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return True, checkpoint['epoch'] + 1, checkpoint['counter']
    else:
        return False, 0, 0
    

def train_model(tracker, cur, cfg, datasets):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    since = time.time() 

    train_split, val_split, test_split = datasets

    # Instantiate the model based on configuration type
    model = define_model(cfg, device)
    model.to(device)
    train_utils.print_network(model)

    # Define loss function based on the bag loss type
    if cfg.loss.bag_loss == 'balanced':
        labels = train_split.get_labels() + val_split.get_labels()
        loss_fn = train_utils.define_loss(device, cfg.loss.bag_loss, cfg.settings.n_classes, labels)
    else:
        loss_fn = train_utils.define_loss(device, cfg.loss.bag_loss, cfg.settings.n_classes)
    
    # Get optimizer and scheduler
    optimizer = train_utils.get_optim(model, cfg)
    scheduler = train_utils.SchedulerFactory(optimizer, cfg.lr_scheduler).get_scheduler() if cfg.lr_scheduler.enable else None

    train_loader = train_utils.get_simple_loader(train_split, 'train', cfg)
    val_loader = train_utils.get_simple_loader(val_split, 'val', cfg)
    test_loader = train_utils.get_simple_loader(test_split, 'test', cfg)
    
    # Resume from checkpoint if available
    resume, start_epoch, ckpt_counter = load_checkpoint(cfg, cur, model, optimizer)
    
    # Early stopping
    early_stopping = train_utils.EarlyStopping(
        patience=cfg.early_stopping.patience, 
        stop_epoch=cfg.early_stopping.min_epoch, 
        verbose=True, 
        counter=0 if not resume else ckpt_counter
    ) if cfg.early_stopping.enable else None

    if cfg.testing.only_testing:
        ckpt = torch.load(cfg.testing.ckpt_path)['model_state_dict']
        ckpt_clean = {key.replace('.module', ''): ckpt[key] for key in ckpt.keys()}
        model.load_state_dict(ckpt_clean, strict=True)
        test(cfg, cur, model, test_loader)
        return
    
    # Initialize tracking
    if tracker is not None:
        tracker.define_metric("epoch", summary="max")
        tracker.define_metric("lr", step_metric="epoch")
        
    # Training loop
    for epoch in range(start_epoch, cfg.settings.max_epochs):
        if tracker is not None:
            tracker.log({"epoch": epoch + 1})

        print('Epoch {}/{}\n'.format(epoch, cfg.settings.max_epochs - 1))
        optimizer = train_loop(cfg, epoch, model, train_loader, optimizer, loss_fn,
                        tracker, cfg.model.type, cfg.model.bag_weight)
        stop, loss = eval_loop(cfg, cur, epoch, model, val_loader, loss_fn, cfg.model.type, early_stopping, tracker = tracker, optimizer = optimizer)
        
        if stop: break

        # Step the scheduler if enabled
        if scheduler:
            scheduler.step(loss)
            if cfg.wandb.enable:
                wandb.log({"train/lr": scheduler.get_last_lr()})

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if cfg.early_stopping.enable:
        checkpoint = torch.load(os.path.join(cfg.data.results_dir, "s_{}_checkpoint.pt".format(cur)))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'counter': 0,
            }, os.path.join(cfg.data.results_dir, "s_{}_checkpoint.pt".format(cur)))

    if cfg.wandb.enable:
        wandb.save(str(os.path.join(cfg.data.results_dir, "s_{}_checkpoint.pt".format(cur))))

    print("*** TEST DATA ***")
    test(cfg, cur, model, test_loader)


def train_loop(cfg, epoch, model, loader, optimizer, loss_fn, tracker=None, model_type='abmil', bag_weight=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    # Initialize tracking variables
    running_loss = 0.0
    running_instance_loss = 0.0
    instance_count = 0
    predictions = []
    labels = []
    inst_logger = Accuracy_Logger(n_classes=2) if model_type == 'clam' else None
    train_metrics = {}

    for batch_idx, (data, label, _) in enumerate(loader):
        # Move data to device
        data, label = data.to(device), label.to(device)
        labels.extend(label.detach().cpu().numpy().flatten())
        preds, _, _, instance_dict = model(data)
        loss = loss_fn(preds.flatten(), label)

        if model_type == 'clam':
            instance_loss = instance_dict['instance_loss']
            instance_loss_value = instance_loss.item()
            running_instance_loss += instance_loss_value
            instance_count += 1
            inst_logger.log_batch(instance_dict['inst_preds'], instance_dict['inst_labels'])
            total_loss = bag_weight * loss + (1 - bag_weight) * instance_loss
        else:
            total_loss = loss
        
        if (batch_idx + 1) % 20 == 0:
            msg = (
                f"Batch {batch_idx}, LR: {optimizer.param_groups[0]['lr']:.4f}, "
                f"Loss: {loss.item():.4f}, Label: {label.flatten().detach().cpu().numpy()[0]}, "
            )
            if model_type == 'clam':
                msg += (
                    f"Instance loss: {instance_loss_value:.4f}, Weighted loss (total): {total_loss.item():.4f}, "
                )
            msg += (
                f"Logits: {preds.flatten().detach().cpu().numpy()[0]}, Bag Size: {data.size(1)}"
            )
            print(msg)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log training progress
        running_loss += loss.item() * data.shape[0]
        predictions.extend(preds.detach().cpu().numpy().flatten())

    # Calculate average losses
    running_loss /= len(loader.dataset)
    if instance_count > 0:
        running_instance_loss /= instance_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    # Metrics computation
    predictions = np.vstack(predictions)
    predictions = np.clip(predictions, 0, None)
    labels = np.vstack(labels)
    spearman, _ = spearmanr(labels.ravel(), predictions.ravel())
    pearson, _ = pearsonr(labels.ravel(), predictions.ravel())
    mae = mean_absolute_error(labels.ravel(), predictions.ravel())
    auc = train_utils.compute_auc(labels.ravel(), predictions.ravel())
    train_metrics.update({
        'train_loss': running_loss,
        'train_instance_loss': running_instance_loss if model_type == 'clam' else None,
        'spearman': spearman,
        'pearson': pearson,
        'mae': mae,
    })

    auc_str = f"AUC: {auc:.4f}" if auc is not None else "AUC: N/A"
    if auc_str is not None:
        train_metrics['auc'] = auc

    print(f"Epoch: {epoch}, Train Loss: {running_loss:.4f}, Instance Loss: {running_instance_loss:.4f}, "
        f"Spearman: {spearman:.4f}, Pearson: {pearson:.4f}, MAE: {mae:.4f}, {auc_str}")

    # Log metrics to tracker
    if tracker:
        for metric_name, value in train_metrics.items():
            if value is not None:  # Skip None metrics
                tracker.log({f"train/{metric_name}": value}, step=epoch)

    return optimizer


def eval_loop(cfg, cur, epoch, model, loader, loss_fn, model_type, early_stopping=None, tracker=None, optimizer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    running_loss = 0.0
    instance_loss_total = 0.0
    instance_count = 0
    labs = []
    predictions = []
    val_metrics = {}
    inst_logger = Accuracy_Logger(n_classes=2) if model_type == 'clam' else None

    with torch.no_grad():
        for batch_idx, (data, label, _) in enumerate(loader):
            # Move data to device
            data, label = data.to(device), label.to(device)

            preds, _, _, instance_dict = model(data)
            loss = loss_fn(preds.flatten(), label)

            if model_type == 'clam':
                instance_loss = instance_dict['instance_loss']
                instance_count += 1
                instance_loss_total += instance_loss.item()
                inst_logger.log_batch(instance_dict['inst_preds'], instance_dict['inst_labels'])
                
            running_loss += loss.item() * data.shape[0]
            labs.extend(label.detach().cpu().numpy().flatten())
            predictions.extend(preds.detach().cpu().numpy().flatten())
        
    # Aggregate results
    running_loss /= len(loader.dataset)
    val_metrics['val_loss'] = running_loss
    predictions = np.vstack(predictions) 
    predictions = np.clip(predictions, 0, None)
    labs = np.vstack(labs) 

    val_metrics.update({
        'mae': mean_absolute_error(labs.ravel(), predictions.ravel()),
        'pearson': pearsonr(labs.ravel(), predictions.ravel())[0],
        'spearman': spearmanr(labs.ravel(), predictions.ravel())[0]
    })

    if model_type == 'clam':
        val_metrics['val_instance_loss'] = instance_loss_total / instance_count if instance_count > 0 else 0.0

    auc = train_utils.compute_auc(labs.ravel(), preds.ravel())
    
    auc_str = f"AUC: {auc:.4f}" if auc is not None else "AUC: N/A"
    if auc_str is not None:
        val_metrics['auc'] = auc
        
    print(f"Epoch: {epoch}, Val Loss: {val_metrics['val_loss']:.4f},"
        f"Spearman: {val_metrics['spearman']:.4f}, Pearson: {val_metrics['pearson']:.4f}, MAE: {val_metrics['mae']:.4f}, {auc_str}")

    if model_type == 'clam' and instance_count > 0:
        for i in range(inst_logger.n_classes):
            acc, correct, count = inst_logger.get_summary(i)
            print(f"class {i} clustering acc {acc}: correct {correct}/{count}")

    if tracker is not None:
        for key, value in val_metrics.items():
            wandb.define_metric(f"val/{key}", step_metric="epoch")
            wandb.log({f"val/{key}": value})

    # Early stopping
    if early_stopping is not None and cfg.early_stopping.enable:
        ckpt_name = os.path.join(cfg.data.results_dir, f"s_{cur}_checkpoint.pt")
        early_stopping(epoch, running_loss, model, optimizer, ckpt_name=ckpt_name)
        if early_stopping.early_stop:
            print('-' * 30)
            print("The Validation Loss Didn't Decrease, Early Stopping!!")
            print('-' * 30)
            return True, val_metrics

    return False, val_metrics


def test(cfg, cur, model, loader):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.eval()

    labs, predictions, slides = [], [], []

    with torch.no_grad():
        for batch_idx, (data, label, slide_id) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            preds, instance_scores, attention_scores, instance_dict = model(data)

            instance_scores = instance_scores.detach().cpu().numpy().squeeze()
            attention_scores = attention_scores.detach().cpu().numpy().squeeze()

            labs.extend(label.detach().cpu().numpy().flatten())
            predictions.extend(preds.detach().cpu().numpy().flatten())
            slides.extend(slide_id)

            attention_dir = os.path.join(cfg.data.results_dir, 'attention_scores')
            os.makedirs(attention_dir, exist_ok=True)
            with open(os.path.join(attention_dir, f"{cur}_{slides[-1]}.npy"), 'wb') as f:
                np.save(f, attention_scores)
            
            instance_dir = os.path.join(cfg.data.results_dir, 'instance_scores')
            os.makedirs(instance_dir, exist_ok=True)
            with open(os.path.join(instance_dir, f"{cur}_{slides[-1]}.npy"), 'wb') as f:
                np.save(f, instance_scores)
    
    predictions = np.clip(predictions, 0, None)          
    # Create and save results DataFrame
    df = pd.DataFrame({'slide_id': slides, 'label': labs, 'prediction': predictions})
    df.to_csv(os.path.join(cfg.data.results_dir, f'final_preds_{cur}.csv'), index=False)
    
    # Compute metrics
    mae = mean_absolute_error(labs, predictions)
    pearson, _ = pearsonr(labs, predictions)
    spearman, _ = spearmanr(labs, predictions)
    auc = train_utils.compute_auc(labs, predictions) 

    # Log metrics to Weights & Biases if enabled
    if cfg.wandb.enable:
        wandb.log({
            "test/mae": mae,
            "test/pearson": pearson,
            "test/spearman": spearman,
            **({"test/auc": auc} if auc is not None else {})
        })
    
    # Print final metrics
    print(f"The final metrics are: Pearson {pearson:.4f} - Spearman {spearman:.4f} - "
          f"MAE {mae:.4f} - AUC {auc:.4f}")





            



