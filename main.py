import torch
import os
import numpy as np
import hydra
from app_utils.train_utils import seed_torch
from datasets.dataset import FeatureBagDataset, PatchBagDataset
from train.train_mil_model import train_model 
from train.train_weseg_model import train_model as train_weseg_model
from app_utils.settings import initialize_wandb
from omegaconf import DictConfig


@hydra.main(version_base="1.2.0", config_path="config", config_name="default")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, cfg.settings.seed)

    if cfg.model.type == 'weseg':
        dataset = PatchBagDataset(csv_path=cfg.data.csv_path,
                                  data_dir=cfg.data.data_dir,
                                  slide_dir=cfg.data.slide_dir,
                                  extension=cfg.data.extension,
                                  with_data_augmentation=cfg.data.with_data_augmentation,
                                  max_bag_size=cfg.settings.bag_size,
                                  )
    else:
        dataset = FeatureBagDataset(csv_path=cfg.data.csv_path,
                            features_dirs=cfg.data.features_dirs,
                            bag_size=cfg.settings.bag_size, is_test=False)
    if not os.path.isdir(cfg.data.results_dir):
        os.makedirs(cfg.data.results_dir)
    
    name_run = cfg.data.exp_code + '_' + cfg.model.type + '_' + cfg.model.based +'_' + cfg.model.size + '_' + cfg.optimizer.name + '_' + cfg.loss.inst_loss
    cfg.data.results_dir = os.path.join(cfg.data.results_dir, name_run + '_s{}'.format(cfg.settings.seed))
    if not os.path.isdir(cfg.data.results_dir):
        os.makedirs(cfg.data.results_dir)
    cfg.data.split_dir = os.path.join('splits', cfg.data.split_dir)
    assert os.path.isdir(cfg.data.split_dir)

    folds = np.arange(cfg.settings.k_start if cfg.settings.k_start != -1 else 0, cfg.settings.k_end if cfg.settings.k_end != -1 else cfg.settings.k)

    for i in folds:
        tracker = initialize_wandb(cfg, key=os.environ.get("WANDB_API_KEY"), fold=i) if cfg.wandb.enable else None
        seed_torch(device, cfg.settings.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits('{}/splits_{}.csv'.format(cfg.data.split_dir, i))
        train_dataset.is_train = True
        test_dataset.is_test = True
        datasets = (train_dataset, val_dataset, test_dataset)
        print("Training on {} samples".format(len(train_dataset)))
        print("Validating on {} samples".format(len(val_dataset)))
        print("Testing on {} samples".format(len(test_dataset)))
        
        if cfg.model.type == 'weseg':
            train_weseg_model(tracker, i, cfg, datasets)
        else:
            train_model(tracker, i, cfg, datasets)
        if cfg.wandb.enable:
            tracker.finish()


if __name__ == "__main__":
    main()
    print("finished!")
