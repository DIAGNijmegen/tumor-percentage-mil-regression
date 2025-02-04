__author__ = 'marvinler'

import torch
from torch import nn as nn


class MaxMinMIL(nn.Module):
    def __init__(self, classifier_model, class_imbalance_weights=None):
        super().__init__()

        self.instance_model = classifier_model

        self.class_imbalance_weights = class_imbalance_weights

        self.loss_function = nn.BCEWithLogitsLoss(reduction='none', pos_weight=class_imbalance_weights)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cuda = False
        if device.type == 'cuda':
            self.loss_function.cuda()
            self.use_cuda = True

    def loss(self, predictions, computed_instances_labels):
        """
        Computes instance-wise error signal with self.loss_function using instances predictions and computed
        proxy-labels, and use the input mask for averaging.
        :param predictions: tensor of instances predictions
        :param computed_instances_labels: tensor of computed proxy-labels, same shape
        :param mask_instances_labels: tensor of same shape than computed_instances_labels containing 1 if associated
        instance has an assigned proxy label, 0 otherwise
        :return: batch-averaged loss signal
        """
        instance_wise_loss = self.loss_function(predictions, computed_instances_labels)
        averaged_loss = instance_wise_loss.sum()/instance_wise_loss.shape[1]
        return averaged_loss

    def forward(self, instances, bag_label):
        assert instances.shape[0] == bag_label.shape[0] == 1, instances.shape
        instances = instances.squeeze(0)
        bag_label = bag_label.squeeze(0)
        n_instances = instances.size(0)

        current_device = instances.get_device()

        # Forwards each instance into optimized model
        instances_predictions = self.instance_model(instances)
        # Compute proxy-label based on bag label and predictions
        computed_instances_labels = torch.zeros(instances_predictions.shape, device=current_device).float()
        if bag_label == 0: 
            computed_instances_labels[:] = 0.
        else:  
            _, topk_idx = torch.topk(instances_predictions, k=round(bag_label.item()*n_instances), dim=0)
            computed_instances_labels[topk_idx] = 1.
            if round((1-bag_label.item())*n_instances) != 0:
                _, bottomk_idx = torch.topk(instances_predictions, k=round((1-bag_label.item())*n_instances), largest=False, dim=0)
                computed_instances_labels[bottomk_idx] = 0.
        
        if self.use_cuda:
            computed_instances_labels = computed_instances_labels.cuda()

        # stop gradient flow in backprop
        computed_instances_labels = computed_instances_labels.detach()

        return instances_predictions.unsqueeze(0), \
               computed_instances_labels.unsqueeze(0)
