import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from utils.core import imresize
from ..vgg import MultiVGGFeaturesExtractor

class RangeLoss(nn.Module):
    def __init__(self, min_value=0., max_value=1.):
        super(RangeLoss, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, inputs):
        loss = (F.relu(self.min_value - inputs) + F.relu(inputs - self.max_value)).mean()
        return loss

class PerceptualLoss(nn.Module):
    def __init__(self, features_to_compute, criterion=torch.nn.L1Loss()):
        super(PerceptualLoss, self).__init__()
        self.criterion = criterion
        self.features_extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute).eval()

    def forward(self, inputs, targets):
        inputs_fea = self.features_extractor(inputs)
        with torch.no_grad():
            targets_fea = self.features_extractor(targets)

        loss = 0
        for key in inputs_fea.keys():
            loss += self.criterion(inputs_fea[key], targets_fea[key].detach())

        return loss

class StyleLoss(nn.Module):
    def __init__(self, features_to_compute=['relu1_2', 'relu2_1', 'relu3_1'], criterion=torch.nn.L1Loss()):
        super(StyleLoss, self).__init__()
        self.criterion = criterion
        self.features_extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute).eval()

    def forward(self, inputs, targets):
        inputs_fea = self.features_extractor(inputs)
        with torch.no_grad():
            targets_fea = self.features_extractor(targets)

        loss = 0
        for key in inputs_fea.keys():
            inputs_gram = self._gram_matrix(inputs_fea[key])
            with torch.no_grad():
                targets_gram = self._gram_matrix(targets_fea[key]).detach()

            loss += self.criterion(inputs_gram, targets_gram)

        return loss

    def _gram_matrix(self, x):
        a, b, c, d = x.size()
        features = x.view(a, b, c * d)
        gram = features.bmm(features.transpose(1, 2))
        return gram.div(b * c * d)

class WassersteinLoss(nn.Module):
    def __init__(self, features_to_compute=['relu1_2', 'relu2_1', 'relu3_1'], criterion=torch.nn.MSELoss()):
        super(WassersteinLoss, self).__init__()
        self.criterion = criterion
        self.features_extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute).eval()

    def forward(self, inputs, targets):
        inputs_fea = self.features_extractor(inputs)
        with torch.no_grad():
            targets_fea = self.features_extractor(targets)

        loss = 0.
        for key in inputs_fea.keys():
            loss += self._sliced_wasserstein(inputs_fea[key], targets_fea[key])

        return loss

    def _sliced_wasserstein(self, inputs, targets):
        inputs = inputs.flatten(start_dim=2)
        targets = targets.flatten(start_dim=2)
        sorrted_inputs, _ = torch.sort(inputs, dim=-1)
        sorrted_targets, _ = torch.sort(targets, dim=-1)
        return self.criterion(sorrted_inputs, sorrted_targets.detach())

class DSD(nn.Module):
    def __init__(self, features_to_compute=['relu1_2', 'relu2_1', 'relu3_1'], scales=[0.5], criterion=torch.nn.L1Loss()):
        super(DSD, self).__init__()
        self.scales = scales
        self.criterion = criterion
        self.features_extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute).eval()

    def forward(self, inputs, targets):
        loss = 0.
        for scale in self.scales:
            inputs_scaled = imresize(inputs, scale=scale)
            targets_scaled = imresize(targets, scale=scale)

            inputs_style = self._compute_style(inputs, inputs_scaled)
            with torch.no_grad():
                targets_style = self._compute_style(targets, targets_scaled)
            
            for key in inputs_style.keys():
                loss += self.criterion(inputs_style[key], targets_style[key].detach())
        
        return loss

    def _compute_style(self, inputs, targets):
        inputs_fea = self.features_extractor(inputs)
        targets_fea = self.features_extractor(targets)
        
        style = OrderedDict()
        for key in inputs_fea.keys():
            inputs_gram = self._gram_matrix(inputs_fea[key])
            targets_gram = self._gram_matrix(targets_fea[key])
            diff = inputs_gram - targets_gram
            style.update({key: diff})
        
        return style

    def _gram_matrix(self, x):
        a, b, c, d = x.size()
        features = x.view(a, b, c * d)
        gram = features.bmm(features.transpose(1, 2))
        return gram.div(b * c * d)

class ContextualLoss(nn.Module):
    def __init__(self, features_to_compute=['relu2_1'], h=0.5, eps=1e-5):
        super(ContextualLoss, self).__init__()
        self.h = h
        self.eps = eps
        self.extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute, requires_grad=False, use_input_norm=True).eval()

    def forward(self, inputs, targets):
        inputs_fea = self.extractor(inputs)
        with torch.no_grad():
            targets_fea = self.extractor(targets)

        loss = 0
        for key in inputs_fea.keys():
            loss += self._contextual_loss(inputs_fea[key], targets_fea[key])

        return loss
    
    def _contextual_loss(self, inputs, targets):
        dist = self._cosine_dist(inputs, targets)
        dist_min, _ = torch.min(dist, dim=2, keepdim=True)

        # Eq (2)
        dist_tilde = dist / (dist_min + self.eps)
        
        # Eq (3)
        w = torch.exp((1 - dist_tilde) / self.h)

        # Eq (4)
        cx_ij = w / torch.sum(w, dim=2, keepdim=True)       # (N, H*W, H*W)

        # Eq (1)
        cx = torch.mean(torch.max(cx_ij, dim=1)[0], dim=1)  # (N, )
        loss = torch.mean(-torch.log(cx + self.eps))
    
        return loss

    def _cosine_dist(self, x, y):
        # reduce mean
        y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
        x_centered = x - y_mu
        y_centered = y - y_mu

        # L2 normalization
        x_normalized = F.normalize(x_centered, p=2, dim=1)
        y_normalized = F.normalize(y_centered, p=2, dim=1)

        # channel-wise vectorization
        x_normalized = x_normalized.flatten(start_dim=2)  # (N, C, H*W)
        y_normalized = y_normalized.flatten(start_dim=2)  # (N, C, H*W)

        # consine similarity
        cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, H*W, H*W)

        # convert to distance
        dist = 1 - cosine_sim

        return dist