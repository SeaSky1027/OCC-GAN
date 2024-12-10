import torch
import torch.nn as nn
from torch.nn import functional as F

class Contrastive_Loss(nn.Module):
    def __init__(self, data_temp=0.1, condition_temp=0.1, device='cuda:0'):
        super(Contrastive_Loss, self).__init__()
        self.device = device
        self.data_temp = data_temp
        self.condition_temp = condition_temp
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, features, labels=None, proxy=None):

        batch_size = features.shape[0]
        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        
        ###################################

        features = F.normalize(features, dim=1)

        # compute logits
        logits = torch.div(
            torch.matmul(features, features.T), # [bsz, bsz]
            self.data_temp
        )

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        ###################################

        if proxy is not None:
            proxy = F.normalize(proxy, dim=1)

            # compute logits
            d2c_logits = torch.div(
                self.cosine_similarity(features, proxy).unsqueeze(1), # [bsz, 1]
                self.condition_temp
            )

            logits = torch.cat([logits, d2c_logits], dim=1) # [bsz, bsz+1]
            mask = torch.cat([mask, torch.ones_like(d2c_logits)], dim=1) # [bsz, bsz+1]
            logits_mask = torch.cat([logits_mask, torch.ones_like(d2c_logits)], dim=1) # [bsz, bsz+1]

        ###################################

        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -mean_log_prob_pos
        loss = loss.mean()
        
        return loss