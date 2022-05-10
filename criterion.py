import torch
from torch import nn

class RelativeDepthLoss(nn.Module):
    def __init__(self):
        super(RelativeDepthLoss, self).__init__()

    def ranking_loss(self, z_A, z_B, target):
        """
        loss for a given set of pixels:
        z_A: predicted absolute depth for pixels A
        z_B: predicted absolute depth for pixels B
        ground_truth: Relative depth between A and B (-1, 0, 1)
        """
        pred_depth = z_A - z_B
        log_loss = torch.mean(torch.log(1 + torch.exp(-target[target != 0] * pred_depth[target != 0])))
        squared_loss = torch.mean(pred_depth[target != 0] ** 2)  # if pred depth is not zero adds to loss
        return log_loss + squared_loss

    def forward(self, output, target):
        total_loss = 0
        for index in range(len(output)):
            x_A = target['x_A'][index].long()
            y_A = target['y_A'][index].long()
            x_B = target['x_B'][index].long()
            y_B = target['y_B'][index].long()
            
            x_A = x_A-10
            y_A = y_A-10
            x_B = x_B-10
            y_B = y_B-10
            
            z_A = output[index][0][x_A, y_A]  # all "A" points
            z_B = output[index][0][x_B, y_B]  # all "B" points

            total_loss += self.ranking_loss(z_A, z_B, target['ordinal_relation'][index])

        return total_loss / len(output)