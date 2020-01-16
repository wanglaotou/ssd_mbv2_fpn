import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def sigmoid_focal_loss_cpu(logits, targets, gamma, alpha):
    num_classes = logits.shape[2]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(0, num_classes, dtype=dtype, device=device).unsqueeze(0).unsqueeze(1)
 
 
    t = targets.unsqueeze(2)
    p = torch.sigmoid(logits)
    term1 = (1 - p) ** gamma * torch.log(p)
    term2 = p ** gamma * torch.log(1 - p)
 
    return -(t == class_range).float() * term1 * alpha - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)
 
 
class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
 
    def forward(self, logits, targets):
        loss_func = sigmoid_focal_loss_cpu
        loss = loss_func(logits, targets, self.gamma, self.alpha)
        return loss.sum()
 
    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
 
if __name__=='__main__':
    input = torch.randn((2,3,5))
    target = torch.empty((2,3),dtype=torch.long).random_(5)
    print('input shape:', input.size())     # torch.Size([2, 3, 5])
    print('target shape:', target.size())    # torch.Size([2, 3])
    
    
    focalloss = SigmoidFocalLoss(gamma=2,alpha=0.25)
    output_focalloss = focalloss(input,target)
 
    print(output_focalloss)