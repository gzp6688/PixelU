import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

class PerceptualLoss(nn.Module):
    def __init__(self, use_lpips=False, use_seg=False):
        super().__init__()
        self.use_lpips = use_lpips
        self.use_seg = use_seg

        if self.use_lpips:
            import lpips
            self.lpips = lpips.LPIPS(net='vgg').eval()
            self.lpips.requires_grad_(False)

        if self.use_seg:
            weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
            self.seg_model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights).eval()
            self.seg_model.requires_grad_(False)
            self.seg_normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

    def forward(self, x_gt, x_pred, mask):
        x_gt_perc, x_pred_perc = self.process(x_gt, x_pred, mask)
        loss_lpips = torch.tensor(0.0, device=x_pred.device, dtype=x_pred.dtype)
        loss_seg = torch.tensor(0.0, device=x_pred.device, dtype=x_pred.dtype)
        if self.use_lpips:
            loss_lpips = self.lpips(x_gt_perc, x_pred_perc).mean()
        if self.use_seg:
            loss_seg = self.compute_seg_loss(x_gt_perc, x_pred_perc)
        return loss_lpips, loss_seg
    
    def process(self, x_gt, x_pred, mask):
        x_clean_perc = x_gt[mask]
        x_pred_perc = x_pred[mask]
        return x_clean_perc, x_pred_perc
    
    def compute_seg_loss(self, x_gt, x_pred):
        x_gt_norm = self.seg_normalize((x_gt + 1) * 0.5)
        x_pred_norm = self.seg_normalize((x_pred + 1) * 0.5)
        with torch.no_grad():
            gt_logits = self.seg_model(x_gt_norm)['out']
            gt_labels = gt_logits.argmax(dim=1) # [B, H, W]
        pred_logits = self.seg_model(x_pred_norm)['out'] # [B, NumClasses, H, W]
        loss = F.cross_entropy(pred_logits, gt_labels)
        return loss


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PerceptualLoss(use_lpips=False, use_seg=True).to(device)
    batch_size = 16
    x = torch.randn(batch_size, 3, 256, 256).to(device)
    y = torch.randn(batch_size, 3, 256, 256).to(device)
    mask = [0,3,5,7]
    loss_lpips, loss_seg = model(x, y, mask)
    print(loss_lpips, loss_seg)
