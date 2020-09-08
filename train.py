import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# import class()
from model.model_head import STEmSeg
from utils.cluster import Cluster
from dataloader.dataloader import Loader
from utils.accum_data import AccumData
# import def()
from loss_metric.losses import Losses
from loss_metric.metrics import IoU_metric
from utils.visual_helper import Visual, Visual_inference


parser = argparse.ArgumentParser()

parser.add_argument("--size", type=int, default=512, help="Input size")
parser.add_argument("--w", type=str, default="default_weights", help="weights name")
parser.add_argument("--time", type=int, default=8, help="Time size")
parser.add_argument("--batch", type=int, default=1, help="Batch size")
parser.add_argument("--workers", type=int, default=1, help="Num workers")
parser.add_argument("--epochs", type=int, default=1, help="Num epochs")
parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
parser.add_argument("--lr_step", type=int, default=200, help="Learning scheduler step")
parser.add_argument("--lr_gamma", type=float, default=0.9, help="Learning scheduler gamma")
parser.add_argument("--tb", type=str, default="None", help="Tensorboard")
parser.add_argument("--vis", type=bool, default=False, help="Train visual")
parser.add_argument("--train", type=bool, default=False, help="Train")
parser.add_argument("--mode", type=str, default="xyt", help="Model mode, type 'xyt', 'xytf'")

args = parser.parse_args()
    
# init tensorboard: !tensorboard --logdir=ignore/runs
print("Tensorboard name: ", args.tb)
writer = SummaryWriter('ignore/runs')

# init model
model = STEmSeg(batch_size=args.batch, mode=args.mode, size=args.size).cuda()
model = nn.DataParallel(model)
try:
    model.load_state_dict(torch.load('ignore/weights/%s.pth' % args.w), strict=False)
except FileNotFoundError:
    print("!!!Create new weights!!!: ", "%s.pth" % args.w)
    pass
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Params", params)

# init cluster
cluster = Cluster(vis=args.vis)

# init dataloader
data_loader = Loader(size=args.size, batch_size=args.batch, time=args.time, num_workers=args.workers, shuffle=True)

# init optimizer and lr_scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)#, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

def save_model(epoch, i):
    if epoch % i == 0 and epoch != 0:
        torch.save(model.state_dict(), 'ignore/weights/%s.pth' % args.w)
        print("Save weights: %s.pth" % args.w)


def train():
    accum_data = AccumData()
    for epoch in range(args.epochs):
        print("***"*6,  "epoch: ", epoch, "***"*6)
        accum_data.init()
        if args.train:
            model.train()
            for i, data in enumerate(data_loader["train"]):
                optimizer.zero_grad()
                i += 1
                images, masks = data
                images, masks = images.cuda(), masks.cuda()
                if images.size(0) != args.batch:
                    break
                outs = model(images)
                try:
                    pred_masks = cluster.train(outs, masks)
                except RuntimeError:
                    print("CONTINUE"*9)
                    break
                total_loss, smooth_loss, center_loss, embedding_loss = Losses(pred_masks, outs, masks, mode="train")
                metric = IoU_metric(pred_masks, masks)
                try:
                    pred_clusters = cluster.inference(outs)
                except RuntimeError:
                    print("continue"*9)
                    break  
                accum_data.update("train_Sloss", smooth_loss)
                accum_data.update("train_Closs", center_loss)
                accum_data.update("train_Eloss", embedding_loss)
                accum_data.update("train_Tloss", total_loss)
                accum_data.update("train_metric", metric)
                accum_data.update("iter_metric", metric)
                accum_data.update("train_iter", i)
                accum_data.visual_train(args.vis, Visual, pred_masks, outs, i, "train")
                accum_data.visual_inference(args.vis, Visual_inference, pred_clusters, images, i, "train")
                accum_data.printer_train(i)
                total_loss.backward()
                optimizer.step()
                scheduler.step()
            model.eval()
            for i, data in enumerate(data_loader["valid"]):
                i += 1
                images, masks = data
                images, masks = images.cuda(), masks.cuda()
                if images.size(0) != args.batch:
                    break
                with torch.no_grad():
                    outs = model(images)
                try:
                    pred_masks = cluster.train(outs, masks)
                except RuntimeError:
                    print("CONTINUE"*9)
                    break
                total_loss, smooth_loss, center_loss, embedding_loss = Losses(pred_masks, outs, masks, mode="valid")
                metric = IoU_metric(pred_masks, masks)
                try:
                    pred_clusters = cluster.inference(outs)
                except RuntimeError:
                    print("continue"*9)
                    break
                accum_data.update("valid_Sloss", smooth_loss)
                accum_data.update("valid_Closs", center_loss)
                accum_data.update("valid_Eloss", embedding_loss)
                accum_data.update("valid_Tloss", total_loss)
                accum_data.update("valid_metric", metric)
                accum_data.update("iter_metric", metric)
                accum_data.update("valid_iter", i)
                accum_data.visual_train(args.vis, Visual, pred_masks, outs, i, "valid")
                accum_data.visual_inference(args.vis, Visual_inference, pred_clusters, images, i, "valid")
                accum_data.printer_valid(i)
            accum_data.tensorboard(writer, args.tb, epoch)
            accum_data.printer_epoch()
            save_model(epoch, 5)
            


if __name__ == "__main__":
    train()

