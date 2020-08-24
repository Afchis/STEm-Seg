import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# import class()
from model.model_head import STEmSeg
from utils.cluster import Cluster
from dataloader.dataloader import Loader
# import def()
from loss_metric.losses import SmoothLoss, CenterLoss, EmbeddingLoss


parser = argparse.ArgumentParser()

parser.add_argument("--data", type=str, default="./ignore/data/DAVIS", help="data path")
parser.add_argument("--time", type=int, default=1, help="Time size")
parser.add_argument("--batch", type=int, default=1, help="Batch size")
parser.add_argument("--workers", type=int, default=1, help="Num workers")
parser.add_argument("--epochs", type=int, default=1000, help="Num epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--lr_step", type=int, default=200, help="Learning scheduler step")
parser.add_argument("--lr_gamma", type=float, default=0.90, help="Learning scheduler gamma")
parser.add_argument("--tb", type=str, default="None", help="Tensorboard")

args = parser.parse_args()
    
# init tensorboard: !tensorboard --logdir=ignore/runs
if args.tb != "None":
    print("Tensorboard name: ", args.tb)
    writer = SummaryWriter()

# init model
model = STEmSeg(batch_size=args.batch).cuda()
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Params", params)

# init cluster
cluster = Cluster()

# init dataloader
train_loader = Loader(data_path=args.data, batch_size=args.batch, time=args.time, num_workers=args.workers)

# init optimizer and lr_scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

def train():
    tb_iter = 0
    for epoch in range(args.epochs):
        # print("*"*32, "epoch: ",  epoch, "*"*32)
        for i, data in enumerate(train_loader):
            images, masks, masks4 = data
            images, masks, masks4 = images.cuda(), masks.cuda(), masks4.cuda()

            # if batch != args.batch: break
            try:
                outs = model(images)
            except RuntimeError:
                break

            pred_masks = cluster.run(outs, masks4)
            print("CLUSTR DONE!!!")
            smooth_loss = SmoothLoss(outs, masks4)
            print("SmoothLoss DONE!!!")
            center_loss = CenterLoss(outs, masks4)
            print("CenterLoss DONE!!!")
            embedding_loss = EmbeddingLoss(pred_masks, masks4)
            print("EmbeddingLoss DONE!!!")
            loss = smooth_loss + center_loss + embedding_loss
            # print loss for each iter
            print("iter: ", epoch, "TotalLoss: %.4f" % loss.item(), \
                "SLoss: %.4f" % smooth_loss.item(), "CLoss: %.4f" % center_loss.item(), "ELoss: %.4f" % embedding_loss.item())
            # TODO: print(metric)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            # tensorboard
            if args.tb != "None":
                writer.add_scalars('%s_Total_loss' % args.tb, {'train' : loss.item()}, tb_iter)
                writer.add_scalars('%s_Smooth_loss' % args.tb, {'train' : smooth_loss.item()}, tb_iter)
                writer.add_scalars('%s_Center_loss' % args.tb, {'train' : center_loss.item()}, tb_iter)
                writer.add_scalars('%s_Embeddidg_loss' % args.tb, {'train' : embedding_loss.item()}, tb_iter)
                tb_iter += 1


if __name__ == "__main__":
    train()

