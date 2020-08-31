import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# import class()
from model.model_head import STEmSeg
from utils.cluster import Cluster
from dataloader.dataloader import Loader
# import def()
from loss_metric.losses import SmoothLoss, CenterLoss, EmbeddingLoss, IoU_metric
from utils.visual_helper import Visual


parser = argparse.ArgumentParser()

parser.add_argument("--data", type=str, default="./ignore/data/DAVIS", help="data path")
parser.add_argument("--w", type=str, default="default_weights", help="weights name")
parser.add_argument("--time", type=int, default=8, help="Time size")
parser.add_argument("--batch", type=int, default=1, help="Batch size")
parser.add_argument("--workers", type=int, default=1, help="Num workers")
parser.add_argument("--epochs", type=int, default=1, help="Num epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--lr_step", type=int, default=200, help="Learning scheduler step")
parser.add_argument("--lr_gamma", type=float, default=0.90, help="Learning scheduler gamma")
parser.add_argument("--tb", type=str, default="None", help="Tensorboard")
parser.add_argument("--vis", type=bool, default=False, help="Train visual")
parser.add_argument("--train", type=bool, default=False, help="Train")

args = parser.parse_args()
    
# init tensorboard: !tensorboard --logdir=ignore/runs
if args.tb != "None":
    print("Tensorboard name: ", args.tb)
    writer = SummaryWriter('ignore/runs')

# init model
model = STEmSeg(batch_size=args.batch).cuda()
try:
    model.load_state_dict(torch.load('ignore/weights/%s.pth' % args.w), )#strict=False
except FileNotFoundError:
    print("!!!Create new weights!!!: ", "%s.pth" % args.w)
    pass
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Params", params)

# init cluster
cluster = Cluster(args.vis)

# init dataloader
data_loader = Loader(data_path=args.data, batch_size=args.batch, time=args.time, num_workers=args.workers)

# init optimizer and lr_scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

def train():
    for epoch in range(args.epochs):
        train_Sloss, train_Closs, train_Eloss, train_Tloss = 0, 0, 0, 0
        valid_Sloss, valid_Closs, valid_Eloss, valid_Tloss = 0, 0, 0, 0
        train_metric = 0
        valid_metric = 0
        test_metric = 0
        if args.train:
            for i, data in enumerate(data_loader["train"]):
                model.train()
                images, masks4 = data
                images, masks4 = images.cuda(), masks4.cuda()
                outs = model(images)
                Heat_map, _, _ = outs
                pred_masks = cluster.run(outs, masks4)
                smooth_loss = SmoothLoss(outs, masks4)
                center_loss = CenterLoss(outs, masks4)
                embedding_loss = EmbeddingLoss(pred_masks, masks4)
                loss = smooth_loss + center_loss + embedding_loss
                train_Sloss += smooth_loss.item()
                train_Closs += center_loss.item()
                train_Eloss += embedding_loss.item()
                train_Tloss += loss.item()
                train_metric += IoU_metric(masks4, (pred_masks>0.5).float())
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if args.vis == True:
                    Visual(pred_masks, Heat_map.reshape(pred_masks.size()), i, 'train')
            train_Sloss = train_Sloss/len(data_loader["train"])
            train_Closs = train_Closs/len(data_loader["train"])
            train_Eloss = train_Eloss/len(data_loader["train"])
            train_Tloss = train_Tloss/len(data_loader["train"])
            train_metric =  train_metric/len(data_loader["train"])
            print("epoch: ", epoch, "TotalLoss: %.4f" % train_Tloss, "IoU_metric: %.4f" % train_metric)
            print("SLoss: %.4f" % train_Sloss, "CLoss: %.4f" % train_Closs, "ELoss: %.4f" % train_Eloss)
            if epoch % 10 == 0 and epoch != 0:
                torch.save(model.state_dict(), 'ignore/weights/%s.pth' % args.w)
                print("Save weights: %s.pth" % args.w)
            if args.tb != "None":
                writer.add_scalars('%s_Total_loss' % args.tb, {'train' : train_Tloss}, epoch)
                # writer.add_scalars('%s_Smooth_loss' % args.tb, {'train' : train_Sloss}, epoch)
                # writer.add_scalars('%s_Center_loss' % args.tb, {'train' : train_Closs}, epoch)
                # writer.add_scalars('%s_Embeddidg_loss' % args.tb, {'train' : train_Eloss}, epoch)
                writer.add_scalars('%s_IoU_metric' % args.tb, {'train' : train_metric}, epoch)

            for i, data in enumerate(data_loader["valid"]):
                model.eval()
                images, masks4 = data
                images, masks4 = images.cuda(), masks4.cuda()
                outs = model(images)
                Heat_map, _, _ = outs
                pred_masks = cluster.run(outs, masks4)
                smooth_loss = SmoothLoss(outs, masks4)
                center_loss = CenterLoss(outs, masks4)
                embedding_loss = EmbeddingLoss(pred_masks, masks4)
                loss = smooth_loss + center_loss + embedding_loss
                valid_Sloss += smooth_loss.item()
                valid_Closs += center_loss.item()
                valid_Eloss += embedding_loss.item()
                valid_Tloss += loss.item()
                valid_metric += IoU_metric(masks4, (pred_masks>0.5).float())
                if args.vis == True:
                    Visual(pred_masks, Heat_map.reshape(pred_masks.size()), i, 'valid')
            valid_Sloss = valid_Sloss/len(data_loader["valid"])
            valid_Closs = valid_Closs/len(data_loader["valid"])
            valid_Eloss = valid_Eloss/len(data_loader["valid"])
            valid_Tloss = valid_Tloss/len(data_loader["valid"])
            valid_metric = valid_metric/len(data_loader["train"])
            print("epoch: ", epoch, "TotalLoss: %.4f" % valid_Tloss, "IoU_metric: %.4f" % valid_metric)
            print("SLoss: %.4f" % valid_Sloss, "CLoss: %.4f" % valid_Closs, "ELoss: %.4f" % valid_Eloss)
            if args.tb != "None":
                writer.add_scalars('%s_Total_loss' % args.tb, {'valid' : valid_Tloss}, epoch)
                # writer.add_scalars('%s_Smooth_loss' % args.tb, {'valid' : valid_Sloss}, epoch)
                # writer.add_scalars('%s_Center_loss' % args.tb, {'valid' : valid_Closs}, epoch)
                # writer.add_scalars('%s_Embeddidg_loss' % args.tb, {'valid' : valid_Eloss}, epoch)
                writer.add_scalars('%s_IoU_metric' % args.tb, {'valid' : valid_metric}, epoch)
            
        for i, data in enumerate(data_loader["test"]): 
            model.eval()
            images, masks4 = data
            images, masks4 = images.cuda(), masks4.cuda()
            outs = model(images)
            pred_masks = cluster.test_run(outs, iter_in_epoch=i)
            test_metric += IoU_metric(masks4, pred_masks)
        test_metric = test_metric/len(data_loader["train"])
        print("test metric: %.4f" % test_metric)
        if args.tb != "None":
            writer.add_scalars('%s_IoU_metric' % args.tb, {'test' : test_metric}, epoch)

if __name__ == "__main__":
    train()

