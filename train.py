import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# import class()
from model.model_head import STEmSeg
from utils.cluster import Cluster
from dataloader.dataloader import Loader
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


class AccumData():       
    def init(self):
        self.disp = {
            "train_Sloss" : 0,
            "train_Closs" : 0,
            "train_Eloss" : 0,
            "train_Tloss" : 0,
            "valid_Sloss" : 0,
            "valid_Closs" : 0,
            "valid_Eloss" : 0,
            "valid_Tloss" : 0,
            "train_metric" : 0,
            "valid_metric" : 0,
            "train_cluster_metric" : 0,
            "valid_cluster_metric" : 0,
            "train_iter" : 0,
            "valid_iter" : 0
        }

    def save_weights(self, epoch, weights_name):
        if epoch % 10 == 0 and epoch != 0:
            torch.save(model.state_dict(), 'ignore/weights/%s.pth' % weights_name)
            print("Save weights: %s.pth" % weights_name)

    def update(self, key, x):
        if key is "iter_metric":
            self.iter_metric = x
        elif key is "train_iter":
            self.disp[key] += 1
        elif key is "valid_iter":
            self.disp[key] += 1
        else:
            self.disp[key] += x

    def printer_train(self, i):
        if i % 5 == 0:
            print("Train iter:", i, "Loss: %0.4f" % (self.disp["train_Tloss"]/self.disp["train_iter"]),
                  "MetricIoU: %0.4f" % (self.disp["train_metric"]/self.disp["train_iter"]))
            # print(" "*10, "Smooth: %0.4f" % (self.disp["train_Sloss"]/self.disp["train_iter"]),
            #       "Center: %0.4f" % (self.disp["train_Closs"]/self.disp["train_iter"]),
            #       "Ebmedding: %0.4f" % (self.disp["train_Eloss"]/self.disp["train_iter"]))

    def printer_valid(self, i):
        if i % 5 == 0:
            print("Valid iter:", i, "Loss: %0.4f" % (self.disp["valid_Tloss"]/self.disp["valid_iter"]),
                  "MetricIoU: %0.4f" % (self.disp["valid_metric"]/self.disp["valid_iter"]))
            # print(" "*10, "Smooth: %0.4f" % (self.disp["valid_Sloss"]/self.disp["valid_iter"]),
            #       "Center: %0.4f" % (self.disp["valid_Closs"]/self.disp["valid_iter"]),
            #       "Ebmedding: %0.4f" % (self.disp["valid_Eloss"]/self.disp["valid_iter"]))

    def printer_epoch(self):
        print(" "*10, "TrainLoss: %0.4f" % (self.disp["train_Tloss"]/self.disp["train_iter"]),
              "TrainIoU: %0.4f" % (self.disp["train_metric"]/self.disp["train_iter"]))
        print(" "*10, "ValidLoss: %0.4f" % (self.disp["valid_Tloss"]/self.disp["valid_iter"]),
              "ValidIoU: %0.4f" % (self.disp["valid_metric"]/self.disp["valid_iter"]))

    def tensorboard(self, writer, tb, epoch):
        train_Tloss = self.disp["train_Tloss"]/self.disp["train_iter"]
        train_Sloss = self.disp["train_Sloss"]/self.disp["train_iter"]
        train_Closs = self.disp["train_Closs"]/self.disp["train_iter"]
        train_Eloss = self.disp["train_Eloss"]/self.disp["train_iter"]
        train_metric = self.disp["train_metric"]/self.disp["train_iter"]
        valid_Tloss = self.disp["valid_Tloss"]/self.disp["valid_iter"]
        valid_Sloss = self.disp["valid_Sloss"]/self.disp["valid_iter"]
        valid_Closs = self.disp["valid_Closs"]/self.disp["valid_iter"]
        valid_Eloss = self.disp["valid_Eloss"]/self.disp["valid_iter"]
        valid_metric = self.disp["valid_metric"]/self.disp["valid_iter"]
        if tb is not "None":
            writer.add_scalars('%s_Total_loss' % tb, {'train' : train_Tloss,
                                                      'valid' : valid_Tloss}, epoch)
            writer.add_scalars('%s_Smooth_loss' % args.tb, {'train' : train_Sloss,
                                                            'valid' : valid_Sloss}, epoch)
            writer.add_scalars('%s_Center_loss' % args.tb, {'train' : train_Closs,
                                                            'valid' : valid_Closs}, epoch)
            writer.add_scalars('%s_Embeddidg_loss' % args.tb, {'train' : train_Eloss,
                                                               'valid' : valid_Eloss}, epoch)
            writer.add_scalars('%s_IoU_metric' % tb, {'train' : train_metric,
                                                      'valid' : valid_metric}, epoch)

    def visual_train(self, vis, Visual, pred_masks, outs, i, mode):
        if vis is True:
            Heat_map, _, _ = outs
            Visual(pred_masks, Heat_map, i, mode)

    def visual_inference(self, vis, Visual_inference, pred_clusters, i, mode):
        if vis is True:
            Visual_inference(pred_clusters, i, mode)


def train():
    accum_data = AccumData()
    for epoch in range(args.epochs):
        print("***"*6,  "epoch: ", epoch, "***"*6)
        accum_data.init()
        if args.train:
            model.train()
            for i, data in enumerate(data_loader["train"]):
                i += 1
                images, masks = data
                images, masks = images.cuda(), masks.cuda()
                if images.size(0) != args.batch:
                    break
                outs = model(images)
                try:
                    pred_masks = cluster.train(outs, masks)
                except RuntimeError:
                    break
                total_loss, smooth_loss, center_loss, embedding_loss = Losses(pred_masks, outs, masks, mode="train")
                metric = IoU_metric(pred_masks, masks)
                try:
                    pred_clusters = cluster.inference(outs)
                except RuntimeError:
                    break  
                accum_data.update("train_Sloss", smooth_loss)
                accum_data.update("train_Closs", center_loss)
                accum_data.update("train_Eloss", embedding_loss)
                accum_data.update("train_Tloss", total_loss)
                accum_data.update("train_metric", metric)
                accum_data.update("iter_metric", metric)
                accum_data.update("train_iter", i)
                accum_data.visual_train(args.vis, Visual, pred_masks, outs, i, "train")
                accum_data.visual_inference(args.vis, Visual_inference, pred_clusters, i, "train")
                accum_data.printer_train(i)
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
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
                    break
                total_loss, smooth_loss, center_loss, embedding_loss = Losses(pred_masks, outs, masks, mode="valid")
                metric = IoU_metric(pred_masks, masks)
                try:
                    pred_clusters = cluster.inference(outs)
                except RuntimeError:
                    break
                accum_data.update("valid_Sloss", smooth_loss)
                accum_data.update("valid_Closs", center_loss)
                accum_data.update("valid_Eloss", embedding_loss)
                accum_data.update("valid_Tloss", total_loss)
                accum_data.update("valid_metric", metric)
                accum_data.update("iter_metric", metric)
                accum_data.update("valid_iter", i)
                accum_data.visual_train(args.vis, Visual, pred_masks, outs, i, "valid")
                accum_data.visual_inference(args.vis, Visual_inference, pred_clusters, i, "valid")
                accum_data.printer_valid(i)
            accum_data.tensorboard(writer, args.tb, epoch)
            accum_data.save_weights(epoch, args.w)
            accum_data.printer_epoch()


if __name__ == "__main__":
    train()

