import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# import class()
from model.model_head import STEmSeg
from utils.cluster import Cluster
from dataloader.dataloader import Loader
# import def()
from loss_metric.losses import Losses
from loss_metric.metrics import IoU_metric
from utils.visual_helper import Visual


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
    model.load_state_dict(torch.load('ignore/weights/%s.pth' % args.w), )#strict=False
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
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0005)
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

    def save_weights(epoch, weights_name):
        if epoch % 10 == 0 and epoch != 0:
            torch.save(model.state_dict(), 'ignore/weights/%s.pth' % weights_name)
            print("Save weights: %s.pth" % weights_name)

    def update(self, key, x):
        if key is "train_iter" or "valid_iter":
            self.disp[key] += 1
        elif key is "iter_metric":
            iret_metric = x
        else:
            self.disp[key] += x

    def printer_train(self, i):
        # if i % 2 == 0:
        print("Train iter: ", i, "Loss: %0.4f" % (self.disp["train_Tloss"]/self.disp["train_iter"]),
              "MetricIoU: %0.4f" % iret_metric)
        print("Smooth: %0.4f" % (self.disp["train_Sloss"]/self.disp["train_iter"]),
              "Center: %0.4f" % (self.disp["train_Closs"]/self.disp["train_iter"]),
              "Ebmedding: %0.4f" % (self.disp["train_Eloss"]/self.disp["train_iter"]))

    def printer_valid(self, i):
        # if i % 2 == 0:
        print("Valid iter: ", i, "Loss: %0.4f" % (self.disp["valid_Tloss"]/self.disp["valid_iter"]),
              "MetricIoU: %0.4f" % iret_metric)
        print("Smooth: %0.4f" % (self.disp["valid_Sloss"]/self.disp["valid_iter"]),
              "Center: %0.4f" % (self.disp["valid_Closs"]/self.disp["valid_iter"]),
              "Ebmedding: %0.4f" % (self.disp["valid_Eloss"]/self.disp["valid_iter"]))

    def printer_epoch(self, i):
        print("Epoch train loss: %0.4f" % (self.disp["train_Tloss"]/self.disp["train_iter"]),
              "Epoch valid loss: %0.4f" % (self.disp["valid_Tloss"]/self.disp["valid_iter"]))
        print("Epoch train metric: %0.4f" % (self.disp["train_metric"]/self.disp["train_iter"]),
              "Epoch valid metric: %0.4f" % (self.disp["valid_metric"]/self.disp["valid_iter"]),)

    def tensorboard(self, writer, tb, epoch):
        train_Tloss = self.disp["train_Tloss"]/self.disp["train_iter"]
        train_metric = self.disp["train_metric"]/self.disp["valid_iter"]
        valid_Tloss = self.disp["valid_Tloss"]/self.disp["train_iter"]
        valid_metric = self.disp["valid_metric"]/self.disp["valid_iter"]
        if tb is not "None":
            writer.add_scalars('%s_Total_loss' % tb, {'train' : train_Tloss,
                                                      'valid' : valid_Tloss}, epoch)
            # writer.add_scalars('%s_Smooth_loss' % args.tb, {'train' : train_Sloss}, epoch)
            # writer.add_scalars('%s_Center_loss' % args.tb, {'train' : train_Closs}, epoch)
            # writer.add_scalars('%s_Embeddidg_loss' % args.tb, {'train' : train_Eloss}, epoch)
            writer.add_scalars('%s_IoU_metric' % tb, {'train' : train_metric,
                                                      'valid' : valid_metric}, epoch)

    def visual(self, vis, Visual, pred_masks, outs, i, mode):
        if vis is True:
            Heat_map, _, _ = outs
            Visual(pred_masks, Heat_map, i, mode)


def train():
    accum_data = AccumData()
    for epoch in range(args.epochs):
        print("---"*6,  "epoch: ", epoch, "---"*6)
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
                pred_masks = cluster.train(outs, masks)
                total_loss, smooth_loss, center_loss, embedding_loss = Losses(pred_masks, outs, masks)
                metric = IoU_metric(pred_masks, masks)
                accum_data.update("train_Sloss", smooth_loss)
                accum_data.update("train_Closs", center_loss)
                accum_data.update("train_Eloss", embedding_loss)
                accum_data.update("train_Tloss", total_loss)
                accum_data.update("train_metric", metric)
                accum_data.update("iter_metric", metric)
                accum_data.update("iter", i)
                accum_data.visual(args.vis, Visual, pred_masks, outs, i, "train")
                accum_data.printer(i)
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            model.eval()
            for i, data in enumerate(dataloader["valid"]):
                i += 1
                images, masks = data
                images, masks = images.cuda(), masks.cuda()
                if images.size(0) != args.batch:
                    break
                outs = model(images)
                pred_masks = cluster.train(outs, masks)
                total_loss, smooth_loss, center_loss, embedding_loss = Losses(pred_masks, outs, masks)
                metric = IoU_metric(pred_masks, masks)
                accum_data.update("valid_Sloss", smooth_loss)
                accum_data.update("valid_Closs", center_loss)
                accum_data.update("valid_Eloss", embedding_loss)
                accum_data.update("valid_Tloss", total_loss)
                accum_data.update("valid_metric", metric)
                accum_data.update("iter_metric", metric)
                accum_data.update("iter", i)
                accum_data.visual(args.vis, Visual, pred_masks, outs, i, "train")
                accum_data.printer(i)
            accum_data.tensorboard(writer, args.tb, epoch)
            accum_data.save_weights(epoch, args.w)
            
# def train():
#     for epoch in range(args.epochs):
#         train_Sloss, train_Closs, train_Eloss, train_Tloss = 0, 0, 0, 0
#         valid_Sloss, valid_Closs, valid_Eloss, valid_Tloss = 0, 0, 0, 0
#         train_metric = 0
#         valid_metric = 0
#         test_metric = 0
#         if args.train:
#             for i, data in enumerate(data_loader["train"]):
#                 model.train()
#                 images, masks4 = data
#                 images, masks4 = images.cuda(), masks4.cuda()
#                 outs = model(images)
#                 Heat_map, _, _ = outs
#                 pred_masks = cluster.run(outs, masks4)
#                 smooth_loss = SmoothLoss(outs, masks4)
#                 center_loss = CenterLoss(outs, masks4)
#                 embedding_loss = EmbeddingLoss(pred_masks, masks4)
#                 loss = smooth_loss + center_loss + embedding_loss
#                 train_Sloss += smooth_loss.item()
#                 train_Closs += center_loss.item()
#                 train_Eloss += embedding_loss.item()
#                 train_Tloss += loss.item()
#                 train_metric += IoU_metric(masks4, (pred_masks>0.5).float())
#                 loss.backward()
#                 optimizer.step()
#                 scheduler.step()
#                 optimizer.zero_grad()
#                 if args.vis == True:
#                     Visual(pred_masks, Heat_map.reshape(pred_masks.size()), i, 'train')
#             train_Sloss = train_Sloss/len(data_loader["train"])
#             train_Closs = train_Closs/len(data_loader["train"])
#             train_Eloss = train_Eloss/len(data_loader["train"])
#             train_Tloss = train_Tloss/len(data_loader["train"])
#             train_metric =  train_metric/len(data_loader["train"])
#             print("epoch: ", epoch, "TotalLoss: %.4f" % train_Tloss, "IoU_metric: %.4f" % train_metric)
#             print("SLoss: %.4f" % train_Sloss, "CLoss: %.4f" % train_Closs, "ELoss: %.4f" % train_Eloss)
#             if epoch % 10 == 0 and epoch != 0:
#                 torch.save(model.state_dict(), 'ignore/weights/%s.pth' % args.w)
#                 print("Save weights: %s.pth" % args.w)
#             if args.tb != "None":
#                 writer.add_scalars('%s_Total_loss' % args.tb, {'train' : train_Tloss}, epoch)
#                 # writer.add_scalars('%s_Smooth_loss' % args.tb, {'train' : train_Sloss}, epoch)
#                 # writer.add_scalars('%s_Center_loss' % args.tb, {'train' : train_Closs}, epoch)
#                 # writer.add_scalars('%s_Embeddidg_loss' % args.tb, {'train' : train_Eloss}, epoch)
#                 writer.add_scalars('%s_IoU_metric' % args.tb, {'train' : train_metric}, epoch)

#             for i, data in enumerate(data_loader["valid"]):
#                 model.eval()
#                 images, masks4 = data
#                 images, masks4 = images.cuda(), masks4.cuda()
#                 outs = model(images)
#                 Heat_map, _, _ = outs
#                 pred_masks = cluster.run(outs, masks4)
#                 smooth_loss = SmoothLoss(outs, masks4)
#                 center_loss = CenterLoss(outs, masks4)
#                 embedding_loss = EmbeddingLoss(pred_masks, masks4)
#                 loss = smooth_loss + center_loss + embedding_loss
#                 valid_Sloss += smooth_loss.item()
#                 valid_Closs += center_loss.item()
#                 valid_Eloss += embedding_loss.item()
#                 valid_Tloss += loss.item()
#                 valid_metric += IoU_metric(masks4, (pred_masks>0.5).float())
#                 if args.vis == True:
#                     Visual(pred_masks, Heat_map.reshape(pred_masks.size()), i, 'valid')
#             valid_Sloss = valid_Sloss/len(data_loader["valid"])
#             valid_Closs = valid_Closs/len(data_loader["valid"])
#             valid_Eloss = valid_Eloss/len(data_loader["valid"])
#             valid_Tloss = valid_Tloss/len(data_loader["valid"])
#             valid_metric = valid_metric/len(data_loader["train"])
#             print("epoch: ", epoch, "TotalLoss: %.4f" % valid_Tloss, "IoU_metric: %.4f" % valid_metric)
#             print("SLoss: %.4f" % valid_Sloss, "CLoss: %.4f" % valid_Closs, "ELoss: %.4f" % valid_Eloss)
#             if args.tb != "None":
#                 writer.add_scalars('%s_Total_loss' % args.tb, {'valid' : valid_Tloss}, epoch)
#                 # writer.add_scalars('%s_Smooth_loss' % args.tb, {'valid' : valid_Sloss}, epoch)
#                 # writer.add_scalars('%s_Center_loss' % args.tb, {'valid' : valid_Closs}, epoch)
#                 # writer.add_scalars('%s_Embeddidg_loss' % args.tb, {'valid' : valid_Eloss}, epoch)
#                 writer.add_scalars('%s_IoU_metric' % args.tb, {'valid' : valid_metric}, epoch)
            
#         for i, data in enumerate(data_loader["test"]): 
#             model.eval()
#             images, masks4 = data
#             images, masks4 = images.cuda(), masks4.cuda()
#             outs = model(images)
#             pred_masks = cluster.test_run(outs, iter_in_epoch=i)
#             test_metric += IoU_metric(masks4, pred_masks)
#         test_metric = test_metric/len(data_loader["train"])
#         print("test metric: %.4f" % test_metric)
#         if args.tb != "None":
#             writer.add_scalars('%s_IoU_metric' % args.tb, {'test' : test_metric}, epoch)

if __name__ == "__main__":
    train()

