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
from utils.visual_helper import Visual


parser = argparse.ArgumentParser()

parser.add_argument("--data", type=str, default="./ignore/data/DAVIS", help="data path")
parser.add_argument("--w", type=str, default="default_weights", help="weights name")
parser.add_argument("--time", type=int, default=8, help="Time size")
parser.add_argument("--batch", type=int, default=1, help="Batch size")
parser.add_argument("--workers", type=int, default=1, help="Num workers")
parser.add_argument("--epochs", type=int, default=1000, help="Num epochs")
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
# try:
#     model.load_state_dict(torch.load('ignore/weights/%s.pth' % args.w), )#strict=False
# except FileNotFoundError:
#     print("!!!Create new weights!!!: ", "%s.pth" % args.w)
#     pass
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Params", params)

# init cluster
cluster = Cluster()

# init dataloader
data_loader = Loader(data_path=args.data, batch_size=args.batch, time=args.time, num_workers=args.workers)

# init optimizer and lr_scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0005)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

def train():
    tb_iter = 0
    for epoch in range(args.epochs):
        print("*"*32, "epoch: ",  epoch, "*"*32)
        if args.train:
            for i, data in enumerate(data_loader["train"]):
                model.train()
                images, masks4 = data
                images, masks4 = images.cuda(), masks4.cuda()
                if images.size(0) != args.batch: 
                    break
                outs = model(images)
                Heat_map, _, _ = outs
                pred_masks = cluster.run(outs, masks4)
                smooth_loss = SmoothLoss(outs, masks4)
                center_loss = CenterLoss(outs, masks4)
                embedding_loss = EmbeddingLoss(pred_masks, masks4)
                loss = smooth_loss + center_loss + embedding_loss
                print("iter: ", i, "TotalLoss: %.4f" % loss.item(), \
                    "SLoss: %.4f" % smooth_loss.item(), "CLoss: %.4f" % center_loss.item(), "ELoss: %.4f" % embedding_loss.item())
                loss.backward()
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
                if args.tb != "None":
                    writer.add_scalars('%s_Total_loss' % args.tb, {'train' : loss.item()}, tb_iter)
                    writer.add_scalars('%s_Smooth_loss' % args.tb, {'train' : smooth_loss.item()}, tb_iter)
                    writer.add_scalars('%s_Center_loss' % args.tb, {'train' : center_loss.item()}, tb_iter)
                    writer.add_scalars('%s_Embeddidg_loss' % args.tb, {'train' : embedding_loss.item()}, tb_iter)
                    tb_iter += 1
                if args.vis == True:
                    Visual(pred_masks, i, 'train')
                    Visual(Heat_map.reshape(pred_masks.size()), i, 'heat_map')
                    print(masks4.shape)
                    Visual(masks4, i, 'masks4')
            if epoch % 10 == 0:
                torch.save(model.state_dict(), 'ignore/weights/%s.pth' % args.w)
                print("Save weights: %s.pth" % args.w)

        #     for i, data in enumerate(data_loader["valid"]):
        #         model.eval()
        #         images, masks4 = data
        #         images, masks4 = images.cuda(), masks4.cuda()
        #         if images.size(0) != args.batch: 
        #             break
        #         outs = model(images)
        #         pred_masks = cluster.run(outs, masks4)
        #         smooth_loss = SmoothLoss(outs, masks4)
        #         center_loss = CenterLoss(outs, masks4)
        #         embedding_loss = EmbeddingLoss(pred_masks, masks4)
        #         loss = smooth_loss + center_loss + embedding_loss
        #         print("iter: ", i, "TotalLoss: %.4f" % loss.item(), \
        #             "SLoss: %.4f" % smooth_loss.item(), "CLoss: %.4f" % center_loss.item(), "ELoss: %.4f" % embedding_loss.item())
        #         if args.tb != "None":
        #             writer.add_scalars('%s_Total_loss' % args.tb, {'valid' : loss.item()}, tb_iter)
        #             writer.add_scalars('%s_Smooth_loss' % args.tb, {'valid' : smooth_loss.item()}, tb_iter)
        #             writer.add_scalars('%s_Center_loss' % args.tb, {'valid' : center_loss.item()}, tb_iter)
        #             writer.add_scalars('%s_Embeddidg_loss' % args.tb, {'valid' : embedding_loss.item()}, tb_iter)
        #             tb_iter += 1
        #         if args.vis == True:
        #             Visual(pred_masks, i, 'valid')

            
        # for i, images in enumerate(data_loader["test"]): 
        #     model.eval()
        #     images = images.cuda()
        #     outs = model(images)
        #     pred_masks = cluster.test_run(outs)
        #     if args.vis == True:
        #         Visual(pred_masks, i, 'test')


if __name__ == "__main__":
    train()

