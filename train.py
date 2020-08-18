import torch

from model.model_head import STEmSeg
from utils.cluster import Cluster


images = torch.randn([1, 8, 3, 256, 256])
model = STEmSeg()
outs = model(images)

cluster = Cluster()
pred = cluster.cluster(outs).squeeze()
print(pred)
print("pred.shape: ", pred.shape)

