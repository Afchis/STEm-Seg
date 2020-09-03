import random

from PIL import Image

import torch
import torchvision.transforms as transforms


to_pil = transforms.ToPILImage()
resize = transforms.Resize((128, 128), interpolation=0)

colors = {
    "0white" : torch.tensor([1., 1., 1.]).reshape(1, 3, 1, 1, 1).expand(1, 3, 8, 128, 128).cuda(),
    "1red" : torch.tensor([1., 0., 0.]).reshape(1, 3, 1, 1, 1).expand(1, 3, 8, 128, 128).cuda(),
    "2green" : torch.tensor([0., 1., 0.]).reshape(1, 3, 1, 1, 1).expand(1, 3, 8, 128, 128).cuda(),
    "3blue" : torch.tensor([0., 0., 1.]).reshape(1, 3, 1, 1, 1).expand(1, 3, 8, 128, 128).cuda(),
    "4yellow" : torch.tensor([1., 1., 0.]).reshape(1, 3, 1, 1, 1).expand(1, 3, 8, 128, 128).cuda(),
    "5purple" : torch.tensor([1., 0., 1.]).reshape(1, 3, 1, 1, 1).expand(1, 3, 8, 128, 128).cuda(),
    "6purple" : torch.tensor([0., 1., 1.]).reshape(1, 3, 1, 1, 1).expand(1, 3, 8, 128, 128).cuda(),
}
color_names = ["0white", "1red", "2green", "3blue", "4yellow", "5purple", "6purple"]

def _rand_color_():
    r = random.uniform(0., 1.)
    g = random.uniform(0., 1.)
    b = random.uniform(0., 1.)
    return torch.tensor([r, g, b]).cuda()


def Visual(pred_masks, Heat_map, i, mode):
    pred_masks = pred_masks[0].permute(1, 2, 3, 0)
    rgb_pred = torch.zeros([pred_masks.size(0), pred_masks.size(1), pred_masks.size(2), 3]).cuda()
    for c in range(pred_masks.size(3)):
        rgb_pred += pred_masks[:, :, :, c].unsqueeze(3)*_rand_color_()
    rgb_pred = rgb_pred.permute(0, 3, 1, 2) # [t, c, h, w]
    rgb_pred = to_pil(rgb_pred[0].cpu()).convert("RGBA")
    rgb_pred.save("ignore/visual/" + mode + "/" + mode + "%i.png" % i)

    Heat_map = Heat_map[0, :, 0]
    Heat_map = to_pil(Heat_map.cpu()).convert("RGBA")
    Heat_map.save("ignore/visual/" + mode + "_heatmap/" + mode  + "_heatmap%i.png" % i)


# def Visual_clusters(visual_list, iter_in_epoch):
#     out = torch.zeros([1, 3, 8, 128, 128]).cuda()
#     for i in range(len(visual_list)):
#         color = colors[color_names[i]]
#         color = visual_list[i] * color
#         out += color
#     out = to_pil(out[0, :, 0].cpu()).convert("RGBA")
#     out.save("ignore/visual/" + 'color_test' + "/" + 'color_test' + "%i.png" % iter_in_epoch)