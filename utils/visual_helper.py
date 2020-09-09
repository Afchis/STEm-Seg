import random

from PIL import Image

import torch
import torchvision.transforms as transforms


to_pil = transforms.ToPILImage()
resize = transforms.Resize((512, 512), interpolation=0)

colors = {
    "0white" : torch.tensor([1., 1., 1.]).reshape(3, 1, 1, 1).expand(3, 8, 128, 128).cuda(),
    "1red" : torch.tensor([1., 0., 0.]).reshape(3, 1, 1, 1).expand(3, 8, 128, 128).cuda(),
    "2green" : torch.tensor([0., 1., 0.]).reshape(3, 1, 1, 1).expand(3, 8, 128, 128).cuda(),
    "3blue" : torch.tensor([0., 0., 1.]).reshape(3, 1, 1, 1).expand(3, 8, 128, 128).cuda(),
    "4yellow" : torch.tensor([1., 1., 0.]).reshape(3, 1, 1, 1).expand(3, 8, 128, 128).cuda(),
    "5purple" : torch.tensor([1., 0., 1.]).reshape(3, 1, 1, 1).expand(3, 8, 128, 128).cuda(),
    "6?" : torch.tensor([0., 1., 1.]).reshape(3, 1, 1, 1).expand(3, 8, 128, 128).cuda(),
    "7??" : torch.tensor([1., 0.5, 0.5]).reshape(3, 1, 1, 1).expand(3, 8, 128, 128).cuda(),
    "8???" : torch.tensor([0.5, 1, 0.5]).reshape(3, 1, 1, 1).expand(3, 8, 128, 128).cuda(),
    "9????" : torch.tensor([0.5, 0.5, 1]).reshape(3, 1, 1, 1).expand(3, 8, 128, 128).cuda(),
    "10?????" : torch.tensor([0.5, 0.75, 0.75]).reshape(3, 1, 1, 1).expand(3, 8, 128, 128).cuda(),
    "11?????" : torch.tensor([0.75, 0.5, 0.75]).reshape(3, 1, 1, 1).expand(3, 8, 128, 128).cuda(),
    "12?????" : torch.tensor([0.75, 0.75, 0.5]).reshape(3, 1, 1, 1).expand(3, 8, 128, 128).cuda(),
}

l_to_rgb = {
    "0white" : torch.tensor([1., 1., 1.]).cuda(),
    "1red" : torch.tensor([1., 0., 0.]).cuda(),
    "2green" : torch.tensor([0., 1., 0.]).cuda(),
    "3blue" : torch.tensor([0., 0., 1.]).cuda(),
    "4yellow" : torch.tensor([1., 1., 0.]).cuda(),
    "5purple" : torch.tensor([1., 0., 1.]).cuda(),
    "6?" : torch.tensor([0., 1., 1.]).cuda(),
    "7??" : torch.tensor([1., 0.5, 0.5]).cuda(),
    "8???" : torch.tensor([0.5, 1, 0.5]).cuda(),
    "9????" : torch.tensor([0.5, 0.5, 1]).cuda(),
    "10?????" : torch.tensor([0.5, 0.75, 0.75]).cuda(),
    "11??????" : torch.tensor([0.75, 0.5, 0.75]).cuda(),
    "12???????" : torch.tensor([0.75, 0.75, 0.5]).cuda(),
}

color_names = ["1red", "2green", "3blue", "4yellow", "5purple", "6?", \
               "7??", "8???", "9????", "10?????", "11??????", "12???????"]

def Visual(pred_masks, Heat_map, i, mode):
    pred_masks = pred_masks[0].permute(1, 2, 3, 0)
    rgb_pred = torch.zeros([pred_masks.size(0), pred_masks.size(1), pred_masks.size(2), 3]).cuda()
    for c in range(pred_masks.size(3)):
        rgb_pred += pred_masks[:, :, :, c].unsqueeze(3)*l_to_rgb[color_names[c]]
    rgb_pred = rgb_pred.permute(0, 3, 1, 2) # [t, c, h, w]
    rgb_pred = to_pil(rgb_pred[4].cpu()).convert("RGBA")
    rgb_pred.save("ignore/visual/" + mode + "/" + mode + "%i.png" % i)

    Heat_map = Heat_map[0, :, 4]
    Heat_map = to_pil(Heat_map.cpu()).convert("RGBA")
    Heat_map.save("ignore/visual/" + mode + "_heatmap/" + mode  + "_heatmap%i.png" % i)

def Visual_inference(pred_clusters, images, i, mode, batch=0):
    out = torch.zeros([3, 8, 128, 128]).cuda()
    pred_clusters = pred_clusters[batch]
    images = images[batch]
    for ch in range(len(pred_clusters)):
        color = colors[color_names[ch]]
        color = pred_clusters[ch]*color
        out += color
    rgb_out = list()
    for time in range(out.size(1)):
        img_time = to_pil(images[time].cpu()).convert("RGBA")
        out_time = resize(to_pil(out[:, time].cpu()).convert("RGBA"))
        rgb_out.append(Image.blend(img_time, out_time, 0.5))
    rgb_out[0].save("ignore/visual/" + "inferense_%s/" % mode + "inferense_%s" % mode + "_%i.gif" % i, 
        save_all=True, append_images=rgb_out[1:], optimize=True, duration=400, loop=0)
    # rgb_out.save("ignore/visual/" + "inferense_%s/" % mode + "inferense_%s" % mode + "_%i_" % i + "%i.png" % time)

