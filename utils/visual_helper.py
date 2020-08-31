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


def Visual(pred_masks, Heat_map, i, mode):
    # images = resize(to_pil(images[0][4].cpu()))
    out = to_pil(pred_masks[0][0].cpu()).convert("RGBA")
    # mask = Image.new("RGBA", out.size, 128)
    # im = Image.composite(images, out, mask)
    out.save("ignore/visual/" + mode + "/" + mode + "%i.png" % i)
    if mode != 'test':
        heat = to_pil(Heat_map[0][0].cpu()).convert("RGBA")
        heat.save("ignore/visual/" + mode + "_heatmap" + "/" + mode + "_heatmap" + "%i.png" % i)
        max_heat = to_pil((Heat_map[0][0] == Heat_map[0][0].max()).float().cpu()).convert("RGBA")
        max_heat.save("ignore/visual/" + mode + "_heatmap" + "/" + mode + "_heatmap" + "%imax.png" % i)

def Visual_clusters(visual_list, iter_in_epoch):
    out = torch.zeros([1, 3, 8, 128, 128]).cuda()
    for i in range(len(visual_list)):
        color = colors[color_names[i]]
        color = visual_list[i] * color
        out += color
    out = to_pil(out[0, :, 0].cpu()).convert("RGBA")
    out.save("ignore/visual/" + 'color_test' + "/" + 'color_test' + "%i.png" % iter_in_epoch)

