from PIL import Image

import torchvision.transforms as transforms


to_pil = transforms.ToPILImage()
resize = transforms.Resize((128, 128), interpolation=0)


def Visual(pred_masks, i, mode):
    # images = resize(to_pil(images[0][4].cpu()))
    out = to_pil(pred_masks[0][4].cpu()).convert("RGBA")
    # mask = Image.new("RGBA", out.size, 128)
    # im = Image.composite(images, out, mask)
    out.save("ignore/visual/" + mode + "/" + mode + "%ipred.png" % i)
    if mode == 'heat_map':
    	max_heat = to_pil((pred_masks[0][4] == pred_masks[0][4].max()).float().cpu()).convert("RGBA")
    	max_heat.save("ignore/visual/" + mode + "/" + mode + "%ipred_max.png" % i)


