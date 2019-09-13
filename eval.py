import os
import scipy.misc as misc
import torch
from network import MSRResNetSlim


def main():
    #data_dir = '/mnt/lustre/niuyazhe/code/github/RCAN/RCAN_TestCode/LR/LRBI/DIV2K_V/x4'
    data_dir = '/mnt/lustre/niuyazhe/data/DIV2K/DIV2K_test_LR_bicubic_900'
    output_dir = '/mnt/lustre/niuyazhe/code/github/RCAN/RCAN_TestCode/SR_AIM/' + 'Final'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    model_path = '/mnt/lustre/niuyazhe/code/github/RCAN/RCAN_TrainCode/experiment_aim/PN1_D9/model/model_41.pt'
    cfg = [6, 39, 46, 42, 42, 53, 49, 52, 49, 37]
    model = MSRResNetSlim(cfg=cfg)
    new_state_dict = {k[7:]: v for k, v in torch.load(model_path).items()}  # remove module prefix
    #new_state_dict = torch.load(model_path)
    model.load_state_dict(new_state_dict)
    model.eval()
    model.cuda()
    for idx, path in enumerate(os.listdir(data_dir)):
        img = misc.imread(os.path.join(data_dir, path))
        img = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0).cuda()
        with torch.no_grad():
            output = model(img)
        output = output.clamp(0, 255).round()
        output = output[0].byte().permute(1, 2, 0).cpu().numpy()
        path = ''.join(path.split('_LRBI_'))
        misc.imsave(os.path.join(output_dir, path), output)
        print(idx)


if __name__ == "__main__":
    main()
