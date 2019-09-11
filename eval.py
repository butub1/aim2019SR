import os
import cv2
import torch
from network import MSRResNetSlim


def main():
    #data_dir = '/mnt/lustre/niuyazhe/code/github/RCAN/RCAN_TestCode/LR/LRBI/DIV2K_V/x4'
    data_dir = '/mnt/lustre/niuyazhe/data/DIV2K/DIV2K_test_LR_bicubic_900'
    output_dir = '/mnt/lustre/niuyazhe/code/github/RCAN/RCAN_TestCode/SR_AIM/' + 'P3_2_TEST'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    model_path = '/mnt/lustre/niuyazhe/code/github/RCAN/RCAN_TrainCode/experiment_aim/PRUNE2/model/model_26.pt'
    cfg = [6, 39, 45, 43, 44, 50, 47, 54, 50, 37]
    model = MSRResNetSlim(cfg=cfg)
    new_state_dict = {k[7:]: v for k, v in torch.load(model_path).items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    model.cuda()
    for idx, path in enumerate(os.listdir(data_dir)):
        img = cv2.imread(os.path.join(data_dir, path))
        img = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0).cuda()
        with torch.no_grad():
            output = model(img)
        output = output[0].permute(1, 2, 0).cpu().numpy()
        path = ''.join(path.split('_LRBI_'))
        cv2.imwrite(os.path.join(output_dir, path), output)
        print(idx)


if __name__ == "__main__":
    main()
