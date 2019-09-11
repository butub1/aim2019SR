#-*-coding:utf-8-*-
import os
import cv2
import time
import sys
import argparse
import torch
from network import MSRResNetSlim
from SRResNet import MSRResNet


def test_velocity(args):
    if not os.path.isdir(args.model_dir):
        raise ValueError
    pt = prune_msg = ""
    for filename in os.listdir(args.model_dir):
        if filename[-2:] == "pt":
            pt = os.path.join(args.model_dir, filename)
        if filename[-3:] == "txt" and filename.find("prune") >= 0:
            prune_msg = os.path.join(args.model_dir, filename)
    print(pt)
    print(prune_msg)
    cfg = []
    with open(prune_msg, "r") as f:
        line = f.readlines()[1]
        cfg = [int(x) for x in line[1:-2].split(',')]

    print(cfg)
    model = MSRResNetSlim(cfg=cfg)
    model.cuda()
    model.eval()
    model_bm = MSRResNet()
    model_bm.cuda()
    model_bm.eval()

    img = cv2.imread('./0001x4.png')
    img = torch.FloatTensor(img).cuda()
    img = img.permute(2, 0, 1).unsqueeze(0)
    print(img.shape)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    time_model = 0.
    time_model_record = 0.
    for i in range(100):
        begin = time.time()
        start.record()
        with torch.no_grad():
            model(img)
        end.record()
        torch.cuda.synchronize()
        time_model += time.time() - begin
        time_model_record += start.elapsed_time(end)

    time_model = 0.
    time_model_record = 0.
    for i in range(10):
        begin = time.time()
        start.record()
        with torch.no_grad():
            model(img)
        end.record()
        torch.cuda.synchronize()
        time_model += time.time() - begin
        time_model_record += start.elapsed_time(end)
    print("time:", time_model / 10)
    print("time_record:", time_model_record / 10)
    time_model = 0.
    time_model_record = 0.
    for i in range(10):
        begin = time.time()
        start.record()
        with torch.no_grad():
            model_bm(img)
        end.record()
        torch.cuda.synchronize()
        time_model += time.time() - begin
        time_model_record += start.elapsed_time(end)
    print("time bm:", time_model / 10)
    print("time_record bm:", time_model_record / 10)


def test_quantity(args):
    if not os.path.isdir(args.model_dir):
        raise ValueError
    pt = prune_msg = ""
    for filename in os.listdir(args.model_dir):
        if filename[-2:] == "pt":
            pt = os.path.join(args.model_dir, filename)
        if filename[-3:] == "txt" and filename.find("prune") >= 0:
            prune_msg = os.path.join(args.model_dir, filename)
    print(pt)
    print(prune_msg)
    cfg = []
    with open(prune_msg, "r") as f:
        line = f.readlines()[1]
        cfg = [int(x) for x in line[1:-2].split(',')]
    print(cfg)
    model = MSRResNetSlim(cfg=cfg)
    print("model parameters quantity:", _test_quantity(model))
    model_bm = MSRResNet()
    print("model_bm parameters quantity:", _test_quantity(model_bm))


def _test_quantity(net):
    return sum(map(lambda x: x.numel(), net.parameters()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/mnt/lustre/niuyazhe/code/github/RCAN/RCAN_TrainCode/code/pruned_model/prune3_RE_N2/")

    args = parser.parse_args()
    test_velocity(args)
    test_quantity(args)


