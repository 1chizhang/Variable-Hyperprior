
from tqdm import tqdm
import argparse
import math
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
import os
from compressai.datasets import ImageFolder
# from Cheng2020Attention import Cheng2020Attention
# from Channelwise import Channelwise_DCNv4 as train_model
from ScaleHyperprior import ScaleHyperprior as train_model
import time
import torch._dynamo
torch._dynamo.config.suppress_errors = True
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target, lmbda):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target) * 255 ** 2
        out["loss"] = lmbda  * out["mse_loss"] + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, noise, stage
):
    model.train()
    device = next(model.parameters()).device
    pbar = tqdm(total=len(train_dataloader.dataset), desc="Training Progress")

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        if stage > 1:
            s = random.randint(0, model.levels - 1)  # choose random level from [0, levels-1]
        else:
            s = model.levels - 1

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d, noise, stage, s)

        out_criterion = criterion(out_net, d, model.lmbda[s])
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        pbar.update(len(d))
        pbar.set_description(
            f"Train epoch {epoch} stage{stage}: [{i*len(d)}/{len(train_dataloader.dataset)} "
            f"({100. * i / len(train_dataloader):.0f}%)]"
            f" \tlambda: {model.lmbda[s]} s: {s:.3f}, scale: {model.Gain.data[s].detach().cpu().numpy():0.4f}, |"
            f'\tLoss: {out_criterion["loss"].item():.3f} |'
            f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
            f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
            f"\tAux loss: {aux_loss.item():.2f}"
            f"\tTime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
        )

# 循环结束后关闭进度条
    pbar.close()


def test_epoch(epoch, test_dataloader, model, criterion, noise, stage):
    model.eval()
    device = next(model.parameters()).device
    print("stage:{}, noise quantization:{}".format(stage, noise))


    loss_total = 0
    bpp_loss_total = 0
    mse_loss_total = 0

    with torch.no_grad():
        for s in range(model.levels):
            loss = AverageMeter()
            bpp_loss = AverageMeter()
            mse_loss = AverageMeter()
            aux_loss = AverageMeter()
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(x=d, noise=noise, stage=stage, s=s)
                out_criterion = criterion(out_net, d, model.lmbda[s])

                aux_loss.update(model.aux_loss().item())
                bpp_loss.update(out_criterion["bpp_loss"].item())
                loss.update(out_criterion["loss"].item())
                mse_loss.update(out_criterion["mse_loss"].item())

            loss_total += loss.avg
            bpp_loss_total += bpp_loss.avg
            mse_loss_total += mse_loss.avg
			
            print(
                f"Test epoch {epoch}, lambda: {model.lmbda[s]}, s: {s}, scale: {model.Gain.data[s].cpu().numpy():0.4f}, stage {stage}:"
                f"\tLoss: {loss.avg:.3f} |"
                f"\tMSE loss: {mse_loss.avg:.3f} |"
                f"\tBpp loss: {bpp_loss.avg:.4f} |"
                f"\tAux loss: {aux_loss.avg:.4f}"
            )
    print(
        f"Test epoch {epoch} : Total Average losses:"
        f"\tLoss: {loss_total:.3f} |"
        f"\tMSE loss: {mse_loss_total:.3f} |"
        f"\tBpp loss: {bpp_loss_total:.4f} \n"
    )

    return loss_total


def save_checkpoint(state, is_best, filename="checkpoint_elic.pth.tar"):
    torch.save(state, filename)
    if is_best:
        best_filename = filename.replace(".pth.tar", "_best.pth.tar")
        shutil.copyfile(filename,best_filename)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument('--stage', default=0, type=int, help='trainning stage')
    parser.add_argument("--ste", default=0, type=int, help="Using ste round in the finetune stage")
    parser.add_argument('--loadFromPretrainedSinglemodel', default=0, type=int, help='load models from single rate')
    parser.add_argument("--refresh", default=0, type=int, help="refresh the setting of optimizer and epoch",
    )
    #add compile boolen
    parser.add_argument('--compile',action='store_true', default=False)
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size, pad_if_needed=True, padding_mode='reflect'), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train2017", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="kodak", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = train_model()
    model_name = str(net.__class__.__name__)
    net = net.to(device)
    

        



    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=10)
    criterion = RateDistortionLoss()

    last_epoch = 0
    best_loss = float("inf")
    if args.checkpoint:  # load from previous checkpoint
        if args.loadFromPretrainedSinglemodel:
            print("Loading single lambda pretrained checkpoint: ", args.checkpoint)
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if "state_dict" in checkpoint:
                ckpt = checkpoint["state_dict"]
            else:
                ckpt = checkpoint
            model_dict = net.state_dict()
            pretrained_dict = {k: v for k, v in ckpt.items() if
                               k in model_dict.keys() and v.shape == model_dict[k].shape}

            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)
        else:
            print("Loading: ", args.checkpoint)
            checkpoint = torch.load(args.checkpoint, map_location=device)
            last_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint["best_loss"]
            net.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    if args.refresh:
        optimizer.param_groups[0]['lr'] = args.learning_rate
        aux_optimizer.param_groups[0]['lr'] = args.aux_learning_rate
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=10)
        print("pretrained epoch: ", last_epoch)
        last_epoch = 0

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
    if args.compile:
        net = torch.compile(net)
    stage = args.stage
    noise = True
    ste = False  
    if args.ste or stage>2:
        ste = True
        noise = False

    txt_name = model_name+str(stage)+str(noise)+str(ste)+".txt"
    if os.path.exists(txt_name):
        shutil.copyfile(txt_name, txt_name+"_old.txt")
    #save the ARGs
    f = open(txt_name, "a")
    f.write(str(args)+"\n")
    f.close()
    


    for epoch in range(last_epoch, args.epochs):
        print("noise quant: {}, ste quant:{}, stage:{}".format(noise, ste, stage))
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        print('Epoch Start Time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        #save epoch start time
        f = open(txt_name, "a")
        f.write("Epoch Start Time: "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+"\n")
        f.close()



        loss = test_epoch(epoch, test_dataloader, net, criterion, noise, stage, )
        f = open(txt_name, "a")
        f.write(str(loss)+"Epoch:"+str(epoch)+"\n")
        #writr lr
        f.write("Learning rate: "+str(optimizer.param_groups[0]['lr'])+"\n")
        f.close()

        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
			noise,
            stage,
        )
        #save epoch end time
        f = open(txt_name, "a")
        f.write("Epoch End Time: "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+"\n")
        f.close()

        loss = test_epoch(epoch, test_dataloader, net, criterion, noise, stage, )
        f = open(txt_name, "a")
        f.write(str(loss)+"Epoch:"+str(epoch)+"\n")
        #writr lr
        f.write("Learning rate: "+str(optimizer.param_groups[0]['lr'])+"\n")
        f.close()




        lr_scheduler.step(loss)


        #if loss = nan, then stop training
        if math.isnan(loss):
            break
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
					"best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                filename=model_name+str(stage)+str(noise)+str(ste)+"_checkpoint.pth.tar",
            )


if __name__ == "__main__":
    main(sys.argv[1:])
