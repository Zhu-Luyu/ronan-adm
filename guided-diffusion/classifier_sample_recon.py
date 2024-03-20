"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

"""
Combine classifier_sample.py (from guided-diffusion) and main.py (from RONAN)
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

# For RONAN
from inference_utils import *
import cv2
from PIL import Image
from torch.cuda.amp import GradScaler

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)
    
    # 通过 get_init_noise 函数获取初始噪声 init_noise。
    init_noise=th.randn(*(args.batch_size, 3, args.image_size, args.image_size), device="cuda")#args.image_size=256

    # get image0
    shiba_img = cv2.imread(args.input_selection_name)
    b,g,r = cv2.split(shiba_img)
    shiba_img = cv2.merge([r, g, b])
    shiba_img = cv2.resize(shiba_img, (args.image_size,args.image_size), interpolation=cv2.INTER_AREA)
    #shiba_img = cv2.resize(shiba_img, (32,32), interpolation=cv2.INTER_AREA)
    shiba_img_show = Image.fromarray(shiba_img)
    shiba_img_show.save("input_selection_name_img_show3.jpg")
    shiba_img = shiba_img/255
    shiba_img = th.from_numpy(shiba_img).cuda().clamp(0, 1).permute(2,0,1).unsqueeze(0).float()
    image0 = shiba_img
    image0 = image0.detach()
    
    # 优化器
    cur_noise = th.nn.Parameter(init_noise.clone().detach().requires_grad_(True)).cuda()
    optimizer = th.optim.Adam([cur_noise], lr=args.lr)
    
    # 损失函数
    criterion = th.nn.MSELoss(reduction='none')
    
    # 如果启用了混合精度，使用 GradScaler 来管理梯度的缩放，以优化训练过程。
    if args.mixed_precision:
        scaler = GradScaler()
        
    import time
    args.measure = float("inf")   
    
    experiment_folder_path = create_experiment_folder(args.input_selection)

    logger.log("sampling...")
    
    for i in range(args.num_iter):
        start_time = time.time()
        logger.log(f"step:{i}")
        
        # 
        #     with torch.autocast(device_type='cuda', dtype=torch.float16):
            
        # 用于存储生成的图像和对应的标签。
        all_images = []
        all_labels = []
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            # classes = th.randint(
            #     low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            # )
            # 指定标签的修改
            if args.labels is not None and len(args.labels) > 0:
                # 确保提供的标签数量与批次大小匹配
                if len(args.labels) != args.batch_size:
                    raise ValueError("The number of specified labels must match the batch size.")
                classes = th.tensor(args.labels, device=dist_util.dev())
            else: # 随机采样
                classes = th.randint(low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev())
                
            model_kwargs["y"] = classes
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            
            
            if args.mixed_precision:
                logger.log("sample mixed_precision")
                sample = sample_fn(
                    model_fn,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    noise=cur_noise,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn,
                    device=dist_util.dev(),
                )
                
                # 保存每一次重建的图像
                step_num_str=str(i)
                file_name = f"image_{step_num_str}_{args.input_selection}_{args.distance_metric}_{str(args.lr)}.png"
                image1 = (sample + 1) / 2
                save_image_to_experiment_folder(image1, experiment_folder_path, file_name)
                
                # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                # sample = sample.permute(0, 2, 3, 1)
                # sample = sample.contiguous()
                
                # logger.log(f"image0 shape:{image0.shape}") # torch.Size([1, 3, 1, 1])
                # logger.log(f"sample shape:{sample.shape}")
                loss = criterion(image0,sample).mean()
            
            # 根据指定的策略（如 "min" 或 "mean"），更新最小损失值 args.measure。
            min_value = criterion(image0,sample).mean(-1).mean(-1).mean(-1).min()
            mean_value = criterion(image0,sample).mean()
            if (args.strategy == "min") and (min_value < args.measure):
                args.measure = min_value
            if (args.strategy == "mean") and (mean_value < args.measure):
                args.measure = mean_value
                
            write_loss_to_csv(experiment_folder_path, i, loss.item(), min_value.item(), mean_value.item())
            
            # 根据是否使用混合精度，使用相应的方式进行梯度计算和优化器更新。
            if args.mixed_precision:
                logger.log("optimizer mixed_precision")
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 打印一次迭代的时间。
            end_time = time.time()
            logger.log(f"time for one iter: {end_time-start_time}")
            
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            logger.log(f"created {len(all_images) * args.batch_size} samples")

        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            # np.savez(out_path, arr, label_arr)
            np.savez(out_path, images=arr, labels=label_arr)
            
        dist.barrier()
        logger.log(f"step {i} sampling complete")
        torch.cuda.empty_cache()

    


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    # 自定义标签
    parser.add_argument('--labels', type=int, nargs='+', help='List of labels for sampling')
    
    # From RONAN/main.py
    parser.add_argument("--input_selection", default="", type=str, help="The path of dev set.")
    parser.add_argument("--distance_metric", default="l2", type=str, help="The path of dev set.")
    parser.add_argument("--model_type", default="ddpm_cifar10", type=str, help="The path of dev set.")
    parser.add_argument("--model_path_", default=None, type=str, help="The path of dev set.")

    parser.add_argument("--lr", default=1e-2, type=float, help="")
    parser.add_argument("--dataset_index", default=None, type=int, help="")
    parser.add_argument("--bs", default=8, type=int, help="")
    parser.add_argument("--write_txt_path", default=None, type=str, help="The path of dev set.")
    parser.add_argument("--num_iter", default=2000, type=int, help="The path of dev set.")
    parser.add_argument("--strategy", default="mean", type=str, help="The path of dev set.")
    parser.add_argument("--mixed_precision", action="store_true", help="The path of dev set.")
    parser.add_argument("--sd_prompt", default=None, type=str, help="The path of dev set.")
    parser.add_argument("--input_selection_url", default=None, type=str, help="The path of dev set.")
    parser.add_argument("--input_selection_name", default=None, type=str, help="The path of dev set.")
    parser.add_argument("--input_selection_model_type", default=None, type=str, help="The path of dev set.")
    parser.add_argument("--input_selection_model_path", default=None, type=str, help="The path of dev set.")
    
    return parser


if __name__ == "__main__":
    main()
