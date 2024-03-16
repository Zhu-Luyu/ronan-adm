# 引入 diffusers、torch、torchvision、dnnlib、pickle、pl_bolts 等，用于加载和处理不同类型的生成模型。
from diffusers import DDIMPipeline
import torch
from torchvision import transforms
from dcgan import DCGAN
import dnnlib
import dnnlib.tflib as tflib
import pickle
from pl_bolts.models.autoencoders import VAE
from diffusers import StableDiffusionPipeline
from cm_inference import cm_inference
from cm.script_util import model_and_diffusion_defaults,create_model_and_diffusion,args_to_dict_
from cm.random_util import get_generator
import argparse

"""
这段代码定义了几个函数，用于处理不同类型的生成模型。
这些函数提供了一种通用的接口，用于处理不同类型的生成模型。通过调用 get_model 函数加载预训练的模型，然后使用 get_init_noise 函数生成初始噪声，最后使用 from_noise_to_image 函数将噪声转换为图像。
总的来说，这段代码提供了一个灵活的框架，用于处理各种生成模型，并将噪声转换为图像。通过选择适当的模型类型和加载相应的预训练权重，可以使用这些函数生成不同风格和领域的图像。
"""

def get_init_noise(args,model_type,model,bs=1):
    """
    根据模型类型和批量大小，生成初始噪声。
    对于不同的模型类型（如 "ddpm_cifar10"、"dcgan_cifar10"、"styleganv2ada_cifar10"、"vae_cifar10"、"sd" 和 "cm"），生成相应大小和形状的随机噪声张量。
    生成的噪声张量将在 CUDA 设备上创建。
    """
    if model_type in ["ddpm_cifar10"]:
        init_noise = torch.randn(bs, model.unet.in_channels, model.unet.sample_size, model.unet.sample_size).cuda()
    elif model_type in ["dcgan_cifar10"]:
        init_noise = torch.randn(bs, model.nz, 1, 1).cuda()
    elif model_type in ["styleganv2ada_cifar10"]:
        init_noise = torch.randn([bs, model.z_dim]).cuda()
    elif model_type in ["vae_cifar10"]:
        init_noise = torch.randn([bs, model.latent_dim]).cuda()
    elif model_type in ["sd"]:
        height = model.unet.config.sample_size * model.vae_scale_factor
        width = model.unet.config.sample_size * model.vae_scale_factor
        init_noise = torch.randn([bs, model.unet.in_channels, height // model.vae_scale_factor, width // model.vae_scale_factor]).cuda()
    elif "cm" in model_type:
        init_noise = torch.randn(*(bs, 3, 64, 64)).cuda()

    return init_noise

def from_noise_to_image(args,model,noise,model_type):
    """
    根据模型类型，将噪声转换为图像。
    对于不同的模型类型，使用相应的方法将噪声转换为图像。
    对于某些模型类型，可能需要对生成的图像进行后处理，如调整大小或裁剪。
    """
    if model_type in ["ddpm_cifar10"]:
        image = model.input2output(noise,num_inference_steps=50)
    elif model_type in ["dcgan_cifar10"]:
        image = model.input2output(noise)
        image = transforms.Resize(32)(image)
    elif model_type in ["styleganv2ada_cifar10"]:
        label = torch.zeros([noise.shape[0], model.c_dim]).cuda()
        image = model(noise, label, noise_mode='none')
        image = (image / 2 + 0.5).clamp(0, 1)
        image = transforms.Resize(32)(image)
    elif model_type in ["vae_cifar10"]:
        image = model.decoder(noise)
        image = image*args.vae_t_std + args.vae_t_mean
        image = image.clamp(0, 1)
    elif model_type in ["sd"]:
        image = model.latent2output(noise)
    elif "cm" in model_type:
        image = cm_inference(model,noise)
    return image


def get_model(model_type,model_path,args):
    """
    根据模型类型和模型路径，加载预训练的生成模型。
    支持多种模型类型，如 "ddpm_cifar10"、"dcgan_cifar10"、"styleganv2ada_cifar10"、"vae_cifar10"、"cm" 和 "sd"。
    对于每种模型类型，使用相应的库和方法加载预训练的模型权重。
    根据需要，对加载的模型进行必要的设置，如将模型移动到 CUDA 设备、设置评估模式等。
    对于某些模型类型，可能需要额外的参数设置或数据预处理。
    """
    if model_type == "ddpm_cifar10":
        model_id = "google/ddpm-cifar10-32"
        model_id_1 = "google/ddpm-cifar10-32"
        cur_model = DDIMPipeline.from_pretrained(model_id).to("cuda")
        #ddim_1 = DDIMPipeline.from_pretrained(model_id).to("cuda")
        cur_model.unet.eval()

    elif model_type == "dcgan_cifar10":
        ngpu = 1
        cur_model = DCGAN(ngpu)
        if model_path:
            cur_model.load_state_dict(torch.load(model_path))
        else:
            cur_model.load_state_dict(torch.load("./dcgan_weights/netG_epoch_24.pth"))
        cur_model = cur_model.cuda()
        cur_model.eval()

    elif model_type == "styleganv2ada_cifar10":
        tflib.init_tf()
        network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl"
        print('Loading networks from "%s"...' % network_pkl)
        with dnnlib.util.open_url(network_pkl) as fp:
            #_G, _D, Gs = pickle.load(fp)
            G = pickle.load(fp)['G_ema'].cuda()  # torch.nn.Module
        cur_model = G.eval()
        z = torch.randn([args.bs, cur_model.z_dim]).cuda()    # latent codes
        label = torch.zeros([args.bs, cur_model.c_dim]).cuda()                              # class labels (not used in this example)
        class_idx = 9
        label[:, class_idx] = 1
        img = cur_model(z, label,noise_mode='none')
        args.stylegan_class_idx = class_idx

    elif model_type == "vae_cifar10":
        cur_model = VAE(input_height=32)
        print(VAE.pretrained_weights_available())
        cur_model = cur_model.from_pretrained('cifar10-resnet18')
        cur_model.freeze()
        cur_model = cur_model.cuda()
        cur_model = cur_model.eval()

        mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
        std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
        
        channel = 3
        size = 32
        t_mean = torch.FloatTensor(mean).view(channel,1,1).expand(channel, size, size).cuda()
        t_std = torch.FloatTensor(std).view(channel,1,1).expand(channel, size, size).cuda()
        args.vae_t_mean = t_mean
        args.vae_t_std = t_std

    elif "cm" in model_type:

        defaults = dict(
            training_mode="edm",
            generator="determ",
            clip_denoised=True,
            num_samples=10000,
            batch_size=16,
            sampler="heun",
            s_churn=0.0,
            s_tmin=0.0,
            s_tmax=float("inf"),
            s_noise=1.0,
            steps=40,
            model_path="",
            seed=42,
            ts="",
        )
        defaults.update(model_and_diffusion_defaults())
        args_cm = defaults
        args_cm["batch_size"] = args.bs

        if model_type == "cm_cd_lpips":
            args_cm["training_mode"] = "consistency_distillation"
            #args_cm["sampler"] = "multistep"
            args_cm["sampler"] = "onestep"
            args_cm["ts"] = "0,22,39"
            args_cm["steps"] = 40
            args_cm["model_path_"]= "./consistency_models/scripts/cd_imagenet64_lpips.pt"
        elif model_type == "cm_cd_l2":
            args_cm["training_mode"] = "consistency_distillation"
            #args_cm["sampler"] = "multistep"
            args_cm["sampler"] = "onestep"
            args_cm["ts"] = "0,22,39"
            args_cm["steps"] = 40
            args_cm["model_path_"] = "./consistency_models/scripts/cd_imagenet64_l2.pt"
        elif model_type == "cm_ct":
            args_cm["training_mode"] = "consistency_training"
            #args_cm["sampler"] = "multistep"
            args_cm["sampler"] = "onestep"
            args_cm["ts"] = "0,106,200"
            args_cm["steps"] = 201
            args_cm["model_path_"] = "./consistency_models/scripts/cd_imagenet64_l2.pt"
        args_cm["attention_resolutions"] = "32,16,8"
        args_cm["class_cond"] = True
        args_cm["use_scale_shift_norm"] = True
        args_cm["dropout"] = 0.0
        args_cm["image_size"] = 64
        args_cm["num_channels"] = 192
        args_cm["num_head_channels"] = 64
        args_cm["num_res_blocks"] = 3
        args_cm["num_samples"] = 500
        args_cm["resblock_updown"] = True
        args_cm["use_fp16"] = True
        args_cm["weight_schedule"] = "uniform"

        if "consistency" in args_cm["training_mode"]:
            distillation = True
        else:
            distillation = False

        cm_model, diffusion = create_model_and_diffusion(
            **args_to_dict_(args_cm, model_and_diffusion_defaults().keys()),
            distillation=distillation,
        )

        cm_model.load_state_dict(torch.load(args_cm["model_path_"], map_location="cpu"))
        cm_model.cuda()
        if args_cm["use_fp16"]:
            cm_model.convert_to_fp16()
        cm_model.eval()

        if args_cm["sampler"] == "multistep":
            assert len(args_cm["ts"]) > 0
            ts = tuple(int(x) for x in args_cm["ts"].split(","))
        else:
            ts = None
        args_cm["ts_"] = ts
        generator = get_generator(args_cm["generator"], args_cm["num_samples"], args_cm["seed"])
        args_cm["generator_"] = generator
        args_cm["shape"] = (args_cm["batch_size"], 3, args_cm["image_size"], args_cm["image_size"])

        cur_model = (args_cm, cm_model, diffusion)

    elif model_type in ["sd"]:

        model_id = "stabilityai/stable-diffusion-2-base"

        cur_model = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32).to("cuda")
        #sd = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")
        cur_model.unet.eval()
        cur_model.vae.eval()

    return cur_model
