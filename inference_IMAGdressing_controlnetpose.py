import diffusers
from dressing_sd.pipelines.IMAGDressing_v1_pipeline_controlnet import IMAGDressing_v1
import os
import sys
import torch

from PIL import Image
from diffusers import ControlNetModel, UNet2DConditionModel, \
    AutoencoderKL, DDIMScheduler
from torchvision import transforms
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from adapter.attention_processor import CacheAttnProcessor2_0, RefSAttnProcessor2_0, CAttnProcessor2_0
import argparse
from adapter.resampler import Resampler

# 전체 흐름 : 의류 이미지와 포즈 이미지를 입력으로 받아 의상을 입힌 새로운 이미지 생성 후 저장

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# 입력 이미지를 SD 모델 호환 크기로 리사이징징
def resize_img(input_image, max_side=640, min_side=512, size=None,
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    ratio = min_side / min(h, w)
    w, h = round(ratio * w), round(ratio * h)
    ratio = max_side / max(h, w)
    input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
    w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
    h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    return input_image

# 모델 초기화 및 세팅 함수수
def prepare(args):
    generator = torch.Generator(device=args.device).manual_seed(42)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float16, device=args.device)
    tokenizer = CLIPTokenizer.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="text_encoder").to(
        dtype=torch.float16, device=args.device)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder").to(
        dtype=torch.float16, device=args.device)
    unet = UNet2DConditionModel.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="unet").to(
        dtype=torch.float16,
        device=args.device)

    # load ipa weight
    image_proj = Resampler(
        dim=unet.config.cross_attention_dim,
        depth=4,
        dim_head=64,
        heads=12,
        num_queries=16,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4
    )
    image_proj = image_proj.to(dtype=torch.float16, device=args.device)

    # set attention processor
    attn_procs = {}
    st = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = RefSAttnProcessor2_0(name, hidden_size)
        else:
            attn_procs[name] = CAttnProcessor2_0(name, hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    adapter_modules = adapter_modules.to(dtype=torch.float16, device=args.device)
    del st

    ref_unet = UNet2DConditionModel.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="unet").to(
        dtype=torch.float16,
        device=args.device)
    ref_unet.set_attn_processor(
        {name: CacheAttnProcessor2_0() for name in ref_unet.attn_processors.keys()})  # set cache
    ref_unet.set_attn_processor(
        {name: CacheAttnProcessor2_0() for name in ref_unet.attn_processors.keys()})  # set cache

    # weights load
    model_sd = torch.load(args.model_ckpt, map_location="cpu")["module"]

    ref_unet_dict = {}
    unet_dict = {}
    image_proj_dict = {}
    adapter_modules_dict = {}
    for k in model_sd.keys():
        if k.startswith("ref_unet"):
            ref_unet_dict[k.replace("ref_unet.", "")] = model_sd[k]
        elif k.startswith("unet"):
            unet_dict[k.replace("unet.", "")] = model_sd[k]
        elif k.startswith("proj"):
            image_proj_dict[k.replace("proj.", "")] = model_sd[k]
        elif k.startswith("adapter_modules") and 'ref' in k:
            adapter_modules_dict[k.replace("adapter_modules.", "")] = model_sd[k]
        else:
            print(k)

    ref_unet.load_state_dict(ref_unet_dict)
    image_proj.load_state_dict(image_proj_dict)
    adapter_modules.load_state_dict(adapter_modules_dict, strict=False)

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    control_net_openpose = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose",
                                                           torch_dtype=torch.float16).to(device=args.device)
    pipe = IMAGDressing_v1(vae=vae, reference_unet=ref_unet, unet=unet, tokenizer=tokenizer,
                         text_encoder=text_encoder, controlnet=control_net_openpose, image_encoder=image_encoder,
                         ImgProj=image_proj, scheduler=noise_scheduler,
                         safety_checker=StableDiffusionSafetyChecker,
                         feature_extractor=CLIPImageProcessor)

    return pipe, generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IMAGDressing_v1')
    # 파라미터 설정정
    parser.add_argument('--model_ckpt',
                        default="ckpt/IMAGDressing-v1_512.pt",
                        type=str)
    parser.add_argument('--cloth_path', type=str, required=True)
    parser.add_argument('--pose_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default="./output_sd_control")
    parser.add_argument('--device', type=str, default="cuda:0")
    args = parser.parse_args() 

    # svae path
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 파이프 라인 준비
    pipe, generator = prepare(args)
    print('====================== pipe load finish ===================')
    
    num_samples = 1
    clip_image_processor = CLIPImageProcessor()

    img_transform = transforms.Compose([
        transforms.Resize([640, 512], interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    prompt = 'A beautiful woman'
    prompt = prompt + ', best quality, high quality'
    null_prompt = ''
    negative_prompt = 'bare, naked, nude, undressed, monochrome, lowres, bad anatomy, worst quality, low quality'

    # 이미지 전처리리
    clothes_img = Image.open(args.cloth_path).convert("RGB")
    clothes_img = resize_img(clothes_img)
    vae_clothes = img_transform(clothes_img).unsqueeze(0)
    ref_clip_image = clip_image_processor(images=clothes_img, return_tensors="pt").pixel_values
    
    # 프롬프트 설정정
    pose_image = diffusers.utils.load_image(args.pose_path)

    # 이미지 생성
    output = pipe(
        ref_image=vae_clothes,
        prompt=prompt,
        ref_clip_image=ref_clip_image,
        pose_image=pose_image,
        null_prompt=null_prompt,
        negative_prompt=negative_prompt,
        width=512,
        height=640,
        num_images_per_prompt=num_samples,
        guidance_scale=7.5,
        image_scale=1.0,
        generator=generator,
        num_inference_steps=50,
    ).images

    save_output = []
    save_output.append(output[0])
    save_output.insert(0, clothes_img.resize((512, 640), Image.BICUBIC))

    grid = image_grid(save_output, 1, 2)
    grid.save(
        output_path + '/' + args.cloth_path.split("/")[-1])
