from dressing_sd.pipelines.IMAGDressing_v1_pipeline import IMAGDressing_v1
import os
import torch

from PIL import Image
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from torchvision import transforms
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from adapter.attention_processor import CacheAttnProcessor2_0, RefSAttnProcessor2_0, CAttnProcessor2_0
import argparse
from adapter.resampler import Resampler

# 전체 흐름 : 의상 이미지 입력 받은 후 "A beautiful woman" 텍스트 프롬프트에 따라
# 새로운 이미지를 생성하는 가상 의상 입히기 파이프라인 실행행

# transforms.RandomCrop([1024, 768]),
# 입력 이미지를 지정된 최소/최대 크기와 base pixel size에 맞게 리사이즈
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

# 여러 이미지를 그리드 형태로 병합해 1장의 이미지
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# 전체 파이프라인 객체와 random seed generator 초기화화
def prepare(args):
    generator = torch.Generator(device=args.device).manual_seed(42)
    vae = AutoencoderKL.from_pretrained("stablediffusionapi/counterfeit-v30").to(dtype=torch.float16, device=args.device)
    tokenizer = CLIPTokenizer.from_pretrained("stablediffusionapi/counterfeit-v30", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("stablediffusionapi/counterfeit-v30", subfolder="text_encoder").to(
        dtype=torch.float16, device=args.device)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter",
                                                                  subfolder="models/image_encoder").to(
        dtype=torch.float16, device=args.device)
    unet = UNet2DConditionModel.from_pretrained("stablediffusionapi/counterfeit-v30", subfolder="unet").to(
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

    ref_unet = UNet2DConditionModel.from_pretrained("stablediffusionapi/counterfeit-v30", subfolder="unet").to(
        dtype=torch.float16,
        device=args.device)
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
        elif k.startswith("adapter_modules"):
            adapter_modules_dict[k.replace("adapter_modules.", "")] = model_sd[k]
        else:
            print(k)

    ref_unet.load_state_dict(ref_unet_dict)
    image_proj.load_state_dict(image_proj_dict)
    adapter_modules.load_state_dict(adapter_modules_dict)

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    # 가상 피팅을 수행하는 Diffusion 기반 이미지 생성
    pipe = IMAGDressing_v1(unet=unet, reference_unet=ref_unet, vae=vae, tokenizer=tokenizer,
                           text_encoder=text_encoder, image_encoder=image_encoder,
                           ImgProj=image_proj,
                           scheduler=noise_scheduler,
                           safety_checker=StableDiffusionSafetyChecker,
                           feature_extractor=CLIPImageProcessor)
    return pipe, generator

# 모델 체크포인트, 입력 의상 이미지 경로등, 출력 디렉토리 생성, 파이프라인(여러 모델과 전처리/후처리 과정을 하나로 묶음) 및 generator 초기화
# 텍스트 프롬프트 구성, 입력 의상 이미지 로드 및 전처리, 파이프라인 실행, 결과 이미지와 입력 이미지 나란히 병합해 저장

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IMAGDressing_v1')
    parser.add_argument('--model_ckpt',
                        default="ckpt/IMAGDressing-v1_512.pt",
                        type=str)
    parser.add_argument('--cloth_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default="./output_sd_base")
    parser.add_argument('--device', type=str, default="cuda:0")
    args = parser.parse_args()

    # svae path
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

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
    prompt = prompt + ', confident smile expression, best quality, high quality'
    null_prompt = ''
    negative_prompt = 'bare, naked, nude, undressed, monochrome, lowres, bad anatomy, worst quality, low quality'

    clothes_img = Image.open(args.cloth_path).convert("RGB")
    clothes_img = resize_img(clothes_img)
    vae_clothes = img_transform(clothes_img).unsqueeze(0)
    ref_clip_image = clip_image_processor(images=clothes_img, return_tensors="pt").pixel_values

    output = pipe(
        ref_image=vae_clothes,
        prompt=prompt,
        ref_clip_image=ref_clip_image,
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