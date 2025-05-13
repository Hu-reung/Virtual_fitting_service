import json
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from random import choice

# ---------- 가상 피팅 학습용 데이터셋 클래스 ----------
class VDDataset(Dataset):
    def __init__(
            self,
            json_file,         # 학습 데이터의 메타 정보를 담은 JSON 파일 경로 또는 리스트
            tokenizer,         # 텍스트를 토크나이징할 tokenizer 객체
            size=512,
            image_root_path="",
    ):

        # JSON 파일 불러오기 (단일 파일 또는 여러 개 가능)
        if isinstance(json_file, str):
            with open(json_file, 'r') as file:
                self.data = json.load(file)
        elif isinstance(json_file, list):
            for file_path in json_file:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    if not hasattr(self, 'data'):
                        self.data = data
                    else:
                        self.data.extend(data)
        else:
            raise ValueError("Input should be either a JSON file path (string) or a list")

        print('=========', len(self.data))

        self.tokenizer = tokenizer
        self.size = size
        self.image_root_path = image_root_path

        # 이미지 변환 정의 (512로 리사이즈 후 랜덤 크롭, 정규화)
        self.transform = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop([640, 512]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # CLIP 입력 이미지 전처리기
        self.clip_image_processor = CLIPImageProcessor()

    # 하나의 샘플을 가져오는 함수
    def __getitem__(self, idx):
        item = self.data[idx]

        # 사람 이미지 및 옷 이미지 불러오기
        person_path = item["image_file"]
        person_img = Image.open(person_path).convert("RGB")
        cloth_path = item["cloth_file"]
        clothes_img = Image.open(cloth_path).convert("RGB")

        # 텍스트 중 랜덤으로 하나 선택
        text = choice(item['text'])

        # 일부 확률로 텍스트 또는 이미지 임베딩을 제거 (dropout 효과)
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < 0.05:
            drop_image_embed = 1
        elif rand_num < 0.1:
            text = ""
        elif rand_num < 0.15:
            text = ""
            drop_image_embed = 1

        # 텍스트 토큰화
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        null_text_input_ids = self.tokenizer(
            "",
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        # 이미지 변환 적용
        vae_person = self.transform(person_img)
        vae_clothes = self.transform(clothes_img)

        # CLIP 이미지 입력 준비
        clip_image = self.clip_image_processor(images=clothes_img, return_tensors="pt").pixel_values

        return {
            "vae_person": vae_person,
            "vae_clothes": vae_clothes,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
            "text": text,
            "text_input_ids": text_input_ids,
            "null_text_input_ids": null_text_input_ids,
        }

    # 전체 데이터 길이 반환
    def __len__(self):
        return len(self.data)


# ---------- 배치 단위로 텐서 정리하는 collate 함수 ----------
def collate_fn(data):
    vae_person = torch.stack([example["vae_person"] for example in data]).to(memory_format=torch.contiguous_format).float()
    vae_clothes = torch.stack([example["vae_clothes"] for example in data]).to(memory_format=torch.contiguous_format).float()

    clip_image = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embed = [example["drop_image_embed"] for example in data]

    text = [example["text"] for example in data]
    input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    null_input_ids = torch.cat([example["null_text_input_ids"] for example in data], dim=0)

    return {
        "vae_person": vae_person,
        "vae_clothes": vae_clothes,
        "clip_image": clip_image,
        "drop_image_embed": drop_image_embed,
        "text": text,
        "input_ids": input_ids,
        "null_input_ids": null_input_ids,
    }
