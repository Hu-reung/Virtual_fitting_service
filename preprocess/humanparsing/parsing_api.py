import pdb
from pathlib import Path
import sys

# 현재 파일의 부모 디렉토리를 프로젝트 루트로 설정하고 경로 등록
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import os
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets.simple_extractor_dataset import SimpleFolderDataset
from utils.transforms import transform_logits
from tqdm import tqdm
from PIL import Image

# ---------- 시각화를 위한 팔레트(색상 맵) 생성 함수 ----------
def get_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

# ---------- 불필요한 부위를 제거하고, 옷/드레스 타입 판단 ----------
def delete_irregular(logits_result):
    parsing_result = np.argmax(logits_result, axis=2)
    upper_cloth = np.where(parsing_result == 4, 255, 0)  # 상의 영역
    contours, hierarchy = cv2.findContours(upper_cloth.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = [abs(cv2.contourArea(c, True)) for c in contours]
    if len(area) != 0:
        top = area.index(max(area))
        M = cv2.moments(contours[top])
        cY = int(M["m01"] / M["m00"])

    dresses = np.where(parsing_result == 7, 255, 0)
    contours_dress, _ = cv2.findContours(dresses.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area_dress = [abs(cv2.contourArea(c, True)) for c in contours_dress]
    if len(area_dress) != 0:
        top_dress = area_dress.index(max(area_dress))
        M_dress = cv2.moments(contours_dress[top_dress])
        cY_dress = int(M_dress["m01"] / M_dress["m00"])

    wear_type = "dresses"
    if len(area) != 0:
        if len(area_dress) != 0 and cY_dress > cY:
            logits_result[:, :, [4, 5, 6]] = -1
        else:
            logits_result[:cY, :, [5, 6, 7, 8, 9, 10, 12, 13]] = -1
            wear_type = "cloth_pant"
        parsing_result = np.argmax(logits_result, axis=2)
    parsing_result = np.pad(parsing_result, pad_width=1, mode='constant', constant_values=0)
    return parsing_result, wear_type

# ---------- 이미지에 구멍(hole)이 있을 때 채우기 ----------
def hole_fill(img):
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst

# ---------- 잡영역 제거하고 마스크 정제 ----------
def refine_mask(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = [abs(cv2.contourArea(c, True)) for c in contours]
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)
        for j in range(len(area)):
            if j != i and area[i] > 2000:
                cv2.drawContours(refine_mask, contours, j, color=255, thickness=-1)
    return refine_mask

# ---------- 옷과 팔 사이 구멍을 정제하는 함수 ----------
def refine_hole(parsing_result_filled, parsing_result, arm_mask):
    filled_hole = cv2.bitwise_and(np.where(parsing_result_filled == 4, 255, 0),
                                  np.where(parsing_result != 4, 255, 0)) - arm_mask * 255
    contours, _ = cv2.findContours(filled_hole, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    refine_hole_mask = np.zeros_like(parsing_result).astype(np.uint8)
    for c in contours:
        a = cv2.contourArea(c, True)
        if abs(a) > 2000:
            cv2.drawContours(refine_hole_mask, [c], -1, color=255, thickness=-1)
    return refine_hole_mask + arm_mask

# ---------- ONNX 모델을 이용한 이미지 파싱 메인 함수 ----------
def onnx_inference(session, lip_session, input_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])

    # 첫 번째 파싱 모델 처리 (512 해상도)
    dataset = SimpleFolderDataset(root=input_dir, input_size=[512, 512], transform=transform)
    dataloader = DataLoader(dataset)
    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader)):
            image, meta = batch
            c, s, w, h = meta['center'].numpy()[0], meta['scale'].numpy()[0], meta['width'].numpy()[0], meta['height'].numpy()[0]
            output = session.run(None, {"input.1": image.numpy().astype(np.float32)})
            upsample = torch.nn.Upsample(size=[512, 512], mode='bilinear', align_corners=True)
            upsample_output = upsample(torch.from_numpy(output[1][0]).unsqueeze(0)).squeeze().permute(1, 2, 0)
            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=[512, 512])
            parsing_result = np.argmax(logits_result, axis=2)
            parsing_result = np.pad(parsing_result, pad_width=1, mode='constant', constant_values=0)

            # 옷 + 팔 부분 마스크 생성
            arm_mask = (parsing_result == 14).astype(np.float32) + (parsing_result == 15).astype(np.float32)
            upper_cloth_mask = (parsing_result == 4).astype(np.float32) + arm_mask
            dst = hole_fill(np.where(upper_cloth_mask, 255, 0).astype(np.uint8))
            parsing_result_filled = dst / 255 * 4
            parsing_result_woarm = np.where(parsing_result_filled == 4, parsing_result_filled, parsing_result)

            # 팔+옷 사이 구멍 정제
            refine_hole_mask = refine_hole(parsing_result_filled.astype(np.uint8), parsing_result.astype(np.uint8), arm_mask.astype(np.uint8))
            parsing_result = np.where(refine_hole_mask, parsing_result, parsing_result_woarm)
            parsing_result = parsing_result[1:-1, 1:-1]  # 패딩 제거

    # 두 번째 파싱 모델 처리 (473 해상도, 얼굴 마스크용)
    dataset_lip = SimpleFolderDataset(root=input_dir, input_size=[473, 473], transform=transform)
    dataloader_lip = DataLoader(dataset_lip)
    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader_lip)):
            image, meta = batch
            c, s, w, h = meta['center'].numpy()[0], meta['scale'].numpy()[0], meta['width'].numpy()[0], meta['height'].numpy()[0]
            output_lip = lip_session.run(None, {"input.1": image.numpy().astype(np.float32)})
            upsample = torch.nn.Upsample(size=[473, 473], mode='bilinear', align_corners=True)
            upsample_output_lip = upsample(torch.from_numpy(output_lip[1][0]).unsqueeze(0)).squeeze().permute(1, 2, 0)
            logits_result_lip = transform_logits(upsample_output_lip.data.cpu().numpy(), c, s, w, h, input_size=[473, 473])
            parsing_result_lip = np.argmax(logits_result_lip, axis=2)

    # 얼굴과 목 마스크 합치기
    neck_mask = np.logical_and(~(parsing_result_lip == 13).astype(np.float32), (parsing_result == 11).astype(np.float32))
    parsing_result = np.where(neck_mask, 18, parsing_result)

    palette = get_palette(19)
    output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
    output_img.putpalette(palette)
    face_mask = torch.from_numpy((parsing_result == 11).astype(np.float32))

    return output_img, face_mask
