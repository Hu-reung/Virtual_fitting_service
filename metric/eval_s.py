# ------------ 기본 라이브러리 import -------------
import cv2
import numpy
import numpy as np
import torch
import clip
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern
import insightface
from insightface.app import FaceAnalysis
import os
from pathlib import Path
import sys
import argparse

# ------------ 루트 경로 등록 (내부 모듈 import 위해) -------------
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from preprocess.openpose.run_openpose import OpenPose


# ------------ 옷 이미지에서 코너(윤곽선) 키포인트 추출 -------------
def extract_clothing_keypoints(image_path):
    img = cv2.imread(image_path)  # 이미지 읽기
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 흑백 변환
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)  # Harris 코너 감지

    dst = cv2.dilate(dst, None)  # 팽창 처리해서 코너 강조

    img[dst > 0.01 * dst.max()] = [0, 0, 255]  # 코너 부분 빨간색으로 표시

    keypoints = np.argwhere(dst > 0.01 * dst.max())  # 코너 위치 추출
    keypoints = [tuple(point) for point in keypoints]

    return keypoints, img


# ------------ SSIM 이미지 유사도 계산 -------------
def calculate_ssim(img1, img2):
    return ssim(img1, img2, multichannel=True, channel_axis=2)


# ------------ 코너 포인트 유사도 계산 (거리가 가까울수록 좋음) -------------
def calculate_keypoint_matching(keypoints1, keypoints2):
    keypoints1 = np.array(keypoints1)
    keypoints2 = np.array(keypoints2)
    if len(keypoints2) == 0 or len(keypoints2) > 5000:
        return 0.99  # 예외 처리: 너무 적거나 너무 많으면 0.99로 고정
    else:
        # 각 점 사이 거리 계산 → 최소 거리 평균값 사용
        distances_array1 = np.linalg.norm(keypoints1[:, np.newaxis, :] - keypoints2[np.newaxis, :, :], axis=2)
        min_distances_array1 = np.min(distances_array1, axis=1)
        return np.mean(min_distances_array1) / (512. * np.sqrt(2))  # 정규화된 평균 거리


# ------------ LBP 기반 텍스처(질감) 유사도 계산 -------------
def calculate_texture_similarity(img1, img2, P=8, R=1.0):
    lbp1 = local_binary_pattern(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), P, R)
    lbp2 = local_binary_pattern(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), P, R)
    hist1, _ = np.histogram(lbp1, bins=np.arange(0, P ** 2 + 1), density=True)
    hist2, _ = np.histogram(lbp2, bins=np.arange(0, P ** 2 + 1), density=True)
    hist2 = hist2.astype(np.float32)
    hist1 = hist1.astype(np.float32)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)  # 두 히스토그램의 상관 관계 계산


# ------------ CLIP으로 텍스트-이미지 의미 유사도 계산 -------------
def calculate_clip_similarity(text, image):
    text_inputs = clip.tokenize([text]).to(device)
    image_inputs = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        image_features = model.encode_image(image_inputs)

    text_features /= text_features.norm(dim=-1, keepdim=True)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    similarity = (text_features @ image_features.T).item()  # 코사인 유사도 계산
    return similarity


# ------------ OpenPose 기반 포즈 이미지 유사도 계산 -------------
def calculate_openpose_distance(pose_image1, pose_image2):
    pose1 = numpy.array(pose_image1)
    pose_map = pose1[:, :, 0] + pose1[:, :, 1] + pose1[:, :, 2]
    num = np.count_nonzero(pose_map) * 1.
    pose2 = np.array(pose_image2)
    pose_map2 = pose2[:, :, 0] + pose2[:, :, 1] + pose2[:, :, 2]
    num_overlap = np.count_nonzero(pose_map * pose_map2) * 1.
    return num_overlap / num  # 겹치는 영역 비율로 유사도 계산


# ------------ 얼굴 임베딩 기반 유사도 계산 -------------
def calculate_face_similarity(face1, face2):
    face1 = cv2.imread(face1)
    face2 = cv2.imread(face2)
    face1_embed = app.get(face1)[0].normed_embedding
    face2_embed = app.get(face2)[0].normed_embedding
    similarity = np.dot(face1_embed, face2_embed.T)  # 코사인 유사도
    return similarity


# ------------ CAMI-U 점수 계산 함수 (옷 이미지 품질 평가) -------------
def calculate_cami_us(img1, img2, keypoints1, keypoints2):
    ssim_score = calculate_ssim(img1, img2)
    keypoint_matching = calculate_keypoint_matching(keypoints1, keypoints2)
    texture_similarity = calculate_texture_similarity(img1, img2)
    cami_us_score = ssim_score + (1 - keypoint_matching) + texture_similarity
    return cami_us_score


# ------------ CAMI-S 점수 계산 함수 (사람 전체 결과 평가) -------------
def calculate_cami_s(pose1, pose2, face1, face2, text, img):
    openpose_score = calculate_openpose_distance(pose1, pose2)
    face_score = calculate_face_similarity(face1, face2)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    clip_similarity = calculate_clip_similarity(text, img_pil)
    cami_s_score = (face_score + clip_similarity + openpose_score)
    return cami_s_score


# ------------ 메인 실행부 -------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute the metrics for the generated images")
    parser.add_argument('--cloth_path', type=str, help='input cloth path')
    parser.add_argument('--cloth_mask_path', type=str, help='generate cloth mask path')
    parser.add_argument('--model_path', type=str, help='generate model path')
    parser.add_argument('--pose_path', type=str, help='input pose path')
    parser.add_argument('--face_path', type=str, help='input face path')
    parser.add_argument('--text_prompts', type=list, help='input text prompts list')
    args = parser.parse_args()

    # 디바이스 설정 및 모델 로드
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model_pose = OpenPose(5)
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # 데이터 불러오기
    cloth_paths = os.listdir(args.cloth_path)
    score_list = []
    prompts = args.text_prompts
    faces_list = os.listdir(args.face_path)
    poses_list = os.listdir(args.pose_path)

    # ------------ 모든 조합 반복 평가 -------------
    for i_prompt in range(len(prompts)):
        prompt = prompts[i_prompt]

        for j_pose in range(len(poses_list)):
            pose_path = poses_list[j_pose]
            pose_path = os.path.join(args.pose_path, pose_path)
            pose1 = cv2.imread(pose_path)

            for k_face in range(5):  # 얼굴 5개까지만 사용
                face_path = faces_list[k_face]
                face_path1 = os.path.join(args.face_path, face_path)

                for cloth in cloth_paths:
                    cloth = os.path.join(args.cloth_path, cloth)

                    # 생성된 파일명은 규칙 기반으로 구성되어 있음
                    generate = f'pt_{i_prompt}_pose_{j_pose}_face_{k_face}_{cloth.split("/")[-1]}'
                    generate_cloth_path = os.path.join(args.cloth_mask_path, generate)
                    reference_cloth_img = cv2.imread(cloth)
                    generate_cloth_img = cv2.imread(generate_cloth_path)

                    # 생성된 전체 이미지 경로
                    img_path = os.path.join(args.model_path, generate)
                    img = cv2.imread(img_path)

                    # 사이즈 맞춤
                    h, w = generate_cloth_img.shape[0], generate_cloth_img.shape[1]
                    reference_cloth_img = cv2.resize(reference_cloth_img, (w, h))

                    # 포즈 추출
                    _, pose_image2 = model_pose(img_path)
                    pose2 = cv2.resize(pose_image2, (pose1.shape[1], pose1.shape[0]))

                    # 키포인트 추출
                    reference_keypoints, _ = extract_clothing_keypoints(cloth)
                    generated_keypoints, _ = extract_clothing_keypoints(generate_cloth_path)

                    # CAMI-U 계산
                    cami_us_score = calculate_cami_us(reference_cloth_img, generate_cloth_img, reference_keypoints, generated_keypoints)

                    # CAMI-S 계산 (CAMI-U 포함)
                    cami_s_score = cami_us_score + calculate_cami_s(pose1, pose2, face_path1, img_path, prompt, img)

                    score_list.append(cami_s_score)

    # ------------ 최종 점수 평균 출력 -------------
    print("cami_s_score: ", np.mean(score_list))
