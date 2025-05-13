import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern
import os
import argparse

# eval.py : CAMI-U 방식으로 옷만 평가하는 코드
# --- 사용된 라이브러리 ---
"""
cv2: 이미지 입출력 및 처리

ssim: 이미지 유사도 비교용

local_binary_pattern: 텍스처 분석용

argparse: 터미널에서 인자 받을 때 씀

"""

# -- 해리스 코너로 이미지 불러와서 키포인트와 시각화된 이미지 반환 --
def extract_clothing_keypoints(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04) #해리스 코너 추출 

    dst = cv2.dilate(dst, None)

    img[dst > 0.01 * dst.max()] = [0, 0, 255] #코너에 빨간 점 찍기

    keypoints = np.argwhere(dst > 0.01 * dst.max()) #코너 좌표 모음
    keypoints = [tuple(point) for point in keypoints]

    return keypoints, img

# -- SSIM 점수 계산 (이미지 유사도)--
def calculate_ssim(img1, img2):
    return ssim(img1, img2, multichannel=True, channel_axis=2) 

# -- 옷의 경계점(코너)이 얼마나 비슷한 위치에 있는지를 비교하는 함수 / 값이 작을수록 잘 맞음 --
def calculate_keypoint_matching(keypoints1, keypoints2):
    keypoints1 = np.array(keypoints1)
    keypoints2 = np.array(keypoints2)
    if len(keypoints2) == 0 or len(keypoints2) > 5000:
        return 0.99 #예외처리 : 너무 적거나 많으면 일단 고정된 점수 반환
    else:

        distances_array1 = np.linalg.norm(keypoints1[:, np.newaxis, :] - keypoints2[np.newaxis, :, :], axis=2)
        min_distances_array1 = np.min(distances_array1, axis=1)

        return np.mean(min_distances_array1) / (512. * np.sqrt(2)) # 평균 거리 계산해서 정규화


# -- 옷의 질감, 무늬 비슷한지 확인하는 함수 (텍스쳐 유사도 계산_LBP기반) --
def calculate_texture_similarity(img1, img2, P=8, R=1.0):
    lbp1 = local_binary_pattern(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), P, R)
    lbp2 = local_binary_pattern(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), P, R)
    hist1, _ = np.histogram(lbp1, bins=np.arange(0, P ** 2 + 1), density=True)
    hist2, _ = np.histogram(lbp2, bins=np.arange(0, P ** 2 + 1), density=True)
    hist2 = hist2.astype(np.float32)
    hist1 = hist1.astype(np.float32)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL) # 텍스처 히스토그램 유사도


# -- 최종 점수 합산 공식 --
def calculate_cami_us(img1, img2, keypoints1, keypoints2):  
    ssim_score = calculate_ssim(img1, img2) # ssim : 전체적인 형태/색감
    keypoint_matching = calculate_keypoint_matching(keypoints1, keypoints2)

    texture_similarity = calculate_texture_similarity(img1, img2) # 질감 유사도, -> 세 개 합쳐서 CAMI-U 점수로 반환 

    cami_us_score = ssim_score + (1 - keypoint_matching) + texture_similarity # 코너가 잘 맞을수록 높음

    return cami_us_score

# -- 터미널로 경로 받아오기 --
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute the metrics for the generated images")
    parser.add_argument('--cloth_path', type=str, help='input cloth path')
    parser.add_argument('--cloth_mask_path', type=str, help='generate cloth mask path')
    args = parser.parse_args()
    # -- 실제 옷 이미지 vs 생성된 옷 이미지 쌍마다 1. 크기 2. 경계점 추출 3. 유사도 점수 계산 )-> 평균값 출력
    cloth_paths = os.listdir(args.cloth_path)
    score_list = []
    for cloth in cloth_paths:
        cloth = os.path.join(args.cloth_path, cloth)
        generate_cloth = os.path.join(args.cloth_mask_path, cloth)
        if not os.path.exists(generate_cloth):
            score_list.append(0)
            continue
        reference_cloth_img = cv2.imread(cloth)
        generated_cloth_img = cv2.imread(generate_cloth)

        h, w = generated_cloth_img.shape[0], generated_cloth_img.shape[1]
        reference_cloth_img = cv2.resize(reference_cloth_img, (w, h))

        reference_keypoints, _ = extract_clothing_keypoints(cloth)
        generated_keypoints, _ = extract_clothing_keypoints(generated_cloth_img)

        cami_us_score = calculate_cami_us(reference_cloth_img, generated_cloth_img, reference_keypoints,
                                          generated_keypoints)
        score_list.append(cami_us_score)

    print('cami_us_score:', np.mean(score_list))
