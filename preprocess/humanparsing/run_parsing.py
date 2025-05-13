import pdb
from pathlib import Path
import sys
import os
import onnxruntime as ort
from PIL import Image

# 현재 파일의 경로 기준으로 프로젝트 루트 설정 후 import 경로 추가
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from parsing_api import onnx_inference
import torch


# ---------- Parsing 클래스 정의: ONNX 모델로 파싱 수행 ----------
class Parsing:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)  # 사용할 GPU 설정

        # ONNX 실행 옵션 설정
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.add_session_config_entry('gpu_id', str(gpu_id))

        # 첫 번째 모델 (parsing_atr.onnx) 로드
        self.session = ort.InferenceSession(
            os.path.join(Path(__file__).absolute().parents[2].absolute(), 'ckpt/humanparsing/parsing_atr.onnx'),
            sess_options=session_options, providers=['CPUExecutionProvider'])

        # 두 번째 모델 (parsing_lip.onnx) 로드
        self.lip_session = ort.InferenceSession(
            os.path.join(Path(__file__).absolute().parents[2].absolute(), 'ckpt/humanparsing/parsing_lip.onnx'),
            sess_options=session_options, providers=['CPUExecutionProvider'])

    def __call__(self, input_image):
        # Parsing 객체를 함수처럼 호출할 때 실행됨
        # input_image를 파싱하여 세그멘테이션 결과와 얼굴 마스크 반환
        parsed_image, face_mask = onnx_inference(self.session, self.lip_session, input_image)
        return parsed_image, face_mask
