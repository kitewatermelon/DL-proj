import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["plain", "cleaned"], required=True, help="데이터 전처리 버전 선택")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA 디바이스 번호 (예: 0 또는 1)")
    return parser.parse_args()

def get_device(cuda_idx):
    return f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu"
