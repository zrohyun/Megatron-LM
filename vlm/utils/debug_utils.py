import torch
import os
from megatron.training.utils import unwrap_model


def save_model(model_obj, save_path="/tmp/fast_debug_model.pt"):
    print(f">>> Saving model weights directly to {save_path} ...")

    # 1. Megatron은 모델이 List로 되어 있는 경우가 많음 (PP=1이어도)
    if isinstance(model_obj, list):
        model_obj = model_obj[0]

    # 2. DDP, Float16Module 등 불필요한 껍데기 벗기기 (필수)
    # 이걸 안 하면 나중에 로드할 때 키 값(module.module...)이 안 맞아서 고생합니다.
    clean_model = unwrap_model(model_obj)

    # 3. CPU로 옮겨서 순수 가중치만 저장
    torch.save(clean_model.state_dict(), save_path)
    print(">>> Model Saved! (Clean State Dict)")


def load_model(model_obj, load_path="/tmp/fast_debug_model.pt"):
    print(f">>> Fast loading from {load_path} ...")

    # 1. 모델 리스트 처리
    if isinstance(model_obj, list):
        model_obj = model_obj[0]

    # 2. 껍데기 벗기기 (저장할 때랑 똑같이 벗겨야 매칭됨)
    from megatron.training.utils import unwrap_model

    clean_model = unwrap_model(model_obj)

    # 3. 로드 (CPU로 읽어서 메모리 스파이크 방지)
    state_dict = torch.load(load_path, map_location="cpu")

    # 4. 주입 (strict=False로 설정하여 약간의 키 불일치 허용 - 디버깅용으로 안전)
    missing, unexpected = clean_model.load_state_dict(state_dict, strict=False)

    if len(missing) > 0:
        print(f"[Info] Missing keys: {len(missing)} (It's okay for debugging)")

    print(">>> Model Weights Loaded Successfully!")


def get_latest_checkpoint_iteration(load_dir):
    """
    megatron의 latest_checkpointed_iteration.txt 파일을 읽어 가장 최신 체크포인트 경로와 이터레이션을 반환합니다.
    """
    tracker_filename = os.path.join(load_dir, "latest_checkpointed_iteration.txt")
    if os.path.isfile(tracker_filename):
        print(f"tracker file content: {tracker_filename}")
        with open(tracker_filename, "r") as f:
            try:
                iteration = int(f.read().strip().split()[0])
            except:
                iteration = 1
        checkpoint_path = os.path.join(load_dir, f"iter_{iteration:07d}")
    else:
        checkpoint_path = load_dir
        iteration = 1
    return checkpoint_path, iteration
