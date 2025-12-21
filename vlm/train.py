"""Training Entry"""

# === DEBUGPY (rank0 전용 attach) ===========================
import os, socket
if os.getenv("DEBUGPY","0") == "1":
    import debugpy
    port = int(os.getenv("DEBUGPY_PORT","5678"))
    rank = int(os.getenv("RANK", os.getenv("SLURM_PROCID","0")))
    if rank == 0:
        try:
            debugpy.listen(("127.0.0.1", port))  # ← dev container 면 loopback이 가장 안전
            print(f"[debugpy] waiting on 127.0.0.1:{port} (rank={rank}) ...", flush=True)
            debugpy.wait_for_client()
            print("[debugpy] attached.", flush=True)
        except Exception as e:
            print(f"[debugpy] listen failed: {e}", flush=True)
# ===========================================================

from vlm.train.arguments import parse_train_args
from vlm.train.trainer_builder import build_model_trainer


def main():
    # parse args
    args = parse_train_args()
    
    # get model trainer
    trainer = build_model_trainer(args)

    # start training
    trainer.train()


if __name__ == '__main__':
    main()
