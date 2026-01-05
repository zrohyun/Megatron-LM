import shutil
from pathlib import Path

from setuptools import setup

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "vlm/bridge/WBLVLMoE-A1B-HF-Dummy"
PKG_DIR = Path(__file__).resolve().parent / "wbl_vlm"

shutil.copy(SRC_DIR / "configuration_wbl_vl_moe.py", PKG_DIR / "configuration_wbl_vl_moe.py")
shutil.copy(SRC_DIR / "modeling_wbl_vl_moe.py", PKG_DIR / "modeling_wbl_vl_moe.py")

setup(
    name="wbl_vlm",
    version="0.1.0",
    packages=["wbl_vlm"],
    entry_points={
        "vllm.general_plugins": [
            "wbl_vlm_model = wbl_vlm:register",
        ],
    },
)
