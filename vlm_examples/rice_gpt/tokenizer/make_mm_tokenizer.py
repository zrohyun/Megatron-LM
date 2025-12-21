import os
import shutil
import argparse
import json

from pathlib import Path
from typing import Dict, List, Tuple, Set

from transformers import AutoTokenizer, AutoProcessor


RENAME_MAP: Dict[str, str] = {
    # "<|reserved_2000|>": "<|im_start|>",
    # "<|reserved_2001|>": "<|im_end|>",
    "<|reserved_1997|>": "<gro>",
    "<|reserved_1998|>": "<ocr>",
    "<|reserved_1999|>": "<delim>",
    "<|reserved_2000|>": "<char>",
    "<|reserved_2001|>": "</char>",
    "<|reserved_2002|>": "<obj>",
    "<|reserved_2003|>": "</obj>",
    "<|reserved_2004|>": "<bbox>",
    "<|reserved_2005|>": "</bbox>",
    "<|reserved_2006|>": "<quad>",
    "<|reserved_2007|>": "</quad>",
    "<|reserved_2008|>": "<|vision_start|>",
    "<|reserved_2009|>": "<|vision_end|>",
    "<|reserved_2010|>": "<|vision_pad|>",
    "<|reserved_2011|>": "<|image_pad|>",
    "<|reserved_2012|>": "<|video_pad|>",
}

IS_SPECIAL_TOKEN = {
    # "<|im_start|>": True,
    # "<|im_end|>": True,
    "<gro>": True,
    "<ocr>": True,
    "<delim>": True,
    "<char>": True,
    "</char>": True,
    "<obj>": True,
    "</obj>": True,
    "<bbox>": True,
    "</bbox>": True,
    "<quad>": True,
    "</quad>": True,
    "<|vision_start|>": True,
    "<|vision_end|>": True,
    "<|vision_pad|>": True,
    "<|image_pad|>": True,
    "<|video_pad|>": True,
}


# HF가 “스페셜”로 인지할 목록(추가 id 생성 없이 인지만 함)
SPECIAL_TO_REGISTER: List[str] = sorted(set(RENAME_MAP.values()))


def copy_files(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    target_exts = {".json", ".jinja", ".model"}
            
    for file_path in src_dir.iterdir():
        if file_path.is_file() and file_path.suffix in target_exts:
            dest_file = dst_dir / file_path.name
            shutil.copy2(file_path, dest_file)
            print(f"Copied: {file_path} → {dest_file}")
            

def load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(p: Path, obj: dict):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    

def rename_in_vocab(vocab: Dict[str, int], rename_map: Dict[str, str]) -> Tuple[Dict[str, int], List[str], List[str]]:
    """vocab(dict: token->id)에서 키 문자열만 교체, id는 유지."""
    renamed = []
    skipped = []
    # 타겟 이름 중복 방지 체크
    targets = set(rename_map.values())
    clash = [t for t in targets if t in vocab and t not in rename_map]
    if clash:
        raise ValueError(f"Target tokens already exist in vocab (would clash): {clash}")

    new_vocab = dict(vocab)
    for old, new in rename_map.items():
        if old in new_vocab:
            idx = new_vocab[old]
            del new_vocab[old]
            new_vocab[new] = idx
            renamed.append(f"{old} -> {new} (id={idx})")
        else:
            skipped.append(old)
    return new_vocab, renamed, skipped


def rename_in_added_tokens(added_tokens: List[dict], rename_map: Dict[str, str]) -> Tuple[List[dict], List[str]]:
    """added_tokens 리스트 항목(content 문자열)만 변경, id/flags 보존."""
    renamed = []
    if not added_tokens:
        return [], renamed
    new_added = []
    for at in added_tokens:
        at = dict(at)
        content = at.get("content")
        if content in rename_map:
            new_name = rename_map[content]
            at["content"] = new_name
            at["special"] = IS_SPECIAL_TOKEN.get(new_name, False)
            renamed.append(f"{content} -> {new_name} (added_tokens, id={at.get('id')})")
        new_added.append(at)
    return new_added, renamed


def dedup_between_vocab_and_added(
    vocab: Dict[str, int], added_tokens: List[dict]
) -> Tuple[Dict[str, int], List[dict], List[str]]:
    """
    같은 문자열이 vocab/added_tokens 양쪽에 있으면 model.vocab 에서는 지움
    """
    report = []
    if not added_tokens:
        return vocab, added_tokens, report

    vocab_only = dict(vocab)
    origin_added = list(added_tokens)
    new_added = []
    for at in added_tokens:
        s = at.get("content")
        if s in vocab:
            report.append(f"DEDUP: remove from vocab: {s} (kept in added_tokens)")
            del vocab_only[s]
        new_added.append(at)

    return vocab_only, new_added, report
    
    
def modify_tokenizer_file(tokenizer_path: Path, save_path: Path) -> List[dict]:
    tokenizer_json = load_json(tokenizer_path)

    # safety checks
    if "model" not in tokenizer_json or "vocab" not in tokenizer_json["model"]:
        raise ValueError("tokenizer.json malformed: missing model.vocab")
    vocab = tokenizer_json["model"]["vocab"]  # str->id
    added_tokens = tokenizer_json.get("added_tokens", [])

    print(f"[INFO] Before: vocab_size={len(vocab)}, added_tokens={len(added_tokens)}")

    # 1) rename in vocab
    vocab, renamed_vocab, skipped_vocab = rename_in_vocab(vocab, RENAME_MAP)
    print(f"[RENAME][vocab] {len(renamed_vocab)} renamed")
    for line in renamed_vocab:
        print("  -", line)
    if skipped_vocab:
        print(f"[WARN] {len(skipped_vocab)} keys not found in vocab:")
        print("      ", ", ".join(skipped_vocab[:12]), "..." if len(skipped_vocab) > 12 else "")

    # 2) rename in added_tokens
    added_tokens, renamed_added = rename_in_added_tokens(added_tokens, RENAME_MAP)
    print(f"[RENAME][added] {len(renamed_added)} renamed")
    for line in renamed_added:
        print("  -", line)

    # # 3) deduplicate between vocab and added_tokens
    # vocab, added_tokens, dedup_report = dedup_between_vocab_and_added(vocab, added_tokens)
    # for line in dedup_report:
    #     print("[DEDUP]", line)

    # 4) write back tokenizer.json
    tokenizer_json["model"]["vocab"] = vocab
    tokenizer_json["added_tokens"] = added_tokens
    save_json(save_path, tokenizer_json)
    print(f"[INFO] After : vocab_size={len(vocab)}, added_tokens={len(added_tokens)}")
    print(f"[OK] tokenizer.json updated")
    
    return added_tokens


def modify_extra_tokenizer_files(tokenizer_dir: Path, added_tokens: List[Dict], save_dir: Path):
    tcfg = load_json(tokenizer_dir / "tokenizer_config.json")
    
    new_add_tokens = []
    for add in added_tokens:
        if add['content'] in SPECIAL_TO_REGISTER:
            new_add_tokens.append(add)
    
    additional_special_tokens = []
    for add in new_add_tokens:
        _id = add.pop("id")
        tcfg['added_tokens_decoder'][str(_id)] = add
        if add['special']:
            additional_special_tokens.append(add['content'])
    
    tcfg['additional_special_tokens'].extend(additional_special_tokens)
    
    save_json(save_dir / "tokenizer_config.json", tcfg)
    print(f"[OK] {save_dir / "tokenizer_config.json"} updated")
    
    smap = load_json(tokenizer_dir / "special_tokens_map.json")
    smap['additional_special_tokens'].extend(additional_special_tokens)
    save_json(save_dir / "special_tokens_map.json", smap)
    print(f"[OK] {save_dir / "special_tokens_map.json"} updated")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", type=str, default="./tokenizers/wbl_tokenizer_v1", 
                    help="Tokenizer folder (contains tokenizer.json, ...)")
    ap.add_argument("--save-dir", type=str, default="./tokenizers/wbl_tokenizer_v2_mm", 
                    help="")
    args = ap.parse_args()

    tokenizer_dir = Path(args.tokenizer_dir)
    
    tmp_dir = Path(args.save_dir).parent / "tmp_mm_tokenizer"
    copy_files(tokenizer_dir, tmp_dir)

    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"{tokenizer_path} not found")
    
    added_tokens = modify_tokenizer_file(tokenizer_path, tmp_dir / "tokenizer.json")
    
    modify_extra_tokenizer_files(tokenizer_dir, added_tokens, tmp_dir)
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
    
    tokenizer = AutoTokenizer.from_pretrained(
        tmp_dir,
        trust_remote_code=True
    )
    
    processor.image_processor.temporal_patch_size = 1
    processor.image_processor.max_pixels = 1600*1600
    processor.tokenizer = tokenizer
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(save_dir)
    
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
