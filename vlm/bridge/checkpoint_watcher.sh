#!/bin/bash
# vlm/bridge/checkpoint_watcher.sh
# 새로 생성된 체크포인트를 감지하여 HF 포맷으로 변환
set -e

# ========== 환경변수 기본값 ==========
MCORE_BASE_PATH="${MCORE_BASE_PATH:-/mnt/checkpoint/ncai/multimodal/outputs}"
CUTOFF_TIMESTAMP="${CUTOFF_TIMESTAMP:-0}"  # 이 시점 이후에 생성된 것만 변환
STAGES="${STAGES:-vlm_stage_3 vlm_stage_4}" # vlm_stage_1 vlm_stage_2 
DRY_RUN="${DRY_RUN:-false}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_ROOT="${MEGATRON_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# 로그 디렉토리
LOG_DIR="${LOG_DIR:-${MEGATRON_ROOT}/.log}"
mkdir -p "$LOG_DIR" 2>/dev/null || true

# Webhook 설정
WEBHOOK_URL="${WEBHOOK_URL:-https://ncsoftcorp.webhook.office.com/webhookb2/a39ad570-5ba2-4834-82bf-abce9618c6b7@91856527-a446-4990-b48e-37ca10f2ee8d/IncomingWebhook/dd35d751c40c465dae090155f135a6f6/974d28ef-0c77-4fd6-898a-d55a951ff068/V2SKEbUIpZPQfGIGNCSvflcHKkTaiZ7Dl2rk2gD7P_aUI1}"
SEND_WEBHOOK="${SEND_WEBHOOK:-true}"

# HF 더미 모델 경로
HF_DUMMY_MODEL="${HF_DUMMY_MODEL:-${SCRIPT_DIR}/WBLVLMoE-A1B-HF-Dummy}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Webhook 알림 전송
send_webhook() {
    local train_name="$1"
    local iteration="$2"
    local input_path="$3"
    local output_path="$4"
    local status="$5"  # success or failed
    
    if [[ "$SEND_WEBHOOK" != "true" ]]; then
        return 0
    fi
    
    local emoji="✅"
    local status_text="변환 완료"
    if [[ "$status" == "failed" ]]; then
        emoji="❌"
        status_text="변환 실패"
    fi
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    local payload=$(cat <<EOF
{
  "text": "${emoji} **Checkpoint 변환 ${status_text}**\n\n- **학습명**: ${train_name}\n- **Iteration**: ${iteration}\n- **입력 경로**: ${input_path}\n- **출력 경로**: ${output_path}\n- **시간**: ${timestamp}"
}
EOF
)
    
    curl -sS -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$WEBHOOK_URL" || log "WARNING: Failed to send webhook notification"
}

# 단일 체크포인트 변환 함수
convert_checkpoint() {
    local input_path="$1"
    local output_path="${input_path}-HF"
    
    # 메타데이터 추출
    local iteration=$(basename "$input_path")
    local train_name=$(basename "$(dirname "$input_path")")
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY-RUN] Would convert: $input_path → $output_path"
        log "[DRY-RUN] Train: $train_name, Iteration: $iteration"
        return 0
    fi
    
    log "Converting: $input_path → $output_path"
    log "Train: $train_name, Iteration: $iteration"
    
    # 환경변수 설정
    export MCORE_PATH="$input_path"
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    export RANK=0
    export LOCAL_RANK=0
    export WORLD_SIZE=1
    export MASTER_ADDR=localhost
    export MASTER_PORT=12355
    
    cd "$MEGATRON_ROOT"
    
    # 변환 실행 (CPU 모드)
    if python -m torch.distributed.run \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=1 \
        vlm/bridge/export_megatron_to_hf.py \
        --hf-model "$HF_DUMMY_MODEL" \
        --megatron-path "$input_path" \
        --hf-path "$output_path"; then
        
        log "Successfully converted: $output_path"
        
        # 변환된 모델 테스트 실행
        run_model_test "$output_path" "$train_name" "$iteration" "$input_path"
        return 0
    else
        log "ERROR: Failed to convert $input_path"
        send_webhook "$train_name" "$iteration" "$input_path" "$output_path" "failed"
        return 1
    fi
}

# 모델 테스트 함수
run_model_test() {
    local output_path="$1"
    local train_name="$2"
    local iteration="$3"
    local input_path="$4"
    
    log "Testing converted model: $output_path"
    cd "$MEGATRON_ROOT"
    
    # 테스트 이미지 경로
    local test_image="${TEST_IMAGE:-test_images/mario.jpg}"
    
    # 테스트 출력 캡처
    local test_output
    local test_exit_code
    test_output=$(MODEL_PATH="$output_path" TEST_IMAGE="$test_image" python vlm/bridge/test_model_generation.py 2>&1)
    test_exit_code=$?
    
    # 출력 로그에 기록
    echo "$test_output"
    
    # MODEL_OUTPUT 마커 사이의 내용 추출
    local model_response
    model_response=$(echo "$test_output" | sed -n '/===MODEL_OUTPUT_START===/,/===MODEL_OUTPUT_END===/p' | grep -v '===MODEL_OUTPUT' | head -5)
    
    if [[ $test_exit_code -eq 0 ]]; then
        log "✅ Model test passed: $output_path"
        send_webhook_with_output "$train_name" "$iteration" "$input_path" "$output_path" "success" "$model_response" "$test_image"
    else
        log "⚠️ Model test failed: $output_path (conversion was successful)"
        send_webhook_with_output "$train_name" "$iteration" "$input_path" "$output_path" "test_failed" "$model_response" "$test_image"
    fi
}

# Webhook 알림 전송 (테스트 출력 포함)
send_webhook_with_output() {
    local train_name="$1"
    local iteration="$2"
    local input_path="$3"
    local output_path="$4"
    local status="$5"
    local model_output="$6"
    local test_image="$7"
    
    if [[ "$SEND_WEBHOOK" != "true" ]]; then
        return 0
    fi
    
    local emoji="✅"
    local status_text="변환 및 테스트 완료"
    if [[ "$status" == "test_failed" ]]; then
        emoji="⚠️"
        status_text="변환 완료 (테스트 실패)"
    elif [[ "$status" == "failed" ]]; then
        emoji="❌"
        status_text="변환 실패"
    fi
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 모델 출력 이스케이프 (JSON 안전하게)
    local escaped_output
    escaped_output=$(echo "$model_output" | sed 's/\\/\\\\/g; s/"/\\"/g; s/\n/\\n/g' | tr '\n' ' ' | head -c 500)
    
    local payload=$(cat <<EOF
{
  "text": "${emoji} **Checkpoint ${status_text}**\n\n- **학습명**: ${train_name}\n- **Iteration**: ${iteration}\n- **입력 경로**: ${input_path}\n- **출력 경로**: ${output_path}\n- **시간**: ${timestamp}\n\n**🤖 모델 출력:**\n${escaped_output}\n\n**🖼️ 테스트 이미지:** ${test_image}"
}
EOF
)
    
    curl -sS -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$WEBHOOK_URL" || log "WARNING: Failed to send webhook notification"
}

# 폴더 생성시간 가져오기 (크로스 플랫폼)
get_mtime() {
    local path="$1"
    if stat --version &>/dev/null 2>&1; then
        # GNU stat (Linux)
        stat -c %Y "$path"
    else
        # BSD stat (macOS)
        stat -f %m "$path"
    fi
}

# 메인 로직: 자동 감지 및 변환
main() {
    log "=========================================="
    log "Starting checkpoint watcher..."
    log "MCORE_BASE_PATH: $MCORE_BASE_PATH"
    log "CUTOFF_TIMESTAMP: $CUTOFF_TIMESTAMP ($(date -d "@$CUTOFF_TIMESTAMP" 2>/dev/null || date -r "$CUTOFF_TIMESTAMP" 2>/dev/null || echo 'N/A'))"
    log "STAGES: $STAGES"
    log "DRY_RUN: $DRY_RUN"
    log "=========================================="
    
    local converted_count=0
    local skipped_count=0
    local failed_count=0
    
    for stage in $STAGES; do
        local stage_path="${MCORE_BASE_PATH}/${stage}/checkpoints"
        
        if [[ ! -d "$stage_path" ]]; then
            log "Stage path not found, skipping: $stage_path"
            continue
        fi
        
        log "Scanning: $stage_path"
        
        # 각 학습 디렉토리 순회
        for train_dir in "$stage_path"/*/; do
            [[ ! -d "$train_dir" ]] && continue
            
            local train_name=$(basename "$train_dir")
            
            # iter_* 디렉토리 찾기 (-HF로 끝나지 않는 것만)
            for ckpt_dir in "$train_dir"/iter_*; do
                [[ ! -d "$ckpt_dir" ]] && continue
                [[ "$ckpt_dir" == *-HF ]] && continue  # -HF로 끝나는 건 스킵
                
                local ckpt_name=$(basename "$ckpt_dir")
                local hf_dir="${ckpt_dir}-HF"
                
                # 1. 이미 변환된 경우 스킵
                if [[ -d "$hf_dir" ]]; then
                    log "Skipping (already converted): $ckpt_dir"
                    ((skipped_count++)) || true
                    continue
                fi
                
                # 2. 폴더 생성 시간 확인
                local dir_mtime=$(get_mtime "$ckpt_dir")
                
                if [[ $dir_mtime -lt $CUTOFF_TIMESTAMP ]]; then
                    log "Skipping (older than cutoff): $ckpt_dir (mtime: $dir_mtime)"
                    ((skipped_count++)) || true
                    continue
                fi
                
                # 3. 변환 실행
                if convert_checkpoint "$ckpt_dir"; then
                    ((converted_count++)) || true
                else
                    ((failed_count++)) || true
                fi
            done
        done
    done
    
    log "=========================================="
    log "Completed. Converted: $converted_count, Skipped: $skipped_count, Failed: $failed_count"
    log "=========================================="
}

# 수동 모드: 특정 경로 직접 지정
manual_convert() {
    local input_path="$1"
    
    if [[ -z "$input_path" ]]; then
        echo "Usage: $0 --manual <checkpoint_path>"
        echo ""
        echo "Example:"
        echo "  $0 --manual /mnt/checkpoint/ncai/multimodal/outputs/vlm_stage_1/checkpoints/my_train/iter_0001000"
        exit 1
    fi
    
    if [[ ! -d "$input_path" ]]; then
        log "ERROR: Checkpoint path not found: $input_path"
        exit 1
    fi
    
    local output_path="${input_path}-HF"
    local iteration=$(basename "$input_path")
    local train_name=$(basename "$(dirname "$input_path")")
    
    # -HF 폴더가 이미 있으면 변환 스킵, 테스트만 실행
    if [[ -d "$output_path" ]]; then
        log "HF model already exists: $output_path"
        log "Skipping conversion, running test only..."
        run_model_test "$output_path" "$train_name" "$iteration" "$input_path"
    else
        log "Converting: $input_path → $output_path"
        convert_checkpoint "$input_path"
    fi
}

# 인자 파싱
case "${1:-}" in
    --manual|-m)
        manual_convert "$2"
        ;;
    --dry-run)
        DRY_RUN=true
        main
        ;;
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  (none)        Run automatic checkpoint detection and conversion"
        echo "  --manual PATH Convert a specific checkpoint"
        echo "  --dry-run     Show what would be converted without actually converting"
        echo "  --help        Show this help message"
        echo ""
        echo "Environment Variables:"
        echo "  MCORE_BASE_PATH     Base path for checkpoints (default: /mnt/checkpoint/ncai/multimodal/outputs)"
        echo "  CUTOFF_TIMESTAMP    Only convert checkpoints created after this Unix timestamp"
        echo "  STAGES              Space-separated list of stages to scan"
        echo "  DRY_RUN             Set to 'true' for dry run mode"
        echo "  SEND_WEBHOOK        Set to 'false' to disable webhook notifications"
        echo "  WEBHOOK_URL         Custom webhook URL"
        exit 0
        ;;
    *)
        main
        ;;
esac

