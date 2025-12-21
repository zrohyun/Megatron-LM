PREFIX="h100-i001v8-w-"
GPU_PER_NODE=8

# 대상 네임스페이스 목록(필요 시 추가)
NAMESPACES=("p-ncai-etri-wbl" "p-ncai-wbl")

NODE_SUFFIXES=(
  00d7 016a 03a2 0489 04e0 05fe 07b9 07e2 08b7 0a25 0ac8 0b0f 0d3b
  1250 1384 1535 1771 17f4 1871 1d7f 2346 2a95 2d70 3052 30c8 3176
  358a 3677 36d5 3884 3917 3a18 412e 42d9 42ec 440b 467b 4b78 4bb2
  4bc3 4c65 4d2f 4d36 4dfe 4f85 4fd0 51f9 540d 5423 5a03 5aac 5c4c
  5f03 625f 6338 673e 6890 6c55 6cb8 6d14 6e45 6eff 70a5 713d 71d4
  729b 746b 74a4 7813 78f2 7af3 7c0e 7cac 7e23 7eb9 81c6 8433 85cb
  860e 8873 8a4d 8e38 9368 9409 947b 9b5e 9cff a2fc a453 a5fa a92f
  a9d4 adbf afc4 b35b b7e1 bbdf bc62 c2fc c34b c851 c853 ca37 cb03
  ccb3 ce42 ce57 cfdb d13c d1cf d29f d73c da01 dacc ddb5 e14d e4ff
  e703 e738 eb2e ec76 edcf ee11 ee73 f11f f16c f27c f34c
)

# 전체 합산 변수
TOTAL_GPU=0
TOTAL_USED_NODES=0
USED_NODES_STR_ALL=""
NS_GPU_TOTALS=()
NS_NODE_TOTALS=()

# 네임스페이스별 조회 및 합산
for ns in "${NAMESPACES[@]}"; do
  READOUT=$(kubectl get pods -n "$ns" --field-selector=status.phase=Running -o custom-columns='NAME:.metadata.name,GPUs:.spec.containers[*].resources.requests.nvidia\.com/gpu,NODE:.spec.nodeName' | awk 'BEGIN { GPU_SUM = 0; USED_NODES_LIST = ""; } NR==1 { next } { gsub(/,/, "", $2); gsub(/[[:space:]]/, "", $3); if ($2 != "" && $2 != "<none>" && $2+0 > 0) { GPU_SUM += $2; USED_NODES_LIST = USED_NODES_LIST $3 " "; if (!($3 in nodes_count)) { nodes_count[$3] = 1; } } } END { printf "GPU_TOTAL=%s; NODE_TOTAL=%s; USED_NODES_STR=\"%s\"", GPU_SUM, length(nodes_count), USED_NODES_LIST; }')

  eval "$READOUT"

  echo "### Running GPU pods in namespace $ns ###"
  echo "TOTAL: $GPU_TOTAL GPUs on $NODE_TOTAL nodes"
  echo "Used Nodes from kubectl ($NODE_TOTAL nodes):"
  echo "$USED_NODES_STR"
  echo

  TOTAL_GPU=$((TOTAL_GPU + GPU_TOTAL))
  TOTAL_USED_NODES=$((TOTAL_USED_NODES + NODE_TOTAL))
  USED_NODES_STR_ALL+="$USED_NODES_STR "
  NS_GPU_TOTALS+=("$GPU_TOTAL")
  NS_NODE_TOTALS+=("$NODE_TOTAL")
done

# 이후 로직을 전체 합산값 기준으로 진행
GPU_TOTAL=$TOTAL_GPU
NODE_TOTAL=$TOTAL_USED_NODES
USED_NODES_STR="$USED_NODES_STR_ALL"

echo "### Running GPU pods across all namespaces ###"
echo "TOTAL: $GPU_TOTAL GPUs on $NODE_TOTAL nodes"
echo "Used Nodes (all namespaces):"
echo "$USED_NODES_STR"
echo

# 네임스페이스별/전체 통계 요약
echo "### Namespace usage summary ###"
GPU_EXPR=""
NODE_EXPR=""
for i in "${!NAMESPACES[@]}"; do
  ns="${NAMESPACES[$i]}"
  g="${NS_GPU_TOTALS[$i]:-0}"
  n="${NS_NODE_TOTALS[$i]:-0}"
  echo "- $ns: $g GPUs on $n nodes"
  if [[ -z "$GPU_EXPR" ]]; then
    GPU_EXPR="$g"
    NODE_EXPR="$n"
  else
    GPU_EXPR="${GPU_EXPR}+${g}"
    NODE_EXPR="${NODE_EXPR}+${n}"
  fi
done
echo "Combined GPUs: ${GPU_EXPR} = $GPU_TOTAL"
echo "Combined nodes: ${NODE_EXPR} = $NODE_TOTAL"
echo

echo "### Unused H100 nodes (Total: ${#NODE_SUFFIXES[@]}) ###"
UNUSED_COUNT=0
UNUSED_LIST=()

for suf in "${NODE_SUFFIXES[@]}"; do
  node="${PREFIX}${suf}"
  if [[ "$USED_NODES_STR" != *"$node"* ]]; then
    UNUSED_LIST+=("$node")
    UNUSED_COUNT=$((UNUSED_COUNT + 1))
  fi
done

if [ $UNUSED_COUNT -gt 0 ]; then
    IFS=','
    echo "${UNUSED_LIST[*]}" | sed 's/,/, /g'
else
    echo "(None)"
fi

echo
echo "Unused H100 nodes: $UNUSED_COUNT"
UNUSED_GPU_TOTAL=$((UNUSED_COUNT * GPU_PER_NODE))
echo "Total Unused GPUs: $UNUSED_GPU_TOTAL ($UNUSED_COUNT unused nodes * $GPU_PER_NODE GPUs/node)"
echo "----------------------------------------------------------------------"
TOTAL_EXPECTED=${#NODE_SUFFIXES[@]}
TOTAL_ACTUAL=$((NODE_TOTAL + UNUSED_COUNT))
echo "합산 확인: Used Nodes ($NODE_TOTAL) + Unused Nodes ($UNUSED_COUNT) = $TOTAL_ACTUAL"
echo "예상 총 노드 수: $TOTAL_EXPECTED (NODE_SUFFIXES 배열 길이)"
echo "실제 총 노드 수: $TOTAL_ACTUAL"
if [ $TOTAL_ACTUAL -eq $TOTAL_EXPECTED ]; then
  echo "총 노드 수가 정확히 ${TOTAL_EXPECTED}개입니다!"
else
  echo "총 노드 수가 일치하지 않습니다. (예상: $TOTAL_EXPECTED, 실제: $TOTAL_ACTUAL)"
fi