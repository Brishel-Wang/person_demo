#!/usr/bin/env bash
# 端到端演示：真实视频 → 入库 → 发布 → 真实检索 → 人工复核 → 真实 Qwen 报告。
# 运行前提：PG/Milvus 在线、模型已下载、venv 已安装。
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="/local_data/person_demo/venv/bin:$PATH"
PY=/local_data/person_demo/venv/bin/python
API="${API:-http://127.0.0.1:8010}"

echo "== 1. doctor =="
$PY -m person_demo.cli doctor --profile single_body_rag_demo_linux_milvus | head -40

echo "== 2. probe =="
$PY -m person_demo.cli dataset probe --manifest config/camera_manifest.json \
  --out config/camera_manifest.resolved.json

echo "== 3. import (API) =="
curl -s -X POST $API/v1/datasets/import -H 'Content-Type: application/json' -d '{
  "dataset_id": "single_camera_demo_v1",
  "cameras": [{
    "camera_code": "cam_01", "filename": "cam1.mp4",
    "local_path": "/root/ppt-agent-wildtrack/Wildtrack/Wildtrack/cam1.mp4"
  }],
  "requested_window_us": {"start": 0, "end": 200000000}
}' | tee docs/validation/import.json
echo

echo "== 4. 等待入库与发布（轮询状态） =="
for i in $(seq 1 240); do
  S=$(curl -s $API/v1/datasets/single_camera_demo_v1/status)
  ST=$(echo "$S" | $PY -c 'import json,sys; print(json.load(sys.stdin).get("ingest_status",""))')
  WM=$(echo "$S" | $PY -c 'import json,sys; print(json.load(sys.stdin).get("watermark",""))')
  echo "poll $i: ingest=$ST watermark=$WM"
  if [ "$ST" = "publishing" ] || [ "$ST" = "ready" ]; then
    N=$(curl -s $API/v1/datasets/single_camera_demo_v1/status | $PY -c 'import json,sys; print(json.load(sys.stdin).get("tracklet_count",0))')
    if [ "${WM:-none}" != "none" ] && [ "${WM:-0}" -ge 0 ] && [ "$N" -gt 0 ]; then
      echo "ingest+index ready"; break
    fi
  fi
  sleep 5
done
curl -s $API/v1/datasets/single_camera_demo_v1/status | tee docs/validation/status_after_ingest.json
echo

echo "== 5. 创建查询（取第一个代表观测，来源可信） =="
ASSET=$($PY -c 'import json; print(json.load(open("docs/validation/status_after_ingest.json"))["source_asset_id"])')
OBS=$($PY -c "
import json, urllib.request
d = json.load(urllib.request.urlopen('$API/v1/observations/frames?source_asset_id=$ASSET&representative_only=true&limit=5'))
print(d['items'][0]['observation_id'])")
echo "asset=$ASSET obs=$OBS"
curl -s -X POST $API/v1/query-assets/from-frame -H 'Content-Type: application/json' \
  -d "{\"source_asset_id\": \"$ASSET\", \"observation_id\": \"$OBS\"}" \
  | tee docs/validation/query.json
QA=$($PY -c 'import json; print(json.load(open("docs/validation/query.json"))["asset_id"])')
echo

echo "== 6. 检索（intra_camera_demo） =="
curl -s -X POST $API/v1/search-tasks -H 'Content-Type: application/json' -d "{
  \"dataset_id\": \"single_camera_demo_v1\",
  \"mode\": \"body_only\",
  \"query_asset_id\": \"$QA\",
  \"evaluation_scope\": \"intra_camera_demo\",
  \"scope\": {\"asset_windows\": [{\"source_asset_id\": \"$ASSET\", \"t_start_local_us\": 0, \"t_end_local_us\": 200000000}]}
}" | tee docs/validation/search_task.json
TASK=$($PY -c 'import json; print(json.load(open("docs/validation/search_task.json"))["task_id"])')
for i in $(seq 1 60); do
  ST=$(curl -s $API/v1/search-tasks/$TASK | $PY -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  echo "poll $i: $ST"
  case "$ST" in COMPLETED|PARTIAL|FAILED) break;; esac
  sleep 2
done
curl -s "$API/v1/search-tasks/$TASK/results?page=1&page_size=50" \
  | tee docs/validation/search_results.json
echo

echo "== 7. 人工复核（确认 rank0 候选） =="
HIT=$($PY -c 'import json; d=json.load(open("docs/validation/search_results.json")); print(d["items"][0]["hit_id"]) if d["items"] else print("")')
REV=$($PY -c 'import json; d=json.load(open("docs/validation/search_results.json")); print(d["items"][0]["review_revision"]) if d["items"] else print("0")')
if [ -n "$HIT" ]; then
  curl -s -X POST $API/v1/search-tasks/$TASK/reviews -H 'Content-Type: application/json' \
    -d "{\"hit_id\": \"$HIT\", \"action\": \"confirm\", \"expected_revision\": $REV}" \
    | tee docs/validation/review.json
fi
echo

echo "== 8. 报告（真实 Qwen） =="
curl -s -X POST $API/v1/search-tasks/$TASK/reports -H 'Content-Type: application/json' \
  -d '{"expected_search_revision": 1, "report_profile": "qwen_single_report_local_v1"}' \
  | tee docs/validation/report_job.json
RJOB=$($PY -c 'import json; print(json.load(open("docs/validation/report_job.json"))["report_job_id"])')
for i in $(seq 1 240); do
  J=$(curl -s $API/v1/report-jobs/$RJOB)
  ST=$(echo "$J" | $PY -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  echo "poll $i: $ST"
  case "$ST" in SUCCEEDED|DEGRADED|FAILED) break;; esac
  sleep 2
done
curl -s $API/v1/report-jobs/$RJOB | tee docs/validation/report_job_done.json
echo

echo "== 9. 导出 =="
curl -s "$API/v1/search-tasks/$TASK/export?fmt=json" -o docs/validation/export.json
curl -s "$API/v1/search-tasks/$TASK/export?fmt=csv" -o docs/validation/export.csv
echo "done. 产物在 docs/validation/。"
