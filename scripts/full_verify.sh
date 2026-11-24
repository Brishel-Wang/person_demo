#!/usr/bin/env bash
# 全长窗口验证：reconcile 发布 → 全长检索 → 复核 → 报告。
# 运行前提：全长 ingest run 已 ready（docs/validation/full_ingest.log 完成）。
set -euo pipefail

cd "$(dirname "$0")/.."
PY=/local_data/person_demo/venv/bin/python
API="${API:-http://127.0.0.1:8010}"
FULL_US=2099614183
ASSET=6efee9f8-71db-439e-adb5-f649a0db054c
mkdir -p docs/validation/full

echo "== 1. reconcile（全长 outbox 发布） =="
$PY -m person_demo.cli index reconcile --dataset single_camera_demo_v1 | tail -3

echo "== 2. 状态 =="
curl -s $API/v1/datasets/single_camera_demo_v1/status | tee docs/validation/full/status.json
echo

echo "== 3. 取片中 500s 附近的代表观测创建查询 =="
OBS=$(psql "postgresql://person_demo:person_demo@127.0.0.1:5433/person_demo" -t -A -c \
 "SELECT observation_id FROM observations WHERE local_us BETWEEN 499000000 AND 501000000 AND is_representative ORDER BY local_us LIMIT 1;")
echo "obs=$OBS"
curl -s -X POST $API/v1/query-assets/from-frame -H 'Content-Type: application/json' \
  -d "{\"source_asset_id\": \"$ASSET\", \"observation_id\": \"$OBS\"}" | tee docs/validation/full/query.json
QA=$($PY -c 'import json; print(json.load(open("docs/validation/full/query.json"))["asset_id"])')
echo

echo "== 4. 全长检索（35 桶 × Top20，上限 100 轮询保留） =="
curl -s -X POST $API/v1/search-tasks -H 'Content-Type: application/json' -d "{
  \"dataset_id\": \"single_camera_demo_v1\", \"mode\": \"body_only\",
  \"query_asset_id\": \"$QA\", \"evaluation_scope\": \"intra_camera_demo\",
  \"scope\": {\"asset_windows\": [{\"source_asset_id\": \"$ASSET\", \"t_start_local_us\": 0, \"t_end_local_us\": $FULL_US}]}
}" | tee docs/validation/full/search_task.json
TASK=$($PY -c 'import json; print(json.load(open("docs/validation/full/search_task.json"))["task_id"])')
for i in $(seq 1 120); do
  ST=$(curl -s $API/v1/search-tasks/$TASK | $PY -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  echo "poll $i: $ST"
  case "$ST" in COMPLETED|PARTIAL|FAILED) break;; esac
  sleep 3
done
curl -s "$API/v1/search-tasks/$TASK/results?page=1&page_size=100" | tee docs/validation/full/search_results.json
echo

echo "== 5. 复核 rank0 并生成报告 =="
HIT=$($PY -c 'import json; d=json.load(open("docs/validation/full/search_results.json")); print(d["items"][0]["hit_id"]) if d["items"] else print("")')
REV=$($PY -c 'import json; d=json.load(open("docs/validation/full/search_results.json")); print(d["items"][0]["review_revision"]) if d["items"] else print("0")')
if [ -n "$HIT" ]; then
  curl -s -X POST $API/v1/search-tasks/$TASK/reviews -H 'Content-Type: application/json' \
    -d "{\"hit_id\": \"$HIT\", \"action\": \"confirm\", \"expected_revision\": $REV}" | tee docs/validation/full/review.json
fi
curl -s -X POST $API/v1/search-tasks/$TASK/reports -H 'Content-Type: application/json' \
  -d '{"expected_search_revision": 1, "report_profile": "qwen_single_report_local_v1"}' | tee docs/validation/full/report_job.json
RJOB=$($PY -c 'import json; print(json.load(open("docs/validation/full/report_job.json"))["report_job_id"])')
for i in $(seq 1 120); do
  J=$(curl -s $API/v1/report-jobs/$RJOB)
  ST=$(echo "$J" | $PY -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  echo "report poll $i: $ST"
  case "$ST" in SUCCEEDED|DEGRADED|FAILED) break;; esac
  sleep 5
done
curl -s $API/v1/report-jobs/$RJOB | tee docs/validation/full/report_job_done.json
echo
echo "done. 产物在 docs/validation/full/"
