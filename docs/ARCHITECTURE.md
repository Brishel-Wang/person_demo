# 架构说明

## 模块与数据流

```
交接包 manifest (config/camera_manifest.json)
        │  probe (media/probe.py, PyAV)
        ▼
resolved manifest (sha256/编码/尺寸/PTS锚点/time_base/resolved窗口)
        │  import (POST /v1/datasets/import) → cameras + source_assets + ingest_runs
        ▼
video worker (vision/ingest_worker.py, 单 GPU 串行, 经 gpu_lock)
   窗口帧(10FPS, PTS采样) → YOLO11s person(批量) → ByteTrack(本地实现,30帧缓冲)
   → 30s分块 tracklet(continuation_of/group) → 代表图(等分槽+拉普拉斯方差)
   → OSNet-AIN x1.0 (vendored torchreid, 官方msmt17检查点) 512维L2
   → 原型 mean_l2_v1 → 不可变产物(npy/jsonl/crops, 先临时核验后落位)
   → 单 SQL 事务: tracklets+observations+coverage+outbox(含publish_seq)
        │
publisher (vectors/publisher.py)
   outbox → Milvus upsert(256/批, 稳定主键幂等) → Strong get 校验
   → SQL ready → 连续水位(不跨空洞)
        │
查询 (POST /v1/query-assets/from-frame | upload)
   服务端解析来源(observation→tracklet→group) / origin_unknown
   → 相同裁剪与预处理 → 相同空间 OSNet 编码 → query_assets
        │
search (retrieval/service.py, POST /v1/search-tasks → worker)
   冻结快照(活动run/水位/覆盖/空间/排除规则)
   → 60s桶×Top20(40/80/160/200补取) FLAT/COSINE/Strong
   → SQL回查(ready/墓碑/活动run/媒体/observation)
   → 来源排除: demo 排除来源chunk(同组后续chunk标记) / holdout 排除整组
   → 去重+命中桶列表 → 100上限轮询 → 代表向量重排(top min(2,n)均值)
   → 稳定排序 → 物化 search_candidates(pending_review)
        │
复核 (POST /v1/search-tasks/{id}/reviews, expected_revision 冲突409)
        │
report worker (reports/__init__.py, 同 gpu_lock 串行)
   证据包(≤14条, confirmed优先, 片内时间排序, 截断记录)
   → token预算(6144, 确定性裁剪) → Qwen3.5-9B(do_sample=False,模板关thinking)
   → JSON解析+Schema+引用白名单+人工状态校验(至多修复1次)
   → 失败: 模板 DEGRADED(不计D6) ; 成功: provider=qwen
   → reports/report_claims 持久化(资源与耗时记录)
        │
前端 (React/TS): 状态 → 选目标 → 候选/复核/证据回放 → 报告(可点击引用) → 导出
```

## 关键约定

- `local_us` 一律 asset_relative：`round((pts-pts_anchor)*time_base*1e6)`，有理数计算；
  第一帧为片内原点，不是真实午夜。
- Milvus collection `body_<space_hash12>_ar1`（schema 1.4 字段契约，FLAT/COSINE/CPU/Strong，
  auto_id=false，动态字段关闭）；`embedding_id` = SHA256(紧凑 JSON 数组) 稳定主键。
- 代表向量只存 storage（npy + JSON 行索引），Milvus 只存每片段一个原型；
  重排分独立于原型分，产物损坏标 `rerank_failed`。
- 单路检索不排除整个源相机；demo 口径排除 origin chunk，holdout 口径排除完整
  continuation group；`same_continuation_as_query=true` 必须界面标记，不计独立成果。
- 报告只整理检索证据与人工结论；`hypotheses` 为空；不补造日期/地点/身份/跨机位路线；
  reviewed_observation 只能引用 confirmed 证据。
- 单 GPU 串行调度：视频入库/查询编码与 Qwen 生成共用 `gpu_lock`；
  Qwen 按需加载、任务后释放并记录资源（进程退出前测）。
