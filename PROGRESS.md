# person_demo 进度与实测记录

> 目标：按 v1.6.1 规范完成单路视频人体检索与大模型报告 demo 并验证到可演示状态。
> 开发服务器：Linux node9（Ubuntu 22.04, 5.15 内核），8×A800 80GB，2TB RAM。
> 固定输入视频：`/root/ppt-agent-wildtrack/Wildtrack/Wildtrack/cam1.mp4`。

## 2026-11-21 YOLO26x 全片索引上线

- node9 的 8010 已使用 YOLO26x、输入尺寸 1920。全片采样范围 00:00–34:59.614，10 FPS，20,996 帧。PostgreSQL 与 Milvus 核对一致，914,718 条有效人体观测和 108,350 个已发布轨迹片段。
- 保留 ByteTrack 与逐帧 OSNet 特征。工作台按轨迹片段归组达标向量并回放最高分观测。独立扫描全量向量复现后排样本的 998 条逐帧命中、326 个候选。
- 用户先前上传的图片沿用旧对话追问成功，冻结新处理代。首帧得分 0.947887，23 个候选正确导出。另一张后排图在浏览器自动跳至 03:45.008。真实 Qwen 对话均返回可播放引用。
- 验收样本通过复核修订冲突检查，并生成复核前和复核后两份真实 Qwen 报告。报告引用与复核状态一致，指定版本可以导出。
- 模型、处理时长、显存、任务 ID、复算与回放见 [YOLO26x 全片验收](docs/validation/yolo26x_full_20261120/README.md)。

## 2026-11-20 后排行人检测

- 对照 03:45.008 的同一原片帧。YOLO11s 640 输出 13 个框，YOLO26x 1920 输出 42 个框。对照两个人体姿态模型和更高输入尺寸后，选用 YOLO26x 1920，只输出人体。
- 隔离环境完成 224–227 秒的检测、跟踪、OSNet 和 Milvus，1,237 条有效观测与 1,237 个向量对应。8010 已启用新分析默认值，并在浏览器核对原全片索引的回放。[同帧复测](docs/validation/detector_20261120/README.md)

## 2026-11-20 图片检索召回修复

- 对照用户截图与原片首帧复现漏召回。YOLO11s 目标置信度 0.904，旧跟踪与入库未保存该条观测。修复框宽计算与匹配，新增每条有效检测对应的 OSNet 向量索引，保留片段原型和历史结果。
- 工作台图片搜索改用最长已发布窗口和逐帧余弦阈值，返回全部达标片段。自动播放与导出携带最高分对应的实际帧时间。
- 隔离环境已通过上传、首帧命中、真实 Qwen、浏览器播放、复核、导出、阈值切换和写入后超时重试。首帧重算余弦 0.937264。
- 8010 已切换到全片索引。分析 20,996 帧，发布 275,730 条有效人体向量。使用原截图复测命中首帧，阈值 0.68 返回 16 个待复核候选。证据见 [逐帧检索复测](docs/validation/recall_20261120/README.md)。
- 单帧候选回放已上线。实际上传仅一帧的入库裁剪，搜回 03:30.009，并验证上下文回放结束后停回命中帧。

## 2026-11-19 Video RAG 工作台

- 已上线到 node9 8010。新增多轮视频对话、图片/原片点选人体、按片内时间检索、真实 Qwen 证据问答，以及与候选和引用联动的原片播放器。
- Chromium 实测对话恢复、时间检索、外观检索、真实起播、复核与 CSV 导出；前端增加上传框选及自托管中文字体。发布增加对话表，保留已有 2 个处理代和历史证据。
- 25 条自动化用例通过。实际证据 ID、资源与截图见 [工作台验收](docs/validation/video_rag_20261119/README.md)。

## 2026-11-19 服务器复审与修复

- 已在 node9 独立重跑指定原片的前 200 秒，10 FPS。2,000 帧、5,679 条观测、1,285 个片段进入 PostgreSQL/Milvus，发布水位 1,284。相同窗口再次导入复用原处理代，重新发布 0 行。
- 修正来源观测与处理代约束、召回补取与跨桶计数、连续发布、报告迁移、原片时间映射、报告引用及候选措辞。保留 YOLO11s 和既有 OSNet 权重及向量空间，也保留原全长处理代。
- 真正运行了 Qwen3.5-9B。报告校验区分相似候选、人工确认和同跟踪段后续片段。隔离任务中的目视确认只涵盖同跟踪段后续片段。
- 正样本图库尚未独立标注，独立评测 Recall 为 null。后续人脸、实时流、跨机位轨迹和 Windows 实机另行验收。
- 8010 已同步升级并通过切换后冒烟。切换前后共有 1 个原素材、2 个处理代、38 个检索任务、6 条复核和 19 份报告。
- 本轮依据以 [审查问题](REVIEW_FINDINGS.md) 和 [复测报告](VALIDATION_REPORT.md) 为准。下方为原开发记录。

## 原开发记录（历史自报）

真实链路：视频 probe → YOLO11s+ByteTrack（2000 帧、1285 片段、5679 观测）→ OSNet 512 维
→ Milvus 1285 行发布（水位 1284）→ 单路检索（4 桶 80 候选、来源 chunk 排除、代表向量重排）
→ 人工确认/排除（409 冲突）→ 真实 Qwen3.5-9B 报告（provider=qwen，引用校验 0 错误）
→ 导出 JSON/CSV。独立评测：holdout 口径排除整组后 1 个独立正样本，Recall@1/5/20=1。

## 环境核查结果（2026-11-18/19）

| 项 | 结果 |
|---|---|
| 视频文件 | 存在、可读；7,902,756,029 bytes 与 manifest 预期一致；SHA256 `ea57febe…d6a4` |
| probe | h264 1920x1080 59.94fps VFR=false；2099.6s（125851 帧）；time_base 1/180000；pts_anchor 70740；[0,200s) 窗口有效 |
| GPU | 8×A800 80GB（共享节点；演示用 cuda:0 单卡串行） |
| Python | CPython 3.11.15（uv 安装，/local_data/person_demo/venv，符号链接 .venv） |
| 依赖 | torch 2.7.0+cu126、torchvision 0.22.0、ultralytics 8.4.215、transformers 5.23.0、pymilvus 2.6.18、av 18.1.0、cv2 5.1.0、SQLAlchemy 2.x、Alembic 1.20.1、FastAPI 等；锁定见 uv.lock 与 requirements.lock |
| PostgreSQL 17 | 17.12（阿里云 PGDG apt），端口 5433，库 person_demo / person_demo_test |
| Milvus | milvusdb/milvus:v2.6.4 standalone 容器 + bitnami/etcd:3.5（v2.6 无内嵌 etcd）；健康检查通过 |
| 模型 | yolo11s.pt `85a76fe8…d502d5`；OSNet msmt17 `8a07e8da…d919f`（552/552 层匹配，512 维）；Qwen3.5-9B ModelScope 4 分片 19.3GB（sha256 已核对） |
| 网络约束 | GitHub/HF/docker.io 直连不可达 → 经中国镜像获取；torchreid 架构按 MIT 许可 vendor（仅去 gdown，无结构改动） |

## 阶段进度与实测

- **D0** ✅ 工程/迁移（21 表）/Milvus/PG/模型注册/探针：YOLO 16–20 人/帧；OSNet 范数 1.0；Qwen 加载 8.2s、烟测 JSON 正确、thinking 关闭经模板验证。feature_space_id=`8337a767…b26b17c`。
- **D1** ✅ probe + resolved manifest（ingest_ready）；10FPS 采样修正（59.94fps 下按 local_us 桶采样），2000 帧间隔均值 100.0ms。
- **D2** ✅ 入库 200s 窗口：1285 chunk（continuation_group/30s 分块/失联边界）、5679 观测、代表图（等分槽+拉普拉斯方差）、OSNet 原型 mean_l2_v1；产物 97MB。
- **D3** ✅ outbox→Milvus upsert（稳定主键幂等）→Strong 校验→连续水位 1284；重试不翻倍；瞬时故障重试通过。
- **D4** ✅ 检索 p50 1.54s/p95 1.84s；80 候选（4 桶预算）；来源 chunk 排除、holdout 整组排除（0 同组候选）；代表向量重排与原型分独立；证据回放（bbox 序列 + 代理视频 2000 帧映射）。
- **D5** ✅ 前端构建并托管（8010 端口）；复核 confirm/reject + expected_revision 409；导出 JSON/CSV（revision+report_id）；26 项用例映射见 docs/validation/cases.md；独立评测 Recall@1/5/20=1（1 个独立正样本，无正样本时会如实 INSUFFICIENT_POSITIVE_GALLERY）。
- **D6** ✅ 真实 Qwen 报告 provider=qwen：加载 10.2s、生成 72s、总 84.5s、峰值 17.1GB、任务后释放；输入 2081 tokens；校验 errors=[] repairs=0；claims 引用真实 evidence_id、reviewed_observation 仅用于 confirmed、hypotheses 空、片内时间不补造日期地点。此前两次模板降级（JSON 截断）已记录，不计 D6 通过。

## 演示入口

- API + 前端：http://127.0.0.1:8010（`project serve`，前端构建产物由后端托管）
- 复核/报告产物：docs/validation/（query.json、search_results.json、evidence.json、report.json、export.csv、resource_report.md）

## 已知约束（如实记录）

1. docker.io/GitHub/HF 直连不可达 → 镜像源获取；Milvus 2.6.4 standalone 需独立 etcd 容器（官方等价部署）。
2. 共享节点 GPU 动态占用；演示单卡串行（cuda:0），可用 PERSON_DEMO_DEVICE 换卡。
3. 无真实日期/地点：全程片内时间（asset_relative），报告不补造。
4. Qwen3.5-9B 未装 causal-conv1d/flash-linear-attention（GitHub 不可达），使用参考实现，生成 72s/次；后续可优化。
5. Windows RTX4090 24GB（WSL2/原生）路径已预留（VectorStore 接口 + future_deployment 配置），待目标机实测。

## 第二轮进展（2026-11-19 下午）

- ✅ 同一跟踪段后续片段标记端到端验证：查询 track 13877 chunk0（片内 175.9s）→
  其 chunk1（181.8–188.5s）以 `same_continuation_as_query=true` 出现在候选（proto 0.84 / rerank 0.82），
  来源 chunk 自身被排除（docs/validation/continuation_demo.json）。
- ✅ **全长 2099.6s 入库完成**：37.6 分钟（cuda:1），8109 片段、74018 观测，
  outbox 全部发布、水位 8108；resolved 窗口扩展逻辑已补（CLI/import 同步扩展）。
- ✅ **全长检索验证**：35 桶 × Top20 召回 1820 → 轮询保留 100（cap 截断，覆盖全部 35 桶，
  snapshot.result 记录 buckets/recalled/kept/truncated）；PARTIAL 状态与截断标记符合规范。
- ✅ **全长复核与报告**：复核后生成新版本真实 Qwen 报告（provider=qwen）：
  reviewed_observation 引用人工确认证据、candidate_observation 引用待复核项、hypotheses 空
  （docs/validation/full/report.json）。
- ✅ 独立评测套件就绪：15 个查询完成 holdout 检索并导出复核图
  （docs/validation/eval_review/，labels.json + scripts/apply_reviews.py 评分流程）；
  标注口径见 docs/validation/evaluation_status.md。
- ✅ 修复三处问题：轮询保留超额（122→100）、snapshot JSONB 原地修改不落库（flag_modified）、
  报告幂等复用误用新 bundle 的 hash（reports 增加 evidence_hash 列 + 迁移 31fcc4be0f2a，
  复核变化 → 新 hash → 新报告版本，旧报告不改）。
- ✅ 回归：20/20 测试通过；数据集累计 9394 片段 / 79697 观测 / 2702 候选。
- ⚠️ 机器辅助复核尝试（本机 Qwen3.5-9B 多模态）输出退化（阴性对照同样 yes），
  **未采用**；独立正样本待人工视觉确认。执行模型无视觉输入 → 复核图留给人工；
  已有 1 个代理确认的独立正样本（Recall@1/5/20=1，待人工复核）。

## 下一步

- 人工填写 eval_review/labels.json → apply_reviews.py 出 Recall 统计
- 全长窗口独立评测（同 15 查询流程，窗口 0:2099614183）
- 人脸、RTSP、跨机位 ReID、轨迹地图（后续扩展）
- Windows 4090 双路径实测
