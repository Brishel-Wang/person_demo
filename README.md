# 单路人体检索与大模型报告 Demo（spec v1.6.1）

按交接包 `单路人体检索与大模型报告Demo_开发交接包_v1.6.1/docs/spec_v1.6.1.md` 实现。
本目录是独立代码目录；交接资料保持原样未改动。

## Video RAG 工作台

在工作台选择「上传图片」，框选人体或用整张裁剪图，再发送即可检索。图片查询默认搜索从 00:00 开始的最长已发布范围。完整索引上线后，`cam1.mp4` 的查询范围为 00:00–34:59.614。可在问题中指定片内时间，或用「全片」恢复完整范围。未附图片时可按片内时间或轨迹号查询已发布观测。

全片外观索引（`dense_appearance_v2`）以 10 FPS 读取原片。每个有有效裁剪的 YOLO 人体检测框都有 512 维 OSNet 向量，按观测 ID 写入 Milvus `body_frames_<space_hash12>_v2`。ByteTrack 将关联观测归入片段，单帧检测同样保留。片段原型、代表图和原 v1.6.1 检索 API 继续可用。

工作台在已发布范围内逐帧计算余弦相似度，再按 ByteTrack 划分的轨迹片段归组，最高分的观测成为该片段的回放锚点。归组减少重复候选和后续数据回查，首轮向量比较仍覆盖已索引的观测。首条默认自动播放，可切换相似度门槛、按时间排序、人工确认/排除、导出 JSON/CSV 或生成正式报告。Qwen 引用可播放的证据。遮挡、检测缺失或外观变化仍可能漏掉真实出现，候选需要人工复核。

升级时，在工程根目录执行以下命令。已有同配置处理代会复用。原片路径及解码/权重约定见下文。

```bash
source .venv/bin/activate
alembic upgrade head
project ingest --vision-profile dense_appearance_v2 --window-us 0:2099614183 --artifact-staging-dir /tmp
project index reconcile
npm --prefix frontend run build
```

随后用现有服务配置重启 API。指定暂存目录后，逐帧分析的代表图、向量与行索引会统一归档。存储与启动细节见 [SETUP.md](docs/SETUP.md)。

新分析默认用 YOLO26x 及 1920 输入，只输出 `person`。工作台「视频知识库」显示当前索引和新分析各用的权重与输入尺寸。新检索优先选用覆盖请求范围的已发布逐帧索引，历史任务按创建时冻结的处理代提供回放。模型和尺寸参与相同窗口的处理代复用判断。

## 功能

- 视频导入：`/root/ppt-agent-wildtrack/Wildtrack/Wildtrack/cam1.mp4`（只读），旧版本演示默认前 200 秒，逐帧外观索引覆盖全片，均按 10 FPS 分析
- YOLO26x person 检测（新分析默认 1920）→ 本地 ByteTrack 单镜头跟踪（30 秒分块 + continuation）→ 每 chunk 最多 5 张代表图
  （等分时间槽 + 128×256 灰度拉普拉斯方差，间隔 ≥0.5s）→ OSNet-AIN x1.0 512 维 L2 归一化特征
  → 原型 `mean_l2_v1` 入 Milvus（FLAT/COSINE/Strong），代表向量 npy + JSON 行索引存 storage 用于重排
- PostgreSQL 17：素材/camera/观测/tracklet/continuation/outbox/覆盖/任务/复核/报告/元数据版本
- 旧版片段原型检索：60 秒桶 × Top20（40/80/160/200 补取）、冻结快照、SQL 回查、
  排除查询来源 chunk（intra_camera_demo）或完整来源组（intra_camera_holdout）、
  同跟踪段后续片段明确标记、代表向量重排（top min(2,n) 均值）、稳定排序
- 原片回放：直接读取原片，按容器时间映射跳转；实际观测帧单独显示对应目标框；人工确认/排除（expected_revision，冲突 409）
- 报告：结构化证据包 → Qwen/Qwen3.5-9B（Transformers 本地推理，do_sample=False、模板关闭 thinking）
  → JSON Schema + 引用白名单校验 → 可点击引用；失败降级模板（provider=template, DEGRADED，不计 D6 通过）
- 前端：React + TypeScript（数据集状态、查询目标选择/上传、候选复核、证据回放、报告、导出 JSON/CSV）

## 目录

```
person_demo/
├── src/person_demo/     # Python 包（api/vision/vectors/retrieval/reports/evidence/media/db）
├── frontend/            # React + TS + Vite
├── migrations/          # Alembic
├── config/              # profiles、manifest（resolved）、模型登记、schema、系统提示
├── tests/               # 单元 + 集成测试
├── scripts/             # doctor / probe / e2e 演示脚本
├── storage/             # 运行时产物（代理视频/裁剪图/向量/行索引），勿提交
├── models/              # 模型权重（symlink 或目录），见 docs/MODEL_PREP.md
├── docs/                # 启动、模型准备、验证记录
└── PROGRESS.md          # 进度与实测结果
```

## 快速开始

```bash
# 1. 依赖（Python 3.11 venv，见 docs/SETUP.md）
uv pip install --python <venv>/bin/python -e ".[dev]"

# 2. 数据库（PostgreSQL 17 + Milvus 2.6.x，见 docs/SETUP.md）
alembic upgrade head

# 3. 模型（见 docs/MODEL_PREP.md：yolo26x.pt、osnet checkpoint、Qwen3.5-9B）

# 4. 检查环境
project doctor --profile single_body_rag_demo_linux_milvus

# 5. probe 视频并生成 resolved manifest
project dataset probe --manifest config/camera_manifest.json

# 6. 入库（窗口 0–200s）
project ingest --dataset single_camera_demo_v1 --window-us 0:200000000
project index reconcile --dataset single_camera_demo_v1

# 7. 启动 API + workers（同机）
PERSON_DEMO_PORT=8010 project serve --profile single_body_rag_demo_linux_milvus # node9 历史入口；已运行时选其他端口
# 常规 Linux 启动默认端口 8000；按需用 PERSON_DEMO_PORT 覆盖。
# 启动前 cd frontend && npm ci && npm run build，返回 person_demo 后运行 serve。
# 从本机或经 SSH 端口转发打开服务。
# 前端（开发模式）：cd frontend && npm install --registry=https://registry.npmmirror.com && npm run dev
```

## 测试与验证结果

```bash
pytest tests/ -x -q           # 集成测试须使用可清理的专用 PostgreSQL 数据库
scripts/e2e_demo.sh           # 端到端演示脚本
```

本轮复审见 [REVIEW_FINDINGS.md](REVIEW_FINDINGS.md) 与 [VALIDATION_REPORT.md](VALIDATION_REPORT.md)，
原开发记录见 [PROGRESS.md](PROGRESS.md) 与 `docs/validation/`。
node9 的演示入口为 8010。历次隔离和上线验收的端口、处理代、任务和报告记录在 `docs/validation/`。录入、搜索和报告均使用同一份 cam1.mp4。

详细启动与模型准备见 [SETUP.md](docs/SETUP.md) 与 [MODEL_PREP.md](docs/MODEL_PREP.md)。
