# 启动说明（Linux 开发服务器）


## 建立工作台全片索引

在工程根目录使用已配置的 PostgreSQL、Milvus 和模型运行：

```bash
alembic upgrade head
project ingest --vision-profile dense_appearance_v2 --window-us 0:2099614183 --artifact-staging-dir /tmp
project index reconcile
npm --prefix frontend run build
```

`0:2099614183` 是本机 `cam1.mp4` 的完整片内范围（微秒）。原片只读。逐帧处理可明确设置空闲的 `PERSON_DEMO_DEVICE`，默认检测权重是 `yolo26x.pt`，输入尺寸为 1920。指定本机暂存目录后，代表图、向量和行索引写入单个 SQLite 产物包。暂存需容纳产物包和解码证据，并预留至少 5 GiB。完成完整性检查后将包放入 `storage/bundles/<处理代 UUID>.sqlite3`，随其他产物一并备份。历史散文件证据仍可访问。

视觉处理完成后，通过现有后台发布器或 `project index reconcile` 发布即可以检索。工作台和新片段原型检索优先采用覆盖请求范围的已发布逐帧索引。服务部署沿用原有配置。历史任务仍按冻结的处理代访问原片、复核和报告。

## 依赖版本

- Python 3.11.15（uv 安装于 `/local_data/person_demo/venv`；本项目 `.venv` 符号链接指向它）
- PostgreSQL 17.12（阿里云 PGDG apt）
- Milvus 2.6.4 standalone（CPU，docker 容器，端口 19530）
- Node 22 + npm 10（前端构建）

## 1. PostgreSQL 初始化（首次）

```bash
mkdir -p /local_data/person_demo/pgdata
chown postgres:postgres /local_data/person_demo/pgdata
sudo -u postgres /usr/lib/postgresql/17/bin/initdb -D /local_data/person_demo/pgdata \
  -E UTF8 --locale=C.UTF-8
sudo -u postgres /usr/lib/postgresql/17/bin/pg_ctl -D /local_data/person_demo/pgdata \
  -o "-p 5433 -c listen_addresses=127.0.0.1" -l /local_data/person_demo/pg.log start
sudo -u postgres psql -p 5433 -c "CREATE USER person_demo WITH PASSWORD 'person_demo';"
sudo -u postgres psql -p 5433 -c "CREATE DATABASE person_demo OWNER person_demo;"
sudo -u postgres psql -p 5433 -c "CREATE USER person_demo_test WITH PASSWORD 'person_demo_test';"
sudo -u postgres psql -p 5433 -c "CREATE DATABASE person_demo_test OWNER person_demo_test;"
```

## 2. Milvus

node9 当前使用 `person-demo-milvus`（`milvusdb/milvus:v2.6.4`）与独立的
`person-demo-etcd`（`bitnami/etcd:3.5`）。复用已有容器和持久卷。
可用 `docker ps --filter name=person-demo` 与 `curl -f http://127.0.0.1:9091/healthz` 检查状态。
新环境按固定版本启动 Standalone 与独立 etcd，Milvus 使用本地持久化存储和可访问的 etcd 地址。
重跑验收时通过 `PERSON_DEMO_MILVUS_DATABASE` 单独指定新逻辑库。

## 3. 应用配置

```bash
cp .env.example .env   # 按需修改 DEVICE（默认 cuda:0）、端口等
cd person_demo
export VIRTUAL_ENV=/local_data/person_demo/venv
uv pip install --python /local_data/person_demo/venv/bin/python -e ".[dev]"
alembic upgrade head
cd frontend && npm ci && npm run build
cd ..
```

`55c2c82b4d17` 补齐从旧版本升级时缺失的 `reports.evidence_hash`，已有该列的数据库可直接升级。迁移 `a88bd31a6f20` 增加视频对话表，保留既有人体向量、任务与报告。
测试使用专用库，因测试初始化会重建表。默认 API 绑定 127.0.0.1，远程浏览器使用 SSH 端口转发。


## 4. 一次完整运行

```bash
project doctor --profile single_body_rag_demo_linux_milvus
project dataset probe --manifest config/camera_manifest.json      # → config/camera_manifest.resolved.json
project ingest --dataset single_camera_demo_v1 --window-us 0:200000000
project index reconcile --dataset single_camera_demo_v1
PERSON_DEMO_PORT=8000 project serve --profile single_body_rag_demo_linux_milvus
```

浏览器打开 http://127.0.0.1:8000（若端口被占用，用 PERSON_DEMO_PORT 指定空闲端口），
或开发模式：`cd frontend && npm install --registry=https://registry.npmmirror.com && npm run dev`（5173 代理 /v1）。

服务从配置的 cam1.mp4 提供原片字节范围，指定观测帧按保存的 PTS 解码并显示对应框。
节点上的 8010 服务已升级到本轮版本，独立验收在 18011 完成。8010 沿用节点原有监听地址。默认演示检索前最多 200 秒，已有全长处理代仍可通过匹配其窗口和观测的 API 请求访问。

## 5. 报告模型探针

```bash
project report doctor --profile qwen_single_report_local_v1
```

## 6. 测试与评测

```bash
pytest tests/ -q
project evaluate --suite single_camera_demo
scripts/e2e_demo.sh
```

## Windows RTX4090 24GB 部署路径（后续）

- WSL2：官方 milvus standalone docker compose + 同一 FastAPI/worker（量化与显存另测）
- 纯原生：pgvector 适配器（代码中 VectorStore 接口预留），OSNet/YOLO/Qwen 的
  Windows 运行时分别登记；均需在目标 4090 实测，不以 A800/Linux 结果代替。
