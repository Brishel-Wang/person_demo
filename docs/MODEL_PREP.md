# 模型准备说明（D0）

服务器网络约束：HuggingFace/GitHub 直连不可达；使用 hf-mirror.com 与 ModelScope。
所有模型显式下载并登记 SHA256，运行时不隐式下载。

## 1. YOLO26x（新的人体检测任务）

在工程根目录准备官方 `yolo26x.pt`，输入尺寸 1920，保留 `person` 类。先下载并校验后安装。

```bash
mkdir -p downloads models
curl -fL -o downloads/yolo26x.pt 'https://modelscope.cn/api/v1/models/Ultralytics/YOLO26/repo?Revision=master&FilePath=yolo26x.pt'
sha256sum -c <<'SHA'
9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92  downloads/yolo26x.pt
SHA
cp downloads/yolo26x.pt models/yolo26x.pt
```

来源：[Ultralytics 权重目录](https://huggingface.co/Ultralytics/YOLO26/blob/4558807/yolo26x.pt)。
正式运行前使用 `alembic upgrade head`，为导入任务记录实际尺寸。

### 历史 YOLO11s 权重（输入 640）

```bash
cd person_demo/downloads
curl -sL -o yolo11s.pt \
  "https://hf-mirror.com/Ultralytics/YOLO11/resolve/main/yolo11s.pt"
sha256sum yolo11s.pt   # 85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5
cp yolo11s.pt ../models/yolo11s.pt
```

来源：https://huggingface.co/Ultralytics/YOLO11（经 hf-mirror 代理）。

## 2. OSNet-AIN x1.0（人体嵌入，msmt17 检查点）

```bash
cd person_demo/downloads
curl -sL -o osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth \
  "https://hf-mirror.com/kaiyangzhou/osnet/resolve/main/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
sha256sum <file>
cp <file> ../models/
```

来源：https://huggingface.co/kaiyangzhou/osnet（作者检查点，MIT 架构代码）。
架构代码 vendored：`src/person_demo/vision/reid/osnet_ain.py`（源自 torchreid/deep-person-reid，
最小补丁：fc=None 匹配官方 msmt17 检查点、移除 gdown 隐式下载；修改处标 PATCHED）。
检查点结构实测：552 keys，`module.` 前缀（DataParallel），classifier 形状 [4101, 512]，
无 bnneck/fc → 特征 512 维、eval 输出全局池化向量。

## 3. Qwen/Qwen3.5-9B（报告生成，Transformers 本地推理）

经 ModelScope（aria2 16 连接，约 12MB/s）：

```bash
cd person_demo/downloads
mkdir -p ../models/qwen3.5-9b
for f in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json \
         vocab.json merges.txt chat_template.jinja preprocessor_config.json \
         video_preprocessor_config.json README.md LICENSE; do
  aria2c -x 8 -s 8 -d ../models/qwen3.5-9b -o "$f" \
    "https://modelscope.cn/models/Qwen/Qwen3.5-9B/resolve/master/$f"
done
for i in 1 2 3 4; do
  f="model.safetensors-0000${i}-of-00004.safetensors"
  aria2c -x 16 -s 16 -d ../models/qwen3.5-9b -o "$f" \
    "https://modelscope.cn/models/Qwen/Qwen3.5-9B/resolve/master/$f"
done
# 校验：每个分片 sha256 与 ModelScope API 的 Sha256 字段一致
```

来源：https://modelscope.cn/models/Qwen/Qwen3.5-9B（apache-2.0）。
config: `architectures=[Qwen3_5ForConditionalGeneration]`, `model_type=qwen3_5`；
权重 4 分片共约 19.3GB，bf16 加载。

加载方式（`src/person_demo/reports/llm.py`）：`AutoModelForCausalLM`（回退
`AutoModelForImageTextToText`），`torch_dtype=bfloat16`，`local_files_only=True`；
`do_sample=False`；探针验证 chat template 支持 `enable_thinking=False`（不靠 /nothink 文字）。

## 4. 模型注册与 feature_space_id

下载完成后运行：

```bash
project doctor --profile single_body_rag_demo_linux_milvus
python scripts/resolve_registry.py   # 填写 config/models.registry.json 与 profiles
```

`feature_space_id` = SHA256(规范化 JSON：架构/检查点 sha256/RGB 256×128/mean/std/
输出层/float32/L2/接口版本)，Milvus collection = `body_<前12hex>_ar1`。
查询与图库必须同空间；模型或预处理变化 → 新空间/重建原型。

## 本轮 OSNet 空间兼容

实际输入为等比缩放、居中填充 127 的 RGB 256×128，之后应用 ImageNet mean/std。当前查询与图库保持同一预处理，已有空间 ID `8337a7675314855e48907439ceb4b279f8ff44af840a74344f9cd5738b26b17c` 沿用。历史 v1 hash 的 mean/std 字段为空，其实际运行参数以 `feature_space.py` 和编码器为准。更改任一项时需登记新的、包含实际参数的特征空间并重建原型。
