# 失效分析 Agentic RAG

这是一个本地优先的失效分析 RAG：文档处理和用户搜索分别运行，检索采用
`FAISS embedding + BM25`，本地证据不足时由 LangGraph 流程自动联网补证。

## 工作流程

```text
markdown / Word / PDF
        │
        ▼
ingest.py：解析 → 切块 → 本地 Embedding
        │
        ├─ data/faiss_index/   （语义索引）
        ├─ data/chunks.json    （BM25 语料）
        └─ data/index_manifest.json

用户问题 + 可选 Word/PDF
        │
        ▼
query.py：Hybrid 检索 → 证据判断 → 必要时联网 → 失效分析与建议
```

## 安装

建议使用 Python 3.11，并在项目内创建虚拟环境：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如果项目中还没有 `.env`，复制配置示例并填写模型接口。`.env` 已在
`.gitignore` 中，不会提交密钥：

```powershell
Copy-Item .env.example .env
```

项目支持 OpenAI 兼容接口，最少需要：

```dotenv
API_BASE_URL=https://api.openai.com/v1
API_KEY=你的密钥
API_MODEL=gpt-4.1-mini
```

默认 Embedding 为开源中文模型 `BAAI/bge-small-zh-v1.5`，首次入库时下载到
`models/`。如果机器有 GPU，可把 `rag_core.py` 中的 `device` 从 `cpu` 改成
`cuda`；其余代码无需变化。

## 1. 文档处理

现有固定格式 Markdown 放在 `markdown/`，然后运行：

```powershell
python ingest.py
```

也可以索引另一个目录或单个 Word/PDF：

```powershell
python ingest.py --input test_data
```

所有处理结果都写入 `data/`。入库不会调用大模型 API；FAISS 与 Embedding 均在
本机运行。PDF 需要包含可提取文本，扫描版 PDF 应先 OCR。

## 2. 用户搜索与失效分析

普通查询：

```powershell
python query.py "某泵在热态小流量下振动超标，可能原因和处理建议是什么？"
```

带 Word、PDF 或 Markdown 输入：

```powershell
python query.py "根据附件做失效分析并提出建议" --file test_data/testdata1.docx
```

流程会先查询本地报告。当融合分数低于 `LOCAL_SCORE_THRESHOLD` 时必定联网；
高于阈值时，Agent 仍会检查是否缺少机理、标准或措施依据，并自行决定是否补查。

## 输出边界

结果会给出详细的证据、因果链、待验证项、优先级建议和引用。系统不会展示模型
私有的隐藏思维链，而是输出可由工程师复核的判断依据；这比不可核查的内部思考
更适合正式失效分析。Word/PDF 中的文字和表格文本可直接处理；纯图片或扫描件
当前不做视觉识别，需要先 OCR 后再输入。
