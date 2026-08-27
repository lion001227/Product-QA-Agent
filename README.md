# Exchange-Docs-QA-Agent

基于 **LangChain + LangGraph + Chroma + RAG + Agent + Memory + DeepSeek LLM** 构建的交易所文档知识库智能问答助手。

本项目面向**交易所相关文档知识库**，通过 **RAG（Retrieval Augmented Generation，检索增强生成）** 将大语言模型与交易所文档结合，提高专业问题回答的准确性并减少模型幻觉。

在基础 RAG 问答之上，项目进一步加入了 **Agent、Tools 和 Memory**：Agent 不仅能检索本地知识库，还能实时调用上交所 / 深交所官网接口查询最新公告，并支持多轮对话与"回答来源溯源"。

---

## 项目介绍

传统大语言模型无法直接了解特定交易所文档中的专业内容，例如交易规则、业务规定、产品规则及其他文档信息，也无法获取交易所官网的实时公告。

如果直接依赖 LLM 本身的知识进行回答，可能出现：

- 文档内容缺失
- 专业信息不准确
- 模型幻觉
- 无法回答最新公告 / 最新文档相关问题

因此，本项目使用 RAG 将交易所文档知识库与 LLM 结合，并通过 Agent + Tools 让模型具备**按需检索本地知识库**和**按需查询实时公告**两种能力。

基础 RAG 流程：

```text
用户问题
    ↓
问题向量化 Embedding
    ↓
Chroma 向量数据库检索（MMR，兼顾相关性与多样性）
    ↓
召回相关文档片段（去重）
    ↓
结合 Prompt 构造上下文
    ↓
LLM 生成最终答案
```

加入 Agent、Tools 和 Memory 后，整体流程扩展为：

```text
用户问题
    ↓
Agent（携带历史对话，理解上下文指代）
    ↓
判断问题类型，选择 Tool
    ├── 文档内容问题 → profile_search → RAG 检索 → LLM 生成答案
    ├── 溯源类问题   → return_document_sources → 返回原文/文件名/页码
    ├── 上交所公告   → sse_latest_announcements → 实时抓取官网数据
    └── 深交所公告   → szse_latest_announcements → 实时抓取官网数据
    ↓
LangGraph Checkpointer（按 thread_id）保存本轮对话
    ↓
返回最终回答
```

---

## 核心功能

### 已实现

- ✅ 多格式文档加载（PDF / TXT / CSV / DOCX）
- ✅ 文档文本切分（Chunk Size 500，Overlap 100）
- ✅ 本地 BGE 模型 Embedding 向量化
- ✅ Chroma 向量数据库持久化
- ✅ MMR 相似度检索（兼顾相关性与多样性）+ 检索结果去重
- ✅ RAG 问答（DeepSeek / OpenAI 兼容 LLM）
- ✅ **回答来源溯源**（文件名、页码、原文片段，逐字保留输出）
- ✅ Streamlit Web 问答界面
- ✅ Agent 模块（`create_agent`）+ 明确的工具调用规则（System Prompt）
- ✅ Tools 工具模块：
  - RAG 知识库检索工具
  - 文档来源溯源工具
  - **上交所最新公告实时查询**
  - **深交所最新公告实时查询**
- ✅ 基于 LangGraph `InMemorySaver` 的多轮对话记忆（按会话 `thread_id` 区分）
- ✅ Streamlit 会话级聊天记录展示与"清空对话"功能

### 后续计划

- ⏳ 增加更多 Agent Tools（如财报查询、计算器等）
- ⏳ RAG 检索效果优化（Rerank、Query Rewrite、混合检索）
- ⏳ 持久化 Memory（当前为内存态，重启后丢失）
- ⏳ 用户反馈机制
- ⏳ 项目部署上线（Docker / 云服务器）

---

## 技术栈

| 技术 | 用途 |
|---|---|
| Python | 项目主要开发语言 |
| LangChain | LLM、RAG、Agent 应用开发框架 |
| LangChain-Chroma | LangChain 与 Chroma 集成 |
| LangChain-HuggingFace / langchain_community.embeddings | BGE 本地模型接入 |
| LangGraph | Agent 运行时与多轮对话 Memory（`InMemorySaver`） |
| Chroma | 向量数据库 |
| BGE（本地模型） | 文本向量化 Embedding |
| DeepSeek（OpenAI 兼容 API） | LLM 推理与文本生成 |
| Streamlit | Web 交互界面 |
| Requests | 调用上交所 / 深交所官网公告接口 |
| python-dotenv | 环境变量管理 |
| PyPDFLoader / Docx2txtLoader / TextLoader / CSVLoader | 多格式文档加载 |
| RecursiveCharacterTextSplitter | 文档文本切分 |

---

## 项目结构

```text
Exchange-Docs-QA-Agent/
│
├── agent/
│   ├── __init__.py
│   ├── agent.py
│   │   └── Agent 核心逻辑：create_agent、System Prompt、ask_agent()
│   └── tools.py
│       └── Agent 可调用工具：profile_search / return_document_sources /
│           sse_latest_announcements / szse_latest_announcements
│
├── data/
│   └── 交易所知识库文档（PDF / TXT / CSV / DOCX）
│
├── vector_db/
│   └── Chroma 向量数据库（由 invest.py 生成，勿手动修改）
│
├── invest.py
│   └── 文档加载、文本切分、Embedding、向量数据库构建
│
├── rag.py
│   └── RAG 检索、Prompt 构造、LLM 问答、来源溯源逻辑
│       （注意：需放在项目根目录，agent/tools.py 会以
│        `from rag import rag_qa` 的绝对导入方式引用它）
│
├── app.py
│   └── Streamlit Web 应用入口，调用 agent.agent.ask_agent()
│
├── .env
│   └── API Key、模型路径等环境变量
│
├── .gitignore
├── requirements.txt
└── README.md
```

> ⚠️ **模块位置提醒**：`rag.py` 必须放在项目根目录（与 `app.py`、`invest.py` 同级），而不是放进 `agent/` 包内，否则 `agent/tools.py` 中的 `from rag import rag_qa` / `from rag import search_document_sources` 会导入失败。

> Memory 当前通过 LangGraph 的 `InMemorySaver` 实现，直接集成在 `agent/agent.py` 中，按 `thread_id` 区分不同会话，因此项目没有单独的 `memory/` 目录。

---

## 系统架构

```text
                    ┌─────────────────────┐
                    │      Streamlit       │
                    │        app.py        │
                    └──────────┬───────────┘
                               │ ask_agent(question, thread_id)
                               ↓
                    ┌─────────────────────┐
                    │        Agent         │
                    │   agent/agent.py     │
                    │  (create_agent +     │
                    │   InMemorySaver)     │
                    └──────────┬───────────┘
                               │ 按需选择工具调用
                               ↓
                    ┌─────────────────────┐
                    │        Tools         │
                    │   agent/tools.py     │
                    └──────────┬───────────┘
              ┌────────────────┼──────────────────────┐
              ↓                ↓                       ↓
      ┌───────────────┐ ┌──────────────┐      ┌──────────────────┐
      │ profile_search │ │ return_docume│      │ sse / szse       │
      │  return_docume-│ │ nt_sources   │      │ latest_          │
      │  nt_sources    │ │              │      │ announcements    │
      └───────┬────────┘ └──────┬───────┘      └─────────┬────────┘
              │                 │                          │
              ↓                 ↓                          ↓
      ┌───────────────────────────────┐          ┌──────────────────┐
      │           rag.py              │          │ 上交所 / 深交所    │
      │  Retriever(MMR) + LLM         │          │  官网公告接口      │
      └───────────────┬───────────────┘          │（实时 HTTP 请求）│
                       │                          └──────────────────┘
                       ↓
              ┌─────────────────┐
              │     Chroma      │
              │   vector_db/    │
              └────────┬────────┘
                       │
                       ↓
              ┌─────────────────┐
              │ 交易所文档知识库  │
              │      data/       │
              │ （由 invest.py   │
              │   构建索引）      │
              └─────────────────┘
```

---

# RAG 模块

## `invest.py`（文档向量化）

负责构建交易所文档知识库：

1. 遍历 `data/` 目录，按后缀名（`.pdf` / `.txt` / `.csv` / `.docx`）自动选择对应的 Loader
2. 使用 `RecursiveCharacterTextSplitter` 切分文本（`chunk_size=500`，`chunk_overlap=100`）
3. 使用本地 BGE 模型（`langchain_huggingface.HuggingFaceEmbeddings`，开启 `normalize_embeddings=True`）生成向量
4. 写入 Chroma，持久化到 `<项目根目录>/vector_db`

运行：

```bash
python invest.py
```

如果新增或修改了知识库文档，需要重新执行该脚本以重建向量库。

---

## `rag.py`（检索与问答）

`rag.py` 是知识库检索与问答的核心模块，主要负责：

- 加载已持久化的 Chroma 向量库（`langchain_community.embeddings.HuggingFaceBgeEmbeddings` 加载同一本地 BGE 模型）
- 基于 **MMR（最大边际相关性）** 创建 Retriever：`k=5`、`fetch_k=20`、`lambda_mult=0.5`，兼顾相关性与结果多样性
- `dedup_context()`：对检索结果按内容前 80 字符去重，避免重复片段污染上下文
- `format_docs()` + `ChatPromptTemplate`：构造带金融专家人设的 Prompt（要求完整读取跨页表格、综合多文档块信息、无答案时明确说不知道）
- 调用 DeepSeek（`deepseek-chat`，`base_url=https://api.deepseek.com/v1`）生成回答
- `rag_qa(question)`：对外提供的问答入口，返回纯文本答案
- `search_document_sources(question, k=5)`：**来源溯源**能力，返回文件名、页码（PyPDFLoader 页码从 0 开始，已 +1 修正）及原文片段（截断至 300 字），并按"文件+页码+片段前缀"去重

> ⚠️ **已知不一致（建议后续统一）**：
> `invest.py` 使用 `langchain_huggingface.HuggingFaceEmbeddings` 且开启了 `normalize_embeddings=True`；
> `rag.py` 使用的是 `langchain_community.embeddings.HuggingFaceBgeEmbeddings`（该类在 LangChain 中已标记为过时用法），且未显式设置 `normalize_embeddings`。
> 两处虽加载同一个 `MODEL_DIR` 模型，但构建索引与检索查询时的 Embedding 实现/参数不完全一致，理论上可能影响检索效果，建议后续统一为同一套 Embedding 封装与参数。

---

# Agent 模块

## `agent/agent.py`

负责 Agent 的核心逻辑：

- 创建 DeepSeek LLM（`ChatOpenAI`，`temperature=0.0`）
- 注册工具列表：`profile_search`、`return_document_sources`、`sse_latest_announcements`、`szse_latest_announcements`
- 使用 `langgraph.checkpoint.memory.InMemorySaver` 作为 Agent 的 Checkpointer，实现多轮对话记忆
- 通过 `create_agent()` 组装 Agent，并配置详细的 **System Prompt 规则**，包括：
  1. 结合历史对话，将有指代关系的问题改写为完整问题后再调用 `profile_search`
  2. 用户询问"依据/来源/原文/第几页"时调用 `return_document_sources`，且**必须逐字原样输出**工具返回内容，不得改写、不得增删字段
  3. 用户询问上交所 / 深交所最新公告时，根据用户明确指定的交易所调用对应工具，**不得混用两个交易所的结果**，且必须保留公告发布日期和官网链接，不得编造公告内容
  4. 未检索到资料时不得编造信息
- `ask_agent(question, thread_id)`：对外问答入口，相同 `thread_id` 代表同一段对话，LangGraph 会自动读取并延续历史上下文

---

## `agent/tools.py`

定义 Agent 可调用的四个工具：

| 工具 | 功能 | 触发场景 |
|---|---|---|
| `profile_search` | 调用 `rag.py` 的 `rag_qa`，检索交易所文档回答问题 | 用户咨询交易所文档相关内容 |
| `return_document_sources` | 调用 `rag.py` 的 `search_document_sources`，返回文件名/页码/原文片段 | 用户询问"来源/依据/原文/第几页" |
| `sse_latest_announcements` | 实时请求上交所官网公告接口，支持按股票代码 / 关键词本地筛选，最多返回 10 条 | 用户询问上交所最新公告 |
| `szse_latest_announcements` | 实时请求深交所官网公告接口，支持按关键词筛选，最多返回 10 条 | 用户询问深交所最新公告 |

其中上交所 / 深交所公告工具直接请求官网接口（非知识库数据），对返回的原始字段做了清洗（URL 补全为 `https`、时间戳格式化、按标题/公司代码/内容片段去重等）。

调用关系：

```text
Agent
 ├── profile_search / return_document_sources
 │        ↓
 │      rag.py → Retriever(MMR) → Chroma → 交易所文档
 │
 └── sse_latest_announcements / szse_latest_announcements
          ↓
        上交所 / 深交所官网 HTTP 接口（实时数据，不经过向量库）
```

---

# Memory 模块

Memory 目前通过 **LangGraph 的 `InMemorySaver`** 实现，直接集成在 `agent/agent.py` 的 `create_agent()` 中，因此项目没有单独的 `memory/` 目录。

- 每个 Streamlit 会话在 `st.session_state` 中生成一个唯一的 `thread_id`（`uuid.uuid4()`）
- 相同 `thread_id` 的多次 `ask_agent()` 调用会被 LangGraph 自动识别为同一段对话，读取历史消息并在本轮结束后写回
- 点击"清空对话"会重置聊天记录并生成新的 `thread_id`，开启全新会话
- ⚠️ 当前为**进程内内存存储**，应用重启或多进程部署时历史记录会丢失，如需持久化需替换为其他 Checkpointer（如数据库、Redis 等）

单轮 vs 多轮示例：

```text
【无 Memory】
用户：某项交易规则是什么？          → AI：回答相关规则
用户：它适用于哪些情况？            → AI：可能无法确定"它"指什么

【有 Memory，相同 thread_id】
用户：某项交易规则是什么？          → AI：回答相关规则
用户：它适用于哪些情况？            → AI：结合上一轮对话理解"它"，继续回答
```

---

# Streamlit Web 界面

`app.py` 提供基于 Streamlit 的问答界面，负责：

- 用户问题输入（`st.chat_input`）
- 用户 / AI 消息展示（`st.chat_message`）
- 每个会话维护独立的聊天记录与 `thread_id`
- 调用 `agent.agent.ask_agent()` 获取回答，并做基础异常捕获展示
- 侧边栏提供助手说明及"清空对话"按钮（重置消息列表和 `thread_id`）

启动：

```bash
streamlit run app.py
```

不要直接使用 `python app.py`，因为 `app.py` 是 Streamlit 应用入口，需要通过 `streamlit run` 启动。

---

# 环境配置

## 1. 创建虚拟环境

```bash
python -m venv .venv
```

## 2. 激活虚拟环境

Windows：

```bash
.venv\Scripts\activate
```

macOS / Linux：

```bash
source .venv/bin/activate
```

## 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 4. 配置 `.env`

在项目根目录创建 `.env` 文件，包含：

```env
MODEL_DIR=/path/to/your/local/bge-model
OPENAI_API_KEY=your_deepseek_api_key
```

说明：

- `MODEL_DIR`：BGE 本地嵌入模型所在路径（`invest.py` 和 `rag.py` 均需要）
- `OPENAI_API_KEY`：实际用于访问 **DeepSeek** 服务（因为 DeepSeek 提供 OpenAI 兼容 API，`base_url` 已在代码中写死为 `https://api.deepseek.com/v1`）

> `.env` 中可能包含 API Key 等敏感信息，请勿提交到 GitHub。

---

# 构建知识库

将交易所相关文档（PDF / TXT / CSV / DOCX）放入：

```text
data/
```

然后执行：

```bash
python invest.py
```

程序完成：

```text
data/ 中的文档
 ↓
按后缀选择 Loader（PDF / TXT / CSV / DOCX）
 ↓
RecursiveCharacterTextSplitter 切分
 ↓
本地 BGE 模型 Embedding
 ↓
Chroma 持久化
 ↓
vector_db/
```

如果修改或增加知识库文档，需要重新执行 `invest.py` 以重建向量数据库。

---

# 启动应用

```bash
streamlit run app.py
```

启动后在浏览器中使用交易所文档知识库问答助手，可直接提问文档相关内容、追问来源依据，或查询上交所 / 深交所最新公告。

---

# Git 与敏感信息

建议 `.gitignore` 至少包含：

```gitignore
# Virtual Environment
.venv/

# Environment Variables
.env

# Python
__pycache__/
*.py[cod]

# IDE
.idea/

# Chroma Vector Database
vector_db/

# OS
.DS_Store
Thumbs.db
```

通常不建议上传：

```text
.env
vector_db/
.venv/
```

原因：

- `.env` 可能包含 API Key
- `vector_db/` 是本地生成的向量数据库
- `.venv/` 是本地 Python 虚拟环境

---

# 项目版本迭代

## Version 1.0 —— 基础 RAG

- [x] 交易所多格式文档知识库（PDF / TXT / CSV / DOCX）
- [x] 文档加载与切分
- [x] Embedding 向量化
- [x] Chroma 向量数据库
- [x] MMR 相似度检索
- [x] LLM 问答（DeepSeek）

## Version 2.0 —— Web 问答界面

- [x] Streamlit 聊天页面
- [x] 用户输入框
- [x] 用户 / AI 消息展示
- [x] 基础异常处理

## Version 3.0 —— Agent 与 Tools

- [x] Agent 模块（`create_agent`）
- [x] Tools 模块
- [x] RAG 检索工具（`profile_search`）
- [x] 回答来源溯源工具（`return_document_sources`）
- [x] 上交所 / 深交所实时公告查询工具
- [x] Agent System Prompt 规则化（指代改写、来源原样输出、公告不混用）

## Version 4.0 —— 多轮对话 Memory

- [x] 基于 LangGraph `InMemorySaver` 的对话记忆
- [x] 按 `thread_id` 区分会话
- [x] Streamlit 会话级历史管理与清空功能
- [ ] Memory 持久化（当前为进程内内存，重启即丢失）

## Version 5.0 —— RAG 与 Agent 优化（计划中）

- [ ] Rerank
- [ ] Query Rewrite
- [ ] 混合检索
- [ ] 统一 Embedding 实现与参数（解决 `invest.py` 与 `rag.py` 的不一致问题）
- [ ] 检索结果质量评估
- [ ] 用户反馈机制

## Version 6.0 —— 项目部署（计划中）

- [ ] Docker
- [ ] Linux / 云服务器部署
- [ ] API 服务化
- [ ] 日志系统
- [ ] 持久化用户会话管理
- [ ] 项目正式上线

---

# 项目学习目标

本项目同时作为 AI Agent 应用开发实践项目，用于学习现代 LLM 应用开发中的核心技术：

```text
LLM API → Prompt Engineering → Embedding → Vector Database
   → RAG → 来源溯源 → Tool Calling（含实时外部接口）
   → Agent → Memory（LangGraph）→ Web 交互 → Deployment
```

---

## License

This project is for learning and demonstration purposes.
