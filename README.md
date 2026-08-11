# Exchange-Docs-QA-Agent

基于 **LangChain + Chroma + RAG + Agent + Memory + Large Language Model（LLM）** 构建的交易所文档知识库智能问答助手。

本项目面向**交易所相关文档知识库**，通过 **RAG（Retrieval Augmented Generation，检索增强生成）** 将大语言模型与交易所文档结合，提高专业问题回答的准确性并减少模型幻觉。

在基础 RAG 问答的基础上，项目进一步加入了 **Agent、Tools 和 Memory**，实现智能工具调用以及多轮对话能力。

项目当前处于持续迭代阶段，目标是从基础的 RAG 问答系统逐步升级为具备**交易所知识库检索、Agent 智能决策、工具调用、多轮对话和 Web 交互能力的 AI Agent 应用**。

---

## 项目介绍

传统大语言模型无法直接了解特定交易所文档中的专业内容，例如交易规则、业务规定、产品规则及其他文档信息。

如果直接依赖 LLM 本身的知识进行回答，可能出现：

- 文档内容缺失
- 专业信息不准确
- 模型幻觉
- 无法根据最新文档回答问题

因此，本项目使用 RAG 将交易所文档知识库与 LLM 结合。

基础 RAG 流程：

```text
用户问题
    ↓
问题向量化 Embedding
    ↓
Chroma 向量数据库检索
    ↓
召回相关文档片段
    ↓
结合 Prompt 构造上下文
    ↓
LLM 生成最终答案
```

加入 Agent 和 Memory 后，整体流程进一步扩展为：

```text
用户问题
    ↓
Memory 读取历史对话
    ↓
Agent 分析用户需求
    ↓
选择并调用 Tool
    ↓
RAG Tool
    ↓
Chroma 检索交易所相关文档
    ↓
LLM 根据检索结果生成答案
    ↓
Memory 保存本轮对话
    ↓
返回最终回答
```

---

## 核心功能

### 已实现

- ✅ PDF 文档加载
- ✅ 文档文本切分
- ✅ Embedding 向量化
- ✅ Chroma 向量数据库
- ✅ 相似度检索
- ✅ RAG 问答
- ✅ DeepSeek / OpenAI 兼容 LLM 调用
- ✅ Streamlit Web 问答界面
- ✅ Agent 模块
- ✅ Tools 工具模块
- ✅ 将 RAG 能力封装为 Agent 可调用的 Tool
- ✅ Memory 对话记忆
- ✅ 用户聊天记录保存
- ✅ 历史消息管理
- ✅ 多轮连续对话

### 后续计划

- ⏳ 增加更多 Agent Tools
- ⏳ 检索结果来源引用
- ⏳ RAG 检索效果优化
- ⏳ Agent Prompt 优化
- ⏳ Tool Description 优化
- ⏳ 用户反馈机制
- ⏳ 项目部署上线

---

## 技术栈

| 技术 | 用途 |
|---|---|
| Python | 项目主要开发语言 |
| LangChain | LLM、RAG、Agent 应用开发框架 |
| LangChain-Chroma | LangChain 与 Chroma 集成 |
| Chroma | 向量数据库 |
| OpenAI Embeddings | 文本向量化 |
| DeepSeek | LLM 推理与文本生成 |
| Streamlit | Web 交互界面 |
| python-dotenv | 环境变量管理 |
| PDF Loader | 交易所 PDF 文档加载 |
| Text Splitter | 文档文本切分 |

---

## 项目结构

```text
Exchange-Docs-QA-Agent/
│
├── agent/
│   ├── __init__.py
│   ├── agent.py
│   │   └── Agent 核心逻辑
│   └── tools.py
│       └── Agent 可调用工具
│
├── data/
│   └── 交易所知识库文档(PDF)
│
├── vector_db/
│   └── Chroma 向量数据库
│
├── ingest.py
│   └── 文档加载、文本切分、Embedding、向量数据库构建
│
├── rag.py
│   └── RAG 检索、问答及 Memory 相关逻辑
│
├── app.py
│   └── Streamlit Web 应用入口
│
├── .env
│   └── API Key、模型及路径等环境变量
│
├── .gitignore
├── requirements.txt
└── README.md
```

> 当前 Memory 功能直接实现于 `rag.py` 模块中，因此项目没有单独的 `memory/` 目录。

---

# 系统架构

当前项目可以理解为以下几个核心模块：

```text
                    ┌─────────────────────┐
                    │      Streamlit      │
                    │        app.py       │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │       Agent         │
                    │     agent.py        │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │       Tools         │
                    │     tools.py        │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │     RAG + Memory    │
                    │       rag.py        │
                    └──────────┬──────────┘
                               │
                     ┌─────────┴─────────┐
                     ↓                   ↓
              ┌─────────────┐     ┌─────────────┐
              │   Chroma    │     │  Chat LLM   │
              │ vector_db/  │     │ DeepSeek    │
              └──────┬──────┘     └─────────────┘
                     │
                     ↓
              ┌─────────────┐
              │ 交易所文档知识库 │
              │    data/     │
              └─────────────┘
```

---

# RAG 模块

RAG 是本项目的核心知识库能力。

## RAG 工作流程

```text
交易所 PDF 文档
      ↓
Document Loader
      ↓
文本切分
      ↓
Embedding
      ↓
Chroma Vector Database
      ↓
用户问题
      ↓
Retriever
      ↓
召回相关文档 Chunk
      ↓
Prompt + Context
      ↓
LLM
      ↓
最终回答
```

---

## `ingest.py`

`ingest.py` 负责构建交易所文档知识库：

1. 加载交易所 PDF 文档
2. 将文档转换为 LangChain Document
3. 对文档进行 Chunk 切分
4. 使用 Embedding 模型生成向量
5. 将向量写入 Chroma

运行：

```bash
python ingest.py
```

完成后生成本地向量数据库。

如果新增或修改了知识库文档，需要重新执行知识库构建流程。

---

## `rag.py`

`rag.py` 是当前项目的核心模块之一，同时承担 **RAG 和 Memory** 相关逻辑。

主要负责：

- 加载 Chroma 向量数据库
- 创建 Retriever
- 根据用户问题检索相关交易所文档
- 构造 Prompt
- 调用 LLM
- 结合历史聊天记录进行上下文理解
- 保存和管理用户聊天记录
- 返回最终回答

因此目前的模块关系是：

```text
rag.py
├── RAG
│   ├── Retriever
│   ├── Chroma
│   └── LLM
│
└── Memory
    ├── 历史消息
    └── 多轮上下文
```

后续如果项目进一步复杂化，可以再考虑将 RAG 和 Memory 拆分成独立模块。

目前保持在 `rag.py` 中有利于项目结构简洁，也符合当前项目规模。

---

# Agent 模块

在基础 RAG 之上，项目新增 Agent 能力。

## 传统 RAG

传统 RAG 的流程比较固定：

```text
用户问题
    ↓
Retriever
    ↓
检索知识库
    ↓
LLM
    ↓
答案
```

## Agent + RAG

加入 Agent 后：

```text
用户问题
    ↓
Agent
    ↓
分析用户需求
    ↓
决定是否调用 Tool
    ↓
调用 RAG Tool
    ↓
交易所文档知识库检索
    ↓
Agent 根据 Tool 返回结果生成答案
```

因此：

> **Agent 负责决策和工具调用，RAG 负责知识库检索。**

RAG 并没有被 Agent 替代，而是成为 Agent 可以调用的一项能力。

---

## `agent/agent.py`

负责 Agent 的核心逻辑：

- 创建 LLM
- 创建 Agent
- 注册 Tools
- 配置 Agent System Prompt
- 接收用户问题
- 根据用户需求选择 Tool
- 调用 Tool
- 组织最终回答

核心关系：

```text
用户
 ↓
Agent
 ↓
Tool
 ↓
RAG
 ↓
交易所知识库
```

---

## `agent/tools.py`

`tools.py` 负责定义 Agent 可以使用的工具。

当前最重要的工具是交易所知识库查询 Tool。

其调用关系：

```text
Agent
 ↓
RAG Tool
 ↓
rag.py
 ↓
Retriever
 ↓
Chroma
 ↓
交易所文档
```

这种设计方便后续扩展更多工具，例如：

```text
Tools
├── exchange_qa
│   └── 查询交易所知识库
│
├── calculator
│   └── 数学计算
│
├── web_search
│   └── 网络搜索
│
└── ...
```

---

# Memory 模块

当前项目已经增加 Memory 能力，但 Memory **没有单独创建目录**，而是直接实现于：

```text
rag.py
```

Memory 用于保存用户与 AI 的历史聊天记录，使系统具备多轮对话能力。

---

## 单轮问答

没有历史上下文时：

```text
用户：某项交易规则是什么？
AI：回答相关规则。

用户：它适用于哪些情况？
AI：可能无法确定“它”具体指什么。
```

---

## 多轮对话

加入 Memory 后：

```text
用户：某项交易规则是什么？
AI：回答相关规则。

用户：它适用于哪些情况？
AI：结合上一轮对话理解“它”指代的交易规则，并继续回答。
```

因此 Memory 的作用是：

> **保存历史消息，为 Agent / RAG 提供连续对话所需的上下文。**

---

## Memory 工作流程

```text
用户当前问题
      ↓
读取历史聊天记录
      ↓
历史消息 + 当前问题
      ↓
Agent
      ↓
Tool / RAG
      ↓
LLM
      ↓
生成回答
      ↓
保存本轮对话
```

当前架构：

```text
app.py
  ↓
agent.py
  ↓
tools.py
  ↓
rag.py
  ├── RAG
  └── Memory
```

---

# Streamlit Web 界面

项目使用 **Streamlit** 构建 Web 问答界面。

推荐使用：

```bash
streamlit run app.py
```

启动应用。

当前 Web 界面负责：

- 用户问题输入
- 用户消息展示
- AI 回答展示
- Markdown 内容展示
- 多轮聊天交互
- 调用 Agent
- 当前会话历史管理

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

## 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 4. 配置 `.env`

在项目根目录创建：

```text
.env
```

根据实际环境配置，例如：

```env
OPENAI_API_KEY=your_api_key
DB_DIR=D:\your\path\Exchange-Docs-QA-Agent\vector_db
```

如果使用 DeepSeek 的 OpenAI 兼容 API，需要在代码中配置对应的 `base_url`：

```text
https://api.deepseek.com/v1
```

> `.env` 中可能包含 API Key 等敏感信息，因此不要提交到 GitHub。

---

# 构建知识库

将交易所相关 PDF 文档放入：

```text
data/
```

然后执行：

```bash
python ingest.py
```

程序完成：

```text
PDF
 ↓
Document Loader
 ↓
Text Splitter
 ↓
Embedding
 ↓
Chroma
 ↓
vector_db/
```

如果修改或增加交易所知识库文档，需要重新构建向量数据库。

---

# 启动应用

推荐：

```bash
streamlit run app.py
```

启动后在浏览器中使用交易所文档知识库问答助手。

不要直接使用：

```bash
python app.py
```

因为 `app.py` 是 Streamlit 应用入口。

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

已完成：

- [x] 交易所 PDF 知识库
- [x] 文档加载
- [x] 文本切分
- [x] Embedding
- [x] Chroma 向量数据库
- [x] 相似度检索
- [x] LLM 问答

---

## Version 2.0 —— Web 问答界面

已完成：

- [x] Streamlit 聊天页面
- [x] 用户输入框
- [x] 用户 / AI 消息展示
- [x] Markdown 回答展示
- [x] 基础异常处理

---

## Version 3.0 —— 多轮对话 Memory

已完成：

- [x] Memory 能力
- [x] 聊天记录保存
- [x] 历史消息管理
- [x] 上下文连续问答
- [x] Memory 与 RAG 结合

---

## Version 4.0 —— Agent

当前阶段：

- [x] Agent 模块
- [x] Tools 模块
- [x] RAG Tool
- [x] Agent 调用 RAG
- [x] Agent 与 Memory 结合
- [ ] 增加更多 Tools
- [ ] 优化 Agent System Prompt
- [ ] 优化 Tool Description
- [ ] 增加 Tool 调用过程展示

---

## Version 5.0 —— RAG 与 Agent 优化

计划实现：

- [ ] 检索结果来源引用
- [ ] RAG Top-K 优化
- [ ] Chunk Size / Chunk Overlap 优化
- [ ] Rerank
- [ ] Query Rewrite
- [ ] 混合检索
- [ ] 检索结果质量评估
- [ ] Agent Tool 选择优化
- [ ] Hallucination 控制
- [ ] 用户反馈机制

---

## Version 6.0 —— 项目部署

计划实现：

- [ ] Docker
- [ ] Linux 部署
- [ ] 云服务器部署
- [ ] API 服务化
- [ ] 日志系统
- [ ] 用户会话管理
- [ ] 项目性能优化
- [ ] 项目正式上线

---

# 项目学习目标

本项目同时作为 AI Agent 应用开发实践项目，用于学习现代 LLM 应用开发中的核心技术。

通过项目逐步掌握：

```text
LLM API
   ↓
Prompt Engineering
   ↓
Embedding
   ↓
Vector Database
   ↓
RAG
   ↓
Memory
   ↓
Tool Calling
   ↓
Agent
   ↓
RAG + Agent
   ↓
Deployment
```

最终将项目升级为一个具备：

- 交易所专业知识库检索
- RAG 问答
- Agent 智能决策
- Tool Calling
- 多轮对话 Memory
- Web 交互
- 来源引用
- 用户反馈
- 可部署运行

能力的完整 AI Agent 应用。

---

# 当前项目架构总结

目前项目最核心的关系是：

```text
                    User
                     │
                     ↓
                Streamlit
                  app.py
                     │
                     ↓
                  Agent
                agent.py
                     │
                     ↓
                  Tools
                tools.py
                     │
                     ↓
               RAG + Memory
                  rag.py
                     │
            ┌────────┴────────┐
            ↓                 ↓
        Retriever          Memory
            │                 │
            ↓                 │
         Chroma              │
       vector_db/             │
            │                 │
            ↓                 │
      交易所文档知识库 ─────────┘
          data/
```

各模块职责：

```text
app.py
    ↓
负责 Web 用户交互

agent/agent.py
    ↓
负责 Agent 决策与工具调用

agent/tools.py
    ↓
负责向 Agent 提供可调用工具

rag.py
    ↓
负责 RAG 检索、LLM 问答以及当前项目的 Memory 逻辑

ingest.py
    ↓
负责构建交易所文档知识库

vector_db/
    ↓
保存 Chroma 向量数据

data/
    ↓
保存交易所原始知识库文档
```

---

## License

This project is for learning and demonstration purposes.
