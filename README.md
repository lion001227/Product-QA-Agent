# 交易所文档 QA Agent

一个面向交易所文档的智能问答应用。项目结合本地 BGE 向量检索、DeepSeek（OpenAI 兼容接口）、LangGraph 和 Streamlit，支持文档问答、答案来源追溯、上交所/深交所实时公告查询，跨重启保留的多轮对话，以及基于测试集的 Agent 评估。

## 功能

- 读取 `data/` 中的 PDF、TXT、CSV、DOCX 文档，构建 Chroma 向量库。
- 使用 MMR 检索，并对相近文本块去重。
- 通过 LangGraph RAG 图执行“检索 → 资料充分性判断 → 查询改写（最多三次检索）→ 回答生成”。
- 由 Supervisor 根据问题路由到知识库问答、来源追溯或交易所公告专职 Agent。
- 返回来源文件名、页码和原文片段。
- 使用 SQLite checkpoint 按 `thread_id` 保存会话；应用重启后对话状态仍可恢复。
- Streamlit 聊天界面以流式方式展示回答，并支持清空当前会话。
- 提供评估脚本：对测试问题统计工具选择正确率、回答正确率、检索相关度、延迟、token 用量和失败率。

## 架构

```text
Streamlit（app.py）
        │ stream_agent(question, thread_id)
        ▼
LangGraph 多 Agent 图（agent/agent.py）── SQLite checkpoint
        │
        ▼
Supervisor（仅负责路由）
        ├── rag Agent ─────────── profile_search ──────────────┐
        ├── source Agent ──────── return_document_sources ─────┼── RAG 图（rag/rag.py）── Chroma（vector_db/）
        └── announcement Agent ── sse_latest_announcements ────┼── 上交所官网实时接口
                                  szse_latest_announcements ────┴── 深交所官网实时接口
```

评估入口 `evaluation/evaluate.py` 直接调用编译后的 Agent 图，并对 rag/source 类问题通过 `retrieve_for_evaluation` 取出检索文档用于相关性打分。

## 项目结构

```text
Product-QA-Agent/
├── agent/
│   ├── agent.py          # Supervisor 多 Agent 图、专属工具循环、SQLite 会话记忆与流式输出
│   └── tools.py          # RAG、来源追溯、上交所和深交所公告工具
├── rag/
│   ├── ingest.py         # 文档加载、切分和向量库构建脚本
│   └── rag.py            # RAG StateGraph、检索、改写、回答、来源追溯与评估检索接口
├── evaluation/
│   ├── dataset.py        # 评估测试集
│   ├── evaluate.py       # 跑测试、汇总指标并写出 results.json / summary.json
│   ├── metrics.py        # 工具正确率、延迟、token、LLM 裁判打分等
│   └── results.py        # 读取并打印已保存的评估结果
├── data/                 # 待索引的知识库文档（本地目录，不提交）
├── models/               # 本地 BGE 模型目录（不提交）
├── vector_db/            # Chroma 持久化数据（由 rag/ingest.py 生成，不提交）
├── app.py                # Streamlit 应用入口
├── requirements.txt
└── .env
```

运行时会在项目根目录创建 `langgraph_checkpoints.sqlite` 及其 SQLite 辅助文件，用于保存聊天状态。不要手动删除它们，除非确认要清除所有已持久化的会话。

评估脚本会在当前工作目录写出 `results.json` 和 `summary.json`（已被 `.gitignore` 排除）。

## 环境要求

- Python 3.10 或更高版本
- 可访问 DeepSeek API
- 本地可用的 BGE embedding 模型

## 安装与配置

创建并激活虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
pip install -r requirements.txt
```

在项目根目录创建 `.env`：

```env
OPENAI_API_KEY=your_deepseek_api_key
MODEL_DIR=D:/path/to/bge-small-zh-v1.5
```

`OPENAI_API_KEY` 用于访问 DeepSeek；程序已在代码中指定 `https://api.deepseek.com/v1` 和 `deepseek-chat`。`MODEL_DIR` 必须指向本地 BGE 模型目录。Windows 路径建议使用正斜杠。


## 构建知识库

将 PDF、TXT、CSV 或 DOCX 文件放入项目根目录的 `data/`。`rag/ingest.py` 按当前工作目录读取 `../data`，因此需要在 `rag/` 目录下执行：

```bash
cd rag
python ingest.py
```

脚本使用 `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)` 切分文档，以本地 BGE 模型生成向量，并将 Chroma 数据写入项目根目录的 `vector_db/`。

新增或修改 `data/` 中的文件后，需要再次执行该命令更新向量库。

## 启动应用

在项目根目录执行：

```bash
streamlit run app.py
```

打开 Streamlit 显示的本地地址后即可提问。不要使用 `python app.py` 启动，因为它是 Streamlit 应用入口。

## 使用说明

- Supervisor 会根据当前问题路由到一个专职 Agent；它不会直接生成最终答案。
- 文档内容、业务规则或制度问题由 rag Agent 处理，并通过本地知识库检索回答。
- 询问“来源、依据、原文或第几页”由 source Agent 处理，返回相关文件、页码与原文片段。
- 明确询问上交所或深交所的最新/近期公告时，由 announcement Agent 调用对应官网接口；公告数据为实时数据，不来自本地向量库。
- 各专职 Agent 只有对应工具可用。模型发出工具调用时，LangGraph 执行工具后将结果交回同一个 Agent 生成答复；不需要工具时直接结束。
- 每个浏览器会话对应一个 `thread_id`。侧边栏的“清空对话”会生成新的会话 ID；旧会话仍保存在 SQLite 文件中。

## 评估

先完成知识库构建并配置好 `.env`，然后在**项目根目录**运行：

```bash
python -m evaluation.evaluate
```

脚本会遍历 `evaluation/dataset.py` 中的测试用例，调用 Agent 作答，并计算：

- 平均延迟
- 工具选择正确率
- 回答正确率（由 DeepSeek 按标准答案裁判）
- 平均 token 用量（输入 / 输出 / 合计）
- rag/source 问题的检索相关度
- 失败率

结果写入当前目录的 `results.json` 和 `summary.json`。可用下列命令再次查看：

```bash
python -m evaluation.results
```

评估会按用例创建独立的 `thread_id`（形如 `eval_0`），并写入 SQLite checkpoint。

## 依赖说明

`requirements.txt` 为运行当前代码所需的最低版本范围。其中 `modelscope` 仅供本地 `1.py` 下载模型使用；`evaluation/` 复用现有 LangChain / LangGraph / DeepSeek 依赖，无额外第三方包。若要获得可复现的部署环境，请在已验证可用的虚拟环境中执行 `pip freeze > requirements.lock.txt`，并在部署时使用该锁定文件。

## 注意事项

- `.env` 包含密钥，切勿提交到版本库。
- `data/`、`models/` 和 `vector_db/` 都是本地资产，当前被 `.gitignore` 排除。
- 应用启动时会加载本地 embedding 模型；首次加载可能较慢。
- SQLite checkpoint 文件包含对话内容，分享或清理项目前请按敏感数据处理。
- `results.json` / `summary.json` 可能包含问题、模型回答和检索原文，同样不要提交。

## License

This project is for learning and demonstration purposes.
