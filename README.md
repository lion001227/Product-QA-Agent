# Profile-QA-Agent

基于 **LangChain + Chroma + Large Language Model(LLM)** 构建的企业产品知识库智能问答助手。

本项目使用 **RAG（Retrieval Augmented Generation，检索增强生成）技术**，让大语言模型结合企业产品文档进行精准问答，减少模型幻觉，提高回答的准确性。

## 项目介绍

传统的大语言模型无法直接了解企业内部文档内容。

本项目通过 RAG 技术实现：

```
用户问题
    ↓
问题向量化 Embedding
    ↓
Chroma 向量数据库检索
    ↓
召回相关知识片段
    ↓
结合 Prompt 构造上下文
    ↓
LLM生成最终答案
```

## 技术栈

- Python
- LangChain
- LangChain-Chroma
- Chroma Vector Database
- OpenAI Embedding
- DeepSeek / OpenAI LLM
- dotenv 环境变量管理

## 项目结构

```
Product-QA-Agent

│
├── data/
│   └── 产品知识文档(PDF)
│
├── vector_db/
│   └── Chroma向量数据库
│
├── ingest.py
│   └── 文档加载、文本切片、Embedding生成
│
├── rag.py
│   └── RAG检索问答核心逻辑
│
├── app.py
│   └── 应用入口
│
├── .env
│   └── API Key配置
│
├── requirements.txt
│
└── README.md
```

## 核心功能

目前已实现：

- ✅ PDF文档加载
- ✅ 文档文本切分
- ✅ Embedding向量化
- ✅ Chroma向量数据库存储
- ✅ 相似度检索
- ✅ RAG问答
- ✅ LLM生成自然语言回答

## 环境配置

### 创建虚拟环境

```bash
python -m venv .venv
```

### 激活虚拟环境

Windows:

```bash
.venv\Scripts\activate
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

创建 `.env`：

```env
OPENAI_API_KEY=your_api_key
MODEL_DIR: Specifies the local cache path (directory) for ModelScope models.
```

## 构建知识库

将产品文档放入：

```
data/
```

运行：

```bash
python ingest.py
```

完成：

- 文档读取
- 文本切片
- Embedding生成
- Chroma数据库创建

## 运行项目

执行：

```bash
python app.py
```

## RAG流程说明

### 1. 文档加载

使用 LangChain Document Loader 加载产品文档。

### 2. 文本切片

将长文档拆分成多个 Chunk。

### 3. 向量化

文本通过 Embedding 模型转换为向量。

### 4. 向量检索

根据用户问题检索最相关文档片段。

### 5. LLM生成回答

结合上下文和用户问题，由大语言模型生成答案。

## 后续优化计划

### Version 1.0

基础 RAG 问答系统：

- PDF知识库
- 向量检索
- LLM问答

### Version 2.0

增加 Web 交互界面：

- [ ] Streamlit聊天页面
- [ ] 用户输入框
- [ ] Markdown回答展示
- [ ] Loading状态优化

### Version 3.0

增加多轮对话能力：

- [ ] Conversation Memory
- [ ] 历史消息管理
- [ ] 上下文连续问答

### Version 4.0

Agent能力增强：

- [ ] Prompt优化
- [ ] 引用来源展示
- [ ] 用户反馈机制
- [ ] 工具调用
- [ ] 项目部署上线

## 项目目标

最终将本项目升级为：

> 一个具备企业知识库检索、多轮对话、智能问答能力的 AI Agent 应用。

## License

This project is for learning and demonstration purposes.
