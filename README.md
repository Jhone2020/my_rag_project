# Agent 知识库系统

基于 RAG（检索增强生成）和 Agent 技术的智能知识库问答系统。支持上传文档、语义搜索、流式对话，并提供 Agent 工具调用接口。

## 功能特性

-  **知识库管理**：支持 TXT、PDF、DOCX 格式文档的上传、删除、搜索和分页展示
-  **语义搜索**：基于向量检索的智能问答，支持同义词理解
-  **流式对话**：类似 DeepSeek 的逐字输出体验
-  **Agent 工具**：将知识库查询封装为 LangChain Tool，可被 Agent 调用
-  **多轮对话**：自动记忆对话历史，支持上下文理解
-  **错误处理**：完善的异常捕获和用户友好提示

## 技术栈

- **前端**：Streamlit
- **大模型**：DeepSeek API
- **向量数据库**：Chroma
- **Embedding 模型**：sentence-transformers/all-MiniLM-L6-v2
- **框架**：LangChain

## 项目结构

```
.
├── Rag.py                 # 主程序
├── requirements.txt       # 依赖列表
├── chroma_db/             # 向量数据库存储目录（自动生成）
└── README.md              # 项目说明
```



## 作者

张重阳 - 软件工程专业


# 运行说明

## 环境要求

- Python 3.9 或更高版本
- DeepSeek API Key

## 安装步骤

### 1. 克隆或下载项目

将代码文件 `Rag.py` 和 `requirements.txt` 放在同一目录下。

### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
3. 安装依赖
bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
4. 配置 API Key

在命令行中设置环境变量：

bash
# Windows
set DEEPSEEK_API_KEY=your_api_key

# macOS / Linux
export DEEPSEEK_API_KEY=your_api_key
5. 运行程序
bash
streamlit run Rag.py