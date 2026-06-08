import os

# ========== 网络配置（必须在其他导入之前） ==========
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import streamlit as st
from langchain_deepseek import ChatDeepSeek
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from docx import Document

# ========== 配置 ==========
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


# ========== 初始化向量库 ==========
@st.cache_resource
def init_vectorstore():
    """初始化向量数据库"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    except Exception as e:
        st.error(f"向量库初始化失败：{str(e)}")
        return None


vectorstore = init_vectorstore()


# ========== 知识库管理 ==========
def get_documents():
    """获取文档列表（带错误处理）"""
    if vectorstore is None:
        return []
    try:
        docs = vectorstore.get(include=["metadatas"])
        sources = set()
        for meta in docs.get("metadatas", []):
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(sources)
    except Exception as e:
        st.error(f"获取文档列表失败：{str(e)}")
        return []


def delete_document(source: str):
    """删除文档（带错误处理）"""
    if vectorstore is None:
        return False
    try:
        docs = vectorstore.get(include=["metadatas"])
        id_to_del = []
        for i, meta in enumerate(docs.get("metadatas", [])):
            if meta and "source" in meta and meta["source"] == source:
                id_to_del.append(docs["ids"][i])
        if id_to_del:
            vectorstore.delete(ids=id_to_del)
            return True
        return False
    except Exception as e:
        st.error(f"删除文档失败：{str(e)}")
        return False


def add_text(content: str, source: str = "用户输入"):
    """添加文本到文本库（同名文档自动覆盖，带错误处理）"""
    if vectorstore is None:
        st.error("向量库未初始化")
        return False

    # 错误处理：内容为空
    if not content or not content.strip():
        st.error("内容为空，无法添加")
        return False

    try:
        # 同名文档先删除（实现覆盖）
        if source in get_documents():
            delete_document(source)

        # 切分文本
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        chunks = text_splitter.split_text(content)

        if not chunks:
            st.error("文本切分失败，没有生成任何片段")
            return False

        # 添加到向量库
        vectorstore.add_texts(texts=chunks, metadatas=[{"source": source}] * len(chunks))
        # 持久化保存
        vectorstore.persist()
        return True

    except Exception as e:
        st.error(f"添加文档失败：{str(e)}")
        return False


# ========== 流式查询 ==========
def rag_stream(query: str):
    """流式返回检索结果"""

    # ========== 优先匹配文档名 ==========
    all_doc_names = get_documents()
    exact_matches = [name for name in all_doc_names if query.lower() in name.lower()]

    # 如果有精确匹配的文档名，优先返回该文档的内容
    if exact_matches:
        # 检索该文档的内容
        try:
            docs = vectorstore.similarity_search(query, k=5)
            # 过滤出属于匹配文档名的结果
            filtered_docs = [doc for doc in docs if doc.metadata.get('source') in exact_matches]
            if filtered_docs:
                context = "\n\n".join([doc.page_content for doc in filtered_docs[:3]])
            else:
                context = "（该文档内容为空或无法检索）"
        except:
            context = "（该文档内容为空或无法检索）"
    else:
        # 没有匹配文档名，正常语义检索
        try:
            docs = vectorstore.similarity_search(query, k=3)
        except Exception as e:
            yield f"❌ 检索失败：{str(e)}"
            return

        if not docs:
            yield "📚 文档中没有提到相关内容"
            return

        context = "\n\n".join([doc.page_content for doc in docs])

    # 获取对话历史
    history_text = ""
    if "messages" in st.session_state and st.session_state.messages:
        recent = st.session_state.messages[-4:]
        history_parts = []
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "助手"
            history_parts.append(f"{role}: {msg['content']}")
        if history_parts:
            history_text = "【对话历史】\n" + "\n".join(history_parts) + "\n\n"

    try:
        llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3, streaming=True)
    except Exception as e:
        yield f"❌ 模型初始化失败：{str(e)}"
        return

    prompt = f"""你是一个知识库问答系统。

{history_text}
【知识库内容】
{context}

【当前问题】
{query}

规则：
1. 如果知识库中有相关信息，请直接回答
2. 如果知识库中没有直接答案，但可以根据内容合理推断，请给出推断结果
3. 只有完全找不到任何相关信息时，才回答"文档中没有提到"
4. 回答要简洁自然

答案："""

    try:
        for chunk in llm.stream(prompt):
            yield chunk.content
    except Exception as e:
        yield f"❌ 生成回答失败：{str(e)}"

# ========== Agent Tool封装 ==========
@tool
def search_knowledge(query: str) -> str:
    """基于知识库查询信息（Agent工具，带错误处理）"""

    if vectorstore is None:
        return "错误：向量库未初始化"

    if not query or not query.strip():
        return "错误：查询内容不能为空"

    try:
        docs = vectorstore.similarity_search(query, k=3)
    except Exception as e:
        return f"检索失败：{str(e)}"

    # 如果内容检索失败，尝试匹配文档名
    if not docs:
        all_doc_names = get_documents()
        matched_docs = [name for name in all_doc_names if query.lower() in name.lower()]
        if matched_docs:
            return f"找到以下相关文档：\n" + "\n".join(matched_docs)
        return "没有找到相关的信息"

    return "\n\n".join([doc.page_content for doc in docs])
# ========== Streamlit UI ==========
st.title("📚 Agent 知识库系统")

with st.sidebar:
    st.header("知识库管理")

    tab1, tab2 = st.tabs(["📁 上传文件", "✏️ 直接输入文本"])

    with tab1:
        with st.form(key="upload_form", clear_on_submit=True):
            uploaded = st.file_uploader("选择文件", type=["txt", "docx", "pdf"])
            st.caption("支持 TXT、PDF、DOCX 格式 | 标题自动使用文件名 | 同名文件自动覆盖")

            submitted = st.form_submit_button("添加到知识库", use_container_width=True)

            if submitted and uploaded:
                try:
                    file_ext = uploaded.name.split('.')[-1].lower()

                    if file_ext == 'txt':
                        content = uploaded.read().decode("utf-8")
                        if add_text(content, uploaded.name):
                            st.success(f"✅ 已添加文件：{uploaded.name}")

                    elif file_ext == 'pdf':
                        pdf_reader = PdfReader(uploaded)
                        content = ""
                        for page in pdf_reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                content += page_text + "\n"

                        if content.strip():
                            if add_text(content, uploaded.name):
                                st.success(f"✅ 已添加文件：{uploaded.name}")
                        else:
                            st.error("PDF 文件无法提取文本内容（可能是扫描件）")

                    elif file_ext == 'docx':
                        doc = Document(uploaded)
                        content = "\n".join([para.text for para in doc.paragraphs])

                        if content.strip():
                            if add_text(content, uploaded.name):
                                st.success(f"✅ 已添加文件：{uploaded.name}")
                        else:
                            st.error("DOCX 文件无法提取文本内容")

                except Exception as e:
                    st.error(f"处理文件时出错：{str(e)}")

            elif submitted:
                st.warning("请选择文件")

    with tab2:
        with st.form(key="text_form", clear_on_submit=True):
            text_title = st.text_input("标题（可选）", placeholder="默认：用户输入")
            text_content = st.text_area("文本内容", height=150, placeholder="在这里粘贴文本...")

            submitted = st.form_submit_button("添加到知识库", use_container_width=True)

            if submitted and text_content:
                title = text_title if text_title else "用户输入"
                if add_text(text_content, title):
                    st.success(f"✅ 已添加文本：{title}")
            elif submitted:
                st.warning("请输入文本内容")

    st.markdown("---")
    st.subheader("📋 文档列表")

    # 搜索框
    search_term = st.text_input("🔍 搜索文档", placeholder="输入文档名关键词...", key="doc_search")

    docs = get_documents()

    # 根据搜索词过滤文档
    if search_term:
        docs = [doc for doc in docs if search_term.lower() in doc.lower()]
        if not docs:
            st.info(f"未找到包含「{search_term}」的文档")

    if not docs:
        st.info("暂无文档")
    else:
        # 分页设置
        page_size = 5
        total_pages = max(1, (len(docs) + page_size - 1) // page_size)

        # 初始化页码
        if "doc_page" not in st.session_state:
            st.session_state.doc_page = 1

        # 如果搜索后当前页超出总页数，重置为第1页
        if st.session_state.doc_page > total_pages:
            st.session_state.doc_page = 1

        # 翻页控件
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("◀ 上一页", disabled=(st.session_state.doc_page <= 1)):
                st.session_state.doc_page -= 1
                st.rerun()
        with col2:
            st.write(f"第 {st.session_state.doc_page} / {total_pages} 页")
        with col3:
            if st.button("下一页 ▶", disabled=(st.session_state.doc_page >= total_pages)):
                st.session_state.doc_page += 1
                st.rerun()

        # 显示当前页文档
        start_idx = (st.session_state.doc_page - 1) * page_size
        end_idx = start_idx + page_size

        for doc in docs[start_idx:end_idx]:
            col1, col2 = st.columns([4, 1])
            col1.write(f"📄 {doc[:50]}")
            if col2.button("删除", key=f"del_{doc}"):
                if delete_document(doc):
                    st.success(f"已删除：{doc}")
                    st.rerun()

# 主界面
st.header("🤖 Agent 问答")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("输入问题"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(rag_stream(prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})