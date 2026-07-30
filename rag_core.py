"""RAG 的共享能力：配置、文档解析、切块、FAISS + BM25 混合检索。"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import jieba
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


# 所有路径和可调参数集中在这里，避免散落在业务代码中。
@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    local_score_threshold: float = 0.48
    vector_weight: float = 0.65
    bm25_weight: float = 0.35
    web_max_results: int = 5

    @classmethod
    def from_env(cls) -> "Settings":
        """从 .env 读取参数；未配置项使用可直接运行的默认值。"""
        load_dotenv()
        return cls(
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            local_score_threshold=float(
                os.getenv("LOCAL_SCORE_THRESHOLD", cls.local_score_threshold)
            ),
            vector_weight=float(os.getenv("VECTOR_WEIGHT", cls.vector_weight)),
            bm25_weight=float(os.getenv("BM25_WEIGHT", cls.bm25_weight)),
            web_max_results=int(os.getenv("WEB_MAX_RESULTS", cls.web_max_results)),
        )


# 文件名本身包含报告编号、类型和主/附件信息，把它们保留为检索元数据。
def parse_report_filename(path: Path) -> dict[str, str]:
    prefix, _, title = path.stem.partition("#")
    fields = prefix.split("@")
    return {
        "source": str(path),
        "file_name": path.name,
        "report_title": title or path.stem,
        "document_code": fields[0],
        "document_type": fields[-2] if len(fields) >= 3 else "",
        "document_role": fields[-1] if len(fields) >= 2 else "",
    }


# Markdown 是主数据；Word/PDF 使用 LangChain Loader，按页或整篇保留来源。
def load_documents(input_path: Path) -> list[Document]:
    """读取一个文件或目录中的 .md/.docx/.pdf，忽略其他格式。"""
    files = [input_path] if input_path.is_file() else sorted(input_path.rglob("*"))
    documents: list[Document] = []

    for path in files:
        suffix = path.suffix.lower()
        if suffix not in {".md", ".docx", ".pdf"}:
            continue

        metadata = parse_report_filename(path)
        if suffix == ".md":
            text = path.read_text(encoding="utf-8")
            documents.append(Document(page_content=text, metadata=metadata))
        else:
            loader = Docx2txtLoader(str(path)) if suffix == ".docx" else PyPDFLoader(str(path))
            for page in loader.load():
                page.metadata.update(metadata)
                documents.append(page)

    if not documents:
        raise ValueError(f"没有在 {input_path} 中找到 .md、.docx 或 .pdf 文件")
    return documents


# 优先按 Markdown 标题和自然段切分，重叠部分用于避免因切块丢失上下文。
def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", ""],
    )
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        identity = f"{chunk.metadata.get('source')}|{chunk.metadata.get('page', '')}|{chunk.page_content}"
        chunk.metadata["chunk_id"] = hashlib.sha1(identity.encode("utf-8")).hexdigest()
    return chunks


# Embedding 在本机运行，模型缓存到 models/，API 密钥不会发送给 Embedding 服务。
def create_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        cache_folder=str(settings.model_dir),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# 中文先用 jieba 分词，英文/编号保留完整 token，供 BM25 使用。
def tokenize(text: str) -> list[str]:
    return [token.lower() for token in jieba.lcut(text) if re.search(r"[\w\u4e00-\u9fff]", token)]


class HybridRetriever:
    """将 FAISS 语义相似度与 BM25 关键词分数融合为统一的 0~1 分数。"""

    def __init__(self, settings: Settings):
        chunks_path = settings.data_dir / "chunks.json"
        index_path = settings.data_dir / "faiss_index"
        if not chunks_path.exists() or not index_path.exists():
            raise FileNotFoundError("尚未建立索引，请先运行 python ingest.py")

        raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.documents = [Document(**item) for item in raw_chunks]
        self.bm25 = BM25Okapi([tokenize(doc.page_content) for doc in self.documents])
        self.settings = settings
        self.vector_store = FAISS.load_local(
            str(index_path),
            create_embeddings(settings),
            allow_dangerous_deserialization=True,  # 只加载本项目自己生成的本地索引。
        )

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        """分别召回候选，再按配置权重融合；分数越高越匹配。"""
        candidate_count = max(top_k * 3, 12)
        vector_hits = self.vector_store.similarity_search_with_score(query, k=candidate_count)
        bm25_scores = self.bm25.get_scores(tokenize(query))
        best_bm25 = max(bm25_scores, default=0.0)
        merged: dict[str, dict] = {}

        for doc, distance in vector_hits:
            chunk_id = doc.metadata["chunk_id"]
            merged[chunk_id] = {
                "document": doc,
                "vector_score": 1.0 / (1.0 + max(float(distance), 0.0)),
                "bm25_score": 0.0,
            }

        for index in sorted(range(len(bm25_scores)), key=bm25_scores.__getitem__, reverse=True)[:candidate_count]:
            doc = self.documents[index]
            chunk_id = doc.metadata["chunk_id"]
            hit = merged.setdefault(
                chunk_id, {"document": doc, "vector_score": 0.0, "bm25_score": 0.0}
            )
            hit["bm25_score"] = float(bm25_scores[index] / best_bm25) if best_bm25 > 0 else 0.0

        for hit in merged.values():
            hit["score"] = (
                self.settings.vector_weight * hit["vector_score"]
                + self.settings.bm25_weight * hit["bm25_score"]
            )
        return sorted(merged.values(), key=lambda item: item["score"], reverse=True)[:top_k]


# 将用户临时上传的 Word/PDF/Markdown 转成上下文，不污染已经持久化的知识库。
def read_attachment(path: Path, max_chars: int = 14_000) -> str:
    documents = load_documents(path)
    text = "\n\n".join(doc.page_content for doc in documents)
    return text[:max_chars]
