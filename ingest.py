"""文档处理入口：把本地报告构建为 data/ 下的 FAISS 和 BM25 数据。"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.vectorstores import FAISS

from rag_core import Settings, create_embeddings, load_documents, split_documents


# 入库与用户查询完全分开；重新运行本脚本即可重建持久化索引。
def build_index(input_path: Path) -> None:
    settings = Settings.from_env()
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    documents = load_documents(input_path)
    chunks = split_documents(documents)
    vector_store = FAISS.from_documents(
        chunks,
        create_embeddings(settings),
        normalize_L2=True,
    )
    vector_store.save_local(str(settings.data_dir / "faiss_index"))

    # BM25 不能只保存 FAISS，因此把相同切块保存为可读 JSON。
    serialized = [
        {"page_content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks
    ]
    (settings.data_dir / "chunks.json").write_text(
        json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "embedding_model": settings.embedding_model,
    }
    (settings.data_dir / "index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"索引完成：{len(documents)} 个文档页，{len(chunks)} 个切块，已写入 {settings.data_dir}")


# 默认处理 markdown/，也可传入单个 Word/PDF 或另一个目录。
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建失效分析知识库")
    parser.add_argument("--input", type=Path, default=Path("markdown"), help="文档文件或目录")
    args = parser.parse_args()
    build_index(args.input)
