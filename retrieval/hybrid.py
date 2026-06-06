"""混合检索器 — MMR 向量检索 + BM25 关键词检索 + RRF 融合"""
import asyncio
import logging
import jieba
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from retrieval.qdrant_store import MemoryQdrantManager

logger = logging.getLogger(__name__)

_db_manager = None


def _scroll_all_records(client, collection_name):
    """生成器：逐条 yield Qdrant 中全部记录，避免将全量数据一次性加载到内存"""
    next_offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=next_offset,
            with_payload=True,
            with_vectors=False
        )
        for record in records:
            yield record
        if next_offset is None:
            break


class SimpleHybridRetriever:
    def __init__(self, qdrant_retriever, bm25_retriever,bm25_docs=None, preprocess_func=None, weights=[0.5, 0.5], top_k=5):
        self.retrievers = [qdrant_retriever, bm25_retriever]
        self.weights = weights
        self.top_k = top_k
        # 留存内部状态，用于动态追加
        self.bm25_docs = bm25_docs or []
        self.preprocess_func = preprocess_func

    def invoke(self, query: str):
        all_docs = [r.invoke(query) for r in self.retrievers]
        return self._rrf_merge(all_docs)

    async def ainvoke(self, query: str):
        all_docs = await asyncio.gather(
            asyncio.to_thread(self.retrievers[0].invoke, query),
            asyncio.to_thread(self.retrievers[1].invoke, query)
        )
        return self._rrf_merge(list(all_docs))

    def add_documents(self, new_docs: list[Document]):
        """在线动态追加新记忆并热重构内存 BM25 索引"""
        self.bm25_docs.extend(new_docs)
        # 获取现有的 k 值设定
        current_k = self.retrievers[1].k if hasattr(self.retrievers[1], 'k') else 5
        
        # 重新生成全新的 BM25 检索器实例
        new_bm25 = BM25Retriever.from_documents(
            self.bm25_docs,
            preprocess_func=self.preprocess_func
        )
        new_bm25.k = current_k
        
        # 替换混合检索器中的BM25 分支
        self.retrievers[1] = new_bm25
        logger.info("内存 BM25 检索器热重构成功！当前总缓存文本块: %d 块", len(self.bm25_docs))

    def _rrf_merge(self, all_docs):
        """RRF (倒数秩融合) 算法：公平地将两种检索结果合并打分"""
        rrf_score = {}
        doc_map = {}

        for i, docs in enumerate(all_docs):
            weight = self.weights[i]
            for rank, doc in enumerate(docs):
                # 使用 (内容, 元数据) 复合键，避免不同文档因内容相同而被错误合并
                key = (doc.page_content, str(doc.metadata))
                if key not in doc_map:
                    doc_map[key] = doc
                    rrf_score[key] = 0.0
                rrf_score[key] += weight / (rank + 60)

        sorted_keys = sorted(rrf_score.keys(), key=lambda k: rrf_score[k], reverse=True)
        return [doc_map[k] for k in sorted_keys[:self.top_k]]


def get_hybrid_retriever():
    """
    实例化并返回自定义的多路融合检索器 (BM25 + Qdrant MMR)
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = MemoryQdrantManager()
    db_manager = _db_manager
    vector_store = db_manager.vector_store

    # 1. 实例化向量检索器 (Qdrant MMR)
    qdrant_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    )

    # 2. 实例化 BM25 关键词检索器
    client = db_manager.client

    bm25_docs = []
    for record in _scroll_all_records(client, db_manager.collection_name):
        content = record.payload.get("page_content", "")
        bm25_docs.append(Document(page_content=content, metadata=record.payload.get("metadata", {})))

    if not bm25_docs:
        logger.warning("BM25 语料为空（向量库无数据），使用占位文档")
        bm25_docs.append(Document(page_content="[记忆库初始化中]"))

    bm25_retriever = BM25Retriever.from_documents(
        bm25_docs,
        preprocess_func=lambda text: jieba.lcut(text) if isinstance(text, str) else text,
    )
    bm25_retriever.k = 5

    # 3. 返回混合检索器
    return SimpleHybridRetriever(
        qdrant_retriever, 
        bm25_retriever, 
        bm25_docs=bm25_docs, 
        preprocess_func=lambda text: jieba.lcut(text) if isinstance(text, str) else text, 
        weights=[0.5, 0.5], top_k=5
        )
