from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from input.qdrant_manager import MemoryQdrantManager

class SimpleHybridRetriever:
    def __init__(self, qdrant_retriever, bm25_retriever, weights=[0.5, 0.5]):
        self.retrievers = [qdrant_retriever, bm25_retriever]
        self.weights = weights

    def invoke(self, query: str):
        # 1. 分别让向量库和关键词库去搜索
        all_docs = [r.invoke(query) for r in self.retrievers]
        return self._rrf_merge(all_docs)

    async def ainvoke(self, query: str):
        # 异步调用接口 
        all_docs = [self.retrievers[0].invoke(query), self.retrievers[1].invoke(query)]
        return self._rrf_merge(all_docs)

    def _rrf_merge(self, all_docs):
        """RRF (倒数秩融合) 算法：公平地将两种检索结果合并打分"""
        rrf_score = {}
        doc_map = {}
        
        for i, docs in enumerate(all_docs):
            weight = self.weights[i]
            for rank, doc in enumerate(docs):
                key = doc.page_content
                if key not in doc_map:
                    doc_map[key] = doc
                    rrf_score[key] = 0.0
                # RRF 打分公式
                rrf_score[key] += weight / (rank + 60)
        
        # 按综合分数从高到低排序，选出最相关的 5 条记忆返回
        sorted_keys = sorted(rrf_score.keys(), key=lambda k: rrf_score[k], reverse=True)
        return [doc_map[k] for k in sorted_keys[:5]]


def get_hybrid_retriever():
    """
    实例化并返回自定义的多路融合检索器 (BM25 + Qdrant MMR)
    """
    db_manager = MemoryQdrantManager()
    vector_store = db_manager.vector_store
    
    # ------------------------------------------------
    # 1. 实例化向量检索器 (Qdrant MMR)
    # ------------------------------------------------
    qdrant_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,          
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    )
    
    # ------------------------------------------------
    # 2. 实例化 BM25 关键词检索器
    # ------------------------------------------------
    client = db_manager.client
    records, _ = client.scroll(
        collection_name=db_manager.collection_name,
        limit=10000, 
        with_payload=True,
        with_vectors=False
    )
    
    bm25_docs = []
    for record in records:
        content = record.payload.get("page_content", "")
        bm25_docs.append(Document(page_content=content, metadata=record.payload.get("metadata", {})))
        
    # 防止刚建库时里面没数据导致 BM25 报错
    if not bm25_docs:
        bm25_docs.append(Document(page_content="[记忆库初始化中]"))
        
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = 5
    
    # ------------------------------------------------
    # 3. 返回混合检索器！
    # ------------------------------------------------
    return SimpleHybridRetriever(qdrant_retriever, bm25_retriever, weights=[0.65, 0.35])

