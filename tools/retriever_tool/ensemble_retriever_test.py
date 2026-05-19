# 文件：Agent/tools/retriever_tool.py
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from input.qdrant_manager import MemoryQdrantManager

def get_hybrid_retriever():
    """
    实例化并返回一个多路融合检索器 (BM25 + Qdrant MMR)
    """
    db_manager = MemoryQdrantManager()
    vector_store = db_manager.vector_store
    
    # ------------------------------------------------
    # 1. 实例化向量检索器 
    # ------------------------------------------------
    qdrant_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,          # 向量库捞 3 条
            "fetch_k": 15,
            "lambda_mult": 0.5
        }
    )
    
    # ------------------------------------------------
    # 2. 实例化 BM25 关键词检索器
    # ------------------------------------------------
    # BM25 需要知道所有的原始文本。
    # 通过 Qdrant 底层的 scroll 方法，把所有文本抽出来
    client = db_manager.client
    records, _ = client.scroll(
        collection_name=db_manager.collection_name,
        limit=10000, # 确保覆盖所有数据
        with_payload=True,
        with_vectors=False
    )
    
    bm25_docs = []
    for record in records:
        content = record.payload.get("page_content", "")
        # 元数据塞进去，保证融合后的数据格式一致
        bm25_docs.append(Document(page_content=content, metadata=record.payload.get("metadata", {})))
        
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = 2  # 关键词精准捞 2 条
    
    # ------------------------------------------------
    # 3. 终极融合
    # ------------------------------------------------
    # 权重 weights: [0.6, 0.4]
    ensemble_retriever = EnsembleRetriever(
        retrievers=[qdrant_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )
    
    return ensemble_retriever