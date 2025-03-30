from enum import Enum

retriever_imports = {
    "arcee": "from langchain_community.retrievers.arcee import ArceeRetriever",
    "arxiv": "from langchain_community.retrievers.arxiv import ArxivRetriever",
    "asknews": "from langchain_community.retrievers.asknews import AskNewsRetriever",
    "azure_ai_search": "from langchain_community.retrievers.azure_ai_search import AzureCognitiveSearchRetriever",
    "bedrock": "from langchain_community.retrievers.bedrock import AmazonKnowledgeBasesRetriever",
    "bm25": "from langchain_community.retrievers.bm25 import BM25Retriever",
    "breebs": "from langchain_community.retrievers.breebs import BreebsRetriever",
    "chaindesk": "from langchain_community.retrievers.chaindesk import ChaindeskRetriever",
    "chatgpt_plugin_retriever": "from langchain_community.retrievers.chatgpt_plugin_retriever import ChatGPTPluginRetriever",
    "cohere_rag_retriever": "from langchain_community.retrievers.cohere_rag_retriever import CohereRagRetriever",
    "databerry": "from langchain_community.retrievers.databerry import DataberryRetriever",
    "docarray": "from langchain_community.retrievers.docarray import DocArrayRetriever",
    "dria_index": "from langchain_community.retrievers.dria_index import DriaRetriever",
    "elastic_search_bm25": "from langchain_community.retrievers.elastic_search_bm25 import ElasticSearchBM25Retriever",
    "embedchain": "from langchain_community.retrievers.embedchain import EmbedchainRetriever",
    "google_cloud_documentai_warehouse": "from langchain_community.retrievers.google_cloud_documentai_warehouse import GoogleDocumentAIWarehouseRetriever",
    "google_vertex_ai_search": "from langchain_community.retrievers.google_vertex_ai_search import GoogleVertexAISearchRetriever",
    "kay": "from langchain_community.retrievers.kay import KayAiRetriever",
    "kendra": "from langchain_community.retrievers.kendra import AmazonKendraRetriever",
    "knn": "from langchain_community.retrievers.knn import KNNRetriever",
    "llama_index": "from langchain_community.retrievers.llama_index import LlamaIndexRetriever",
    "metal": "from langchain_community.retrievers.metal import MetalRetriever",
    "milvus": "from langchain_community.retrievers.milvus import MilvusRetriever",
    "nanopq": "from langchain_community.retrievers.nanopq import NanoPQRetriever",
    "needle": "from langchain_community.retrievers.needle import NeedleRetriever",
    "outline": "from langchain_community.retrievers.outline import OutlineRetriever",
    "pinecone_hybrid_search": "from langchain_community.retrievers.pinecone_hybrid_search import PineconeHybridSearchRetriever",
    "pubmed": "from langchain_community.retrievers.pubmed import PubMedRetriever",
    "pupmed": "from langchain_community.retrievers.pupmed import PubMedRetriever",
    "qdrant_sparse_vector_retriever": "from langchain_community.retrievers.qdrant_sparse_vector_retriever import QdrantSparseVectorRetriever",
    "rememberizer": "from langchain_community.retrievers.rememberizer import RememberizerRetriever",
    "remote_retriever": "from langchain_community.retrievers.remote_retriever import RemoteLangChainRetriever",
    "svm": "from langchain_community.retrievers.svm import SVMRetriever",
    "tavily_search_api": "from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever",
    "tfidf": "from langchain_community.retrievers.tfidf import TFIDFRetriever",
    "thirdai_neuraldb": "from langchain_community.retrievers.thirdai_neuraldb import NeuralDBRetriever",
    "vespa_retriever": "from langchain_community.retrievers.vespa_retriever import VespaRetriever",
    "weaviate_hybrid_search": "from langchain_community.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever",
    "web_research": "from langchain_community.retrievers.web_research import WebResearchRetriever",
    "wikipedia": "from langchain_community.retrievers.wikipedia import WikipediaRetriever",
    "you": "from langchain_community.retrievers.you import YouRetriever",
    "zep": "from langchain_community.retrievers.zep import ZepRetriever",
    "zep_cloud": "from langchain_community.retrievers.zep_cloud import ZepCloudRetriever",
    "zilliz": "from langchain_community.retrievers.zilliz import ZillizRetriever"
}

class RetrieverProvider(Enum):
    ARCEE = "arcee"
    ARXIV = "arxiv"
    ASKNEWS = "asknews"
    AZURE_AI_SEARCH = "azure_ai_search"
    BEDROCK = "bedrock"
    BM25 = "bm25"
    BREEBS = "breebs"
    CHAINDESK = "chaindesk"
    CHATGPT_PLUGIN_RETRIEVER = "chatgpt_plugin_retriever"
    COHERE_RAG_RETRIEVER = "cohere_rag_retriever"
    DATABERRY = "databerry"
    DOCARRAY = "docarray"
    DRIA_INDEX = "dria_index"
    ELASTIC_SEARCH_BM25 = "elastic_search_bm25"
    EMBEDCHAIN = "embedchain"
    GOOGLE_CLOUD_DOCUMENTAI_WAREHOUSE = "google_cloud_documentai_warehouse"
    GOOGLE_VERTEX_AI_SEARCH = "google_vertex_ai_search"
    KAY = "kay"
    KENDRA = "kendra"
    KNN = "knn"
    LLAMA_INDEX = "llama_index"
    METAL = "metal"
    MILVUS = "milvus"
    NANOPQ = "nanopq"
    NEEDLE = "needle"
    OUTLINE = "outline"
    PINECONE_HYBRID_SEARCH = "pinecone_hybrid_search"
    PUBMED = "pubmed"
    PUPMED = "pupmed"
    QDRANT_SPARSE_VECTOR_RETRIEVER = "qdrant_sparse_vector_retriever"
    REMEMBERIZER = "rememberizer"
    REMOTE_RETRIEVER = "remote_retriever"
    SVM = "svm"
    TAVILY_SEARCH_API = "tavily_search_api"
    TFIDFR = "tfidf"
    THIRD_AI_NEURALDB = "thirdai_neuraldb"
    VESPA_RETRIEVER = "vespa_retriever"
    WEAVIATE_HYBRID_SEARCH = "weaviate_hybrid_search"
    WEB_RESEARCH = "web_research"
    WIKIPEDIA = "wikipedia"
    YOU = "you"
    ZEP = "zep"
    ZEP_CLOUD = "zep_cloud"
    ZILLIZ = "zilliz"