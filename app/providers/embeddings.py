from enum import Enum
embeddings_imports = {
    "aleph_alpha": "from langchain_community.embeddings.aleph_alpha import AlephAlphaSymmetricSemanticEmbedding",
    "anyscale": "from langchain_community.embeddings.anyscale import OpenAIEmbeddings",
    "ascend": "from langchain_community.embeddings.ascend import AscendEmbeddings",
    "awa": "from langchain_community.embeddings.awa import AwaEmbeddings",
    "azure_openai": "from langchain_community.embeddings.azure_openai import OpenAIEmbeddings",
    "baichuan": "from langchain_community.embeddings.baichuan import BaichuanTextEmbeddings",
    "baidu_qianfan_endpoint": "from langchain_community.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint",
    "bedrock": "from langchain_community.embeddings.bedrock import BedrockEmbeddings",
    "bookend": "from langchain_community.embeddings.bookend import BookendEmbeddings",
    "clarifai": "from langchain_community.embeddings.clarifai import ClarifaiEmbeddings",
    "cloudflare_workersai": "from langchain_community.embeddings.cloudflare_workersai import CloudflareWorkersAIEmbeddings",
    "clova": "from langchain_community.embeddings.clova import ClovaEmbeddings",
    "cohere": "from langchain_community.embeddings.cohere import CohereEmbeddings",
    "dashscope": "from langchain_community.embeddings.dashscope import DashScopeEmbeddings",
    "databricks": "from langchain_community.embeddings.databricks import MlflowEmbeddings",
    "deepinfra": "from langchain_community.embeddings.deepinfra import DeepInfraEmbeddings",
    "edenai": "from langchain_community.embeddings.edenai import EdenAiEmbeddings",
    "elasticsearch": "from langchain_community.embeddings.elasticsearch import ElasticsearchEmbeddings",
    "embaas": "from langchain_community.embeddings.embaas import EmbaasEmbeddings",
    "ernie": "from langchain_community.embeddings.ernie import ErnieEmbeddings",
    "fake": "from langchain_community.embeddings.fake import FakeEmbeddings",
    "fastembed": "from langchain_community.embeddings.fastembed import FastEmbedEmbeddings",
    "gigachat": "from langchain_community.embeddings.gigachat import GigaChatEmbeddings",
    "google_palm": "from langchain_community.embeddings.google_palm import GooglePalmEmbeddings",
    "gpt4all": "from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings",
    "gradient_ai": "from langchain_community.embeddings.gradient_ai import GradientEmbeddings",
    "huggingface": "from langchain_community.embeddings.huggingface import HuggingFaceInstructEmbeddings",
    "huggingface_hub": "from langchain_community.embeddings.huggingface_hub import HuggingFaceHubEmbeddings",
    "hunyuan": "from langchain_community.embeddings.hunyuan import HunyuanEmbeddings",
    "infinity": "from langchain_community.embeddings.infinity import InfinityEmbeddings",
    "infinity_local": "from langchain_community.embeddings.infinity_local import InfinityEmbeddingsLocal",
    "ipex_llm": "from langchain_community.embeddings.ipex_llm import IpexLLMBgeEmbeddings",
    "itrex": "from langchain_community.embeddings.itrex import QuantizedBgeEmbeddings",
    "javelin_ai_gateway": "from langchain_community.embeddings.javelin_ai_gateway import JavelinAIGatewayEmbeddings",
    "jina": "from langchain_community.embeddings.jina import JinaEmbeddings",
    "johnsnowlabs": "from langchain_community.embeddings.johnsnowlabs import JohnSnowLabsEmbeddings",
    "laser": "from langchain_community.embeddings.laser import LaserEmbeddings",
    "llamacpp": "from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings",
    "llamafile": "from langchain_community.embeddings.llamafile import LlamafileEmbeddings",
    "llm_rails": "from langchain_community.embeddings.llm_rails import LLMRailsEmbeddings",
    "localai": "from langchain_community.embeddings.localai import LocalAIEmbeddings",
    "minimax": "from langchain_community.embeddings.minimax import MiniMaxEmbeddings",
    "mlflow": "from langchain_community.embeddings.mlflow import MlflowEmbeddings",
    "mlflow_gateway": "from langchain_community.embeddings.mlflow_gateway import MlflowAIGatewayEmbeddings",
    "model2vec": "from langchain_community.embeddings.model2vec import Model2vecEmbeddings",
    "modelscope_hub": "from langchain_community.embeddings.modelscope_hub import ModelScopeEmbeddings",
    "mosaicml": "from langchain_community.embeddings.mosaicml import MosaicMLInstructorEmbeddings",
    "naver": "from langchain_community.embeddings.naver import ClovaXEmbeddings",
    "nemo": "from langchain_community.embeddings.nemo import NeMoEmbeddings",
    "nlpcloud": "from langchain_community.embeddings.nlpcloud import NLPCloudEmbeddings",
    "oci_generative_ai": "from langchain_community.embeddings.oci_generative_ai import OCIGenAIEmbeddings",
    "octoai_embeddings": "from langchain_community.embeddings.octoai_embeddings import OpenAIEmbeddings",
    "ollama": "from langchain_community.embeddings.ollama import OllamaEmbeddings",
    "openai": "from langchain_community.embeddings.openai import OpenAIEmbeddings",
    "openvino": "from langchain_community.embeddings.openvino import OpenVINOEmbeddings",
    "optimum_intel": "from langchain_community.embeddings.optimum_intel import QuantizedBiEncoderEmbeddings",
    "oracleai": "from langchain_community.embeddings.oracleai import OracleEmbeddings",
    "ovhcloud": "from langchain_community.embeddings.ovhcloud import OVHCloudEmbeddings",
    "premai": "from langchain_community.embeddings.premai import PremAIEmbeddings",
    "sagemaker_endpoint": "from langchain_community.embeddings.sagemaker_endpoint import SagemakerEndpointEmbeddings",
    "sambanova": "from langchain_community.embeddings.sambanova import SambaStudioEmbeddings",
    "self_hosted": "from langchain_community.embeddings.self_hosted import SelfHostedEmbeddings",
    "self_hosted_hugging_face": "from langchain_community.embeddings.self_hosted_hugging_face import SelfHostedHuggingFaceInstructEmbeddings",
    "sentence_transformer": "from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings",
    "solar": "from langchain_community.embeddings.solar import SolarEmbeddings",
    "spacy_embeddings": "from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings",
    "sparkllm": "from langchain_community.embeddings.sparkllm import SparkLLMTextEmbeddings",
    "tensorflow_hub": "from langchain_community.embeddings.tensorflow_hub import TensorflowHubEmbeddings",
    "text2vec": "from langchain_community.embeddings.text2vec import Text2vecEmbeddings",
    "textembed": "from langchain_community.embeddings.textembed import TextEmbedEmbeddings",
    "titan_takeoff": "from langchain_community.embeddings.titan_takeoff import TitanTakeoffEmbed",
    "vertexai": "from langchain_community.embeddings.vertexai import VertexAIEmbeddings",
    "volcengine": "from langchain_community.embeddings.volcengine import VolcanoEmbeddings",
    "voyageai": "from langchain_community.embeddings.voyageai import VoyageEmbeddings",
    "xinference": "from langchain_community.embeddings.xinference import XinferenceEmbeddings",
    "yandex": "from langchain_community.embeddings.yandex import YandexGPTEmbeddings",
    "zhipuai": "from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings"
}

class EmbeddingsProvider(Enum):
    ALEPH_ALPHA = "aleph_alpha"
    ANYSCALE = "anyscale"
    ASCEND = "ascend"
    AWA = "awa"
    AZURE_OPENAI = "azure_openai"
    BAICHUAN = "baichuan"
    BAIDU_QIANFAN_ENDPOINT = "baidu_qianfan_endpoint"
    BEDROCK = "bedrock"
    BOOKEND = "bookend"
    CLARIFAI = "clarifai"
    CLOUDFLARE_WORKERSAI = "cloudflare_workersai"
    CLOVA = "clova"
    COHERE = "cohere"
    COZE = "coze"
    DASHSCOPE = "dashscope"
    DATABRICKS = "databricks"
    DEEPINFRA = "deepinfra"
    EDENAI = "edenai"
    ELASTICSEARCH = "elasticsearch"
    EMBAAS = "embaas"
    ERNIE = "ernie"
    FAKE = "fake"
    FASTEMBED = "fastembed"
    GIGACHAT = "gigachat"
    GOOGLE_PALM = "google_palm"
    GPT4ALL = "gpt4all"
    GRADIENT_AI = "gradient_ai"
    HUGGINGFACE = "huggingface"
    HUGGINGFACE_HUB = "huggingface_hub"
    HUNYUAN = "hunyuan"
    INFINITY = "infinity"
    INFINITY_LOCAL = "infinity_local"
    IPEX_LLM = "ipex_llm"
    ITREX = "itrex"
    JAVALIN_AI_GATEWAY = "javelin_ai_gateway"
    JINA = "jina"
    JOHN_SNOW_LABS = "johnsnowlabs"
    LASER = "laser"
    LLAMACPP = "llamacpp"
    LLAMAFILE = "llamafile"
    LLM_RAILS = "llm_rails"
    LOCALAI = "localai"
    MINIMAX = "minimax"
    MLFLOW = "mlflow"
    MLFLOW_GATEWAY = "mlflow_gateway"
    MODEL2VEC = "model2vec"
    MODELSCOPE_HUB = "modelscope_hub"
    MOSAICML = "mosaicml"
    NAVER = "naver"
    NEMO = "nemo"
    NLP_CLOUD = "nlpcloud"
    OCI_GENERATIVE_AI = "oci_generative_ai"
    OCTOAI_EMBEDDINGS = "octoai_embeddings"
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENVINO = "openvino"
    OPTIMUM_INTEL = "optimum_intel"
    ORACLEAI = "oracleai"
    OVHCLOUD = "ovhcloud"
    PREMAI = "premai"
    REKA = "reka"
    SAMBANOVA = "sambanova"
    SELF_HOSTED = "self_hosted"
    SELF_HOSTED_HUGGING_FACE = "self_hosted_hugging_face"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    SPACY_EMBEDDINGS = "spacy_embeddings"
    SPARKLLM = "sparkllm"
    TENSORFLOW_HUB = "tensorflow_hub"
    TEXT2VEC = "text2vec"
    TEXTEMBED = "textembed"
    TITAN_TAKEOFF = "titan_takeoff"
    VERTEXAI = "vertexai"
    VOLCENGINE = "volcengine"
    VOYAGEAI = "voyageai"
    XINFERENCE = "xinference"
    YANDEX = "yandex"
    ZHIPUAI = "zhipuai"