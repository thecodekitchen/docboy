from enum import Enum

llm_imports = {
    "ai21": "from langchain_community.llms.ai21 import LLM",
    "aleph_alpha": "from langchain_community.llms.aleph_alpha import LLM",
    "amazon_api_gateway": "from langchain_community.llms.amazon_api_gateway import LLM",
    "anthropic": "from langchain_community.llms.anthropic import LLM",
    "anyscale": "from langchain_community.llms.anyscale import BaseOpenAI",
    "aphrodite": "from langchain_community.llms.aphrodite import Aphrodite",
    "arcee": "from langchain_community.llms.arcee import LLM",
    "aviary": "from langchain_community.llms.aviary import LLM",
    "azureml_endpoint": "from langchain_community.llms.azureml_endpoint import AzureMLOnlineEndpoint",
    "baichuan": "from langchain_community.llms.baichuan import LLM",
    "baidu_qianfan_endpoint": "from langchain_community.llms.baidu_qianfan_endpoint import QianfanLLMEndpoint",
    "bananadev": "from langchain_community.llms.bananadev import LLM",
    "baseten": "from langchain_community.llms.baseten import LLM",
    "beam": "from langchain_community.llms.beam import LLM",
    "bedrock": "from langchain_community.llms.bedrock import LLM",
    "bigdl_llm": "from langchain_community.llms.bigdl_llm import LLM",
    "bittensor": "from langchain_community.llms.bittensor import NIBittensorLLM",
    "cerebriumai": "from langchain_community.llms.cerebriumai import LLM",
    "chatglm": "from langchain_community.llms.chatglm import LLM",
    "chatglm3": "from langchain_community.llms.chatglm3 import LLM",
    "clarifai": "from langchain_community.llms.clarifai import LLM",
    "cloudflare_workersai": "from langchain_community.llms.cloudflare_workersai import LLM",
    "cohere": "from langchain_community.llms.cohere import LLM",
    "ctransformers": "from langchain_community.llms.ctransformers import LLM",
    "ctranslate2": "from langchain_community.llms.ctranslate2 import CTranslate2",
    "databricks": "from langchain_community.llms.databricks import LLM",
    "deepinfra": "from langchain_community.llms.deepinfra import LLM",
    "deepsparse": "from langchain_community.llms.deepsparse import LLM",
    "edenai": "from langchain_community.llms.edenai import LLM",
    "exllamav2": "from langchain_community.llms.exllamav2 import LLM",
    "fake": "from langchain_community.llms.fake import LLM",
    "fireworks": "from langchain_community.llms.fireworks import Fireworks",
    "forefrontai": "from langchain_community.llms.forefrontai import LLM",
    "friendli": "from langchain_community.llms.friendli import LLM",
    "gigachat": "from langchain_community.llms.gigachat import GigaChat",
    "google_palm": "from langchain_community.llms.google_palm import GooglePalm",
    "gooseai": "from langchain_community.llms.gooseai import LLM",
    "gpt4all": "from langchain_community.llms.gpt4all import LLM",
    "gradient_ai": "from langchain_community.llms.gradient_ai import GradientLLM",
    "huggingface_endpoint": "from langchain_community.llms.huggingface_endpoint import LLM",
    "huggingface_hub": "from langchain_community.llms.huggingface_hub import LLM",
    "huggingface_pipeline": "from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline",
    "huggingface_text_gen_inference": "from langchain_community.llms.huggingface_text_gen_inference import LLM",
    "human": "from langchain_community.llms.human import LLM",
    "ipex_llm": "from langchain_community.llms.ipex_llm import LLM",
    "javelin_ai_gateway": "from langchain_community.llms.javelin_ai_gateway import LLM",
    "koboldai": "from langchain_community.llms.koboldai import LLM",
    "konko": "from langchain_community.llms.konko import LLM",
    "layerup_security": "from langchain_community.llms.layerup_security import LayerupSecurity",
    "llamacpp": "from langchain_community.llms.llamacpp import LlamaCpp",
    "llamafile": "from langchain_community.llms.llamafile import Llamafile",
    "manifest": "from langchain_community.llms.manifest import ManifestWrapper",
    "minimax": "from langchain_community.llms.minimax import Minimax",
    "mlflow": "from langchain_community.llms.mlflow import Mlflow",
    "mlflow_ai_gateway": "from langchain_community.llms.mlflow_ai_gateway import MlflowAIGateway",
    "mlx_pipeline": "from langchain_community.llms.mlx_pipeline import MLXPipeline",
    "modal": "from langchain_community.llms.modal import Modal",
    "moonshot": "from langchain_community.llms.moonshot import Moonshot",
    "mosaicml": "from langchain_community.llms.mosaicml import MosaicML",
    "nlpcloud": "from langchain_community.llms.nlpcloud import NLPCloud",
    "oci_data_science_model_deployment_endpoint": "from langchain_community.llms.oci_data_science_model_deployment_endpoint import OCIModelDeploymentVLLM",
    "oci_generative_ai": "from langchain_community.llms.oci_generative_ai import OCIGenAI",
    "octoai_endpoint": "from langchain_community.llms.octoai_endpoint import OctoAIEndpoint",
    "ollama": "from langchain_community.llms.ollama import Ollama",
    "opaqueprompts": "from langchain_community.llms.opaqueprompts import OpaquePrompts",
    "openai": "from langchain_community.llms.openai import OpenAIChat",
    "openllm": "from langchain_community.llms.openllm import OpenLLM",
    "openlm": "from langchain_community.llms.openlm import OpenLM",
    "outlines": "from langchain_community.llms.outlines import Outlines",
    "pai_eas_endpoint": "from langchain_community.llms.pai_eas_endpoint import PaiEasEndpoint",
    "petals": "from langchain_community.llms.petals import Petals",
    "pipelineai": "from langchain_community.llms.pipelineai import PipelineAI",
    "predibase": "from langchain_community.llms.predibase import Predibase",
    "predictionguard": "from langchain_community.llms.predictionguard import PredictionGuard",
    "promptlayer_openai": "from langchain_community.llms.promptlayer_openai import PromptLayerOpenAIChat",
    "replicate": "from langchain_community.llms.replicate import Replicate",
    "rwkv": "from langchain_community.llms.rwkv import RWKV",
    "sagemaker_endpoint": "from langchain_community.llms.sagemaker_endpoint import SagemakerEndpoint",
    "sambanova": "from langchain_community.llms.sambanova import SambaStudio",
    "self_hosted": "from langchain_community.llms.self_hosted import SelfHostedPipeline",
    "self_hosted_hugging_face": "from langchain_community.llms.self_hosted_hugging_face import SelfHostedPipeline",
    "solar": "from langchain_community.llms.solar import Solar",
    "sparkllm": "from langchain_community.llms.sparkllm import SparkLLM",
    "stochasticai": "from langchain_community.llms.stochasticai import StochasticAI",
    "symblai_nebula": "from langchain_community.llms.symblai_nebula import Nebula",
    "textgen": "from langchain_community.llms.textgen import TextGen",
    "titan_takeoff": "from langchain_community.llms.titan_takeoff import TitanTakeoff",
    "together": "from langchain_community.llms.together import Together",
    "tongyi": "from langchain_community.llms.tongyi import Tongyi",
    "vertexai": "from langchain_community.llms.vertexai import VertexAIModelGarden",
    "vllm": "from langchain_community.llms.vllm import VLLMOpenAI",
    "volcengine_maas": "from langchain_community.llms.volcengine_maas import VolcEngineMaasLLM",
    "watsonxllm": "from langchain_community.llms.watsonxllm import WatsonxLLM",
    "weight_only_quantization": "from langchain_community.llms.weight_only_quantization import WeightOnlyQuantPipeline",
    "writer": "from langchain_community.llms.writer import Writer",
    "xinference": "from langchain_community.llms.xinference import Xinference",
    "yandex": "from langchain_community.llms.yandex import YandexGPT",
    "yi": "from langchain_community.llms.yi import YiLLM",
    "you": "from langchain_community.llms.you import You",
    "yuan2": "from langchain_community.llms.yuan2 import Yuan2"
}

class LLMProvider(Enum):
    AI21 = "ai21"
    ALEPH_ALPHA = "aleph_alpha"
    AMAZON_API_GATEWAY = "amazon_api_gateway"
    ANTHROPIC = "anthropic"
    ANYSCALE = "anyscale"
    APHRODITE = "aphrodite"
    ARCEE = "arcee"
    AVIARY = "aviary"
    AZUREML_ENDPOINT = "azureml_endpoint"
    BAICHUAN = "baichuan"
    BAIDU_QIANFAN_ENDPOINT = "baidu_qianfan_endpoint"
    BANANADEV = "bananadev"
    BASETEN = "baseten"
    BEAM = "beam"
    BEDROCK = "bedrock"
    BIGDL_LLM = "bigdl_llm"
    BITTENSOR = "bittensor"
    CEREBRIUMAI = "cerebriumai"
    CHATGLM = "chatglm"
    CHATGLM3 = "chatglm3"
    CLARIFAI = "clarifai"
    CLOUDFLARE_WORKERSAI = "cloudflare_workersai"
    COHERE = "cohere"
    CTRANSFORMERS = "ctransformers"
    CTRANSLATE2 = "ctranslate2"
    DATABRICKS = "databricks"
    DEEPINFRA = "deepinfra"
    DEEPSPARSE = "deepsparse"
    EDENAI = "edenai"
    EXLLAMAV2 = "exllamav2"
    FAKE = "fake"
    FIREWORKS = "fireworks"
    FOREFRONTAI = "forefrontai"
    FRIENDLI = "friendli"
    GIGACHAT = "gigachat"
    GOOGLE_PALM = "google_palm"
    GOOSEAI = "gooseai"
    GPT4ALL = "gpt4all"
    GRADIENT_AI = "gradient_ai"
    HUGGINGFACE_ENDPOINT = "huggingface_endpoint"
    HUGGINGFACE_HUB = "huggingface_hub"
    HUGGINGFACE_PIPELINE = "huggingface_pipeline"
    HUGGINGFACE_TEXT_GEN_INFERENCE = "huggingface_text_gen_inference"
    HUMAN = "human"
    IPEX_LLM = "ipex_llm"
    JAVELIN_AI_GATEWAY = "javelin_ai_gateway"
    KOBOLDAI = "koboldai"
    KONKO = "konko"
    LAYERUP_SECURITY = "layerup_security"
    LLAMACPP = "llamacpp"
    LLAMAFILE = "llamafile"
    MANIFEST = "manifest"
    MINIMAX = "minimax"
    MLFLOW = "mlflow"
    MLFLOW_AI_GATEWAY = "mlflow_ai_gateway"
    MLX_PIPELINE = "mlx_pipeline"
    MODAL = "modal"
    MOONSHOT = "moonshot"
    MOSAICML = "mosaicml"
    NLPCLOUD = "nlpcloud"
    OCI_DATA_SCIENCE_MODEL_DEPLOYMENT_ENDPOINT = "oci_data_science_model_deployment_endpoint"
    OCI_GENERATIVE_AI = "oci_generative_ai"
    OCTOAI_ENDPOINT = "octoai_endpoint"
    OLLAMA = "ollama"
    OPAQUEPROMPTS = "opaqueprompts"
    OPENAI = "openai"
    OPENLLM = "openllm"
    OPENLM = "openlm"
    OUTLINES = "outlines"
    PAI_EAS_ENDPOINT = "pai_eas_endpoint"
    PETALS = "petals"
    PIPELINEAI = "pipelineai"
    PREDIBASE = "predibase"
    PREDICTIONGUARD = "predictionguard"
    PROMPTLAYER_OPENAI = "promptlayer_openai"
    REPLICATE = "replicate"
    RWKV = "rwkv"
    SAGEMAKER_ENDPOINT = "sagemaker_endpoint"
    SAMBANOVA = "sambanova"
    SELF_HOSTED = "self_hosted"
    SELF_HOSTED_HUGGING_FACE = "self_hosted_hugging_face"
    SOLAR = "solar"
    SPARKLLM = "sparkllm"
    STOCHASTICAI = "stochasticai"
    SYMBLAI_NEBULA = "symblai_nebula"
    TEXTGEN = "textgen"
    TITAN_TAKEOFF = "titan_takeoff"
    TOGETHER = "together"
    TONGYI = "tongyi"
    VERTEXAI = "vertexai"
    VLLM = "vllm"
    VOLCENGINE_MAAS = "volcengine_maas"
    WATSONXLLM = "watsonxllm"
    WEIGHT_ONLY_QUANTIZATION = "weight_only_quantization"
    WRITER = "writer"
    XINFERENCE = "xinference"
    YANDEX = "yandex"
    YI = "yi"
    YOU = "you"
    YUAN2 = "yuan2"

