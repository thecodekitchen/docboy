from enum import Enum

chat_model_imports = {
    "anthropic": "from langchain_community.chat_models.anthropic import ChatAnthropic",
    "anyscale": "from langchain_community.chat_models.anyscale import ChatOpenAI",
    "azure_openai": "from langchain_community.chat_models.azure_openai import ChatOpenAI",
    "azureml_endpoint": "from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint",
    "baichuan": "from langchain_community.chat_models.baichuan import ChatBaichuan",
    "baidu_qianfan_endpoint": "from langchain_community.chat_models.baidu_qianfan_endpoint import QianfanChatEndpoint",
    "bedrock": "from langchain_community.chat_models.bedrock import BedrockChat",
    "cloudflare_workersai": "from langchain_community.chat_models.cloudflare_workersai import ChatCloudflareWorkersAI",
    "cohere": "from langchain_community.chat_models.cohere import ChatCohere",
    "coze": "from langchain_community.chat_models.coze import ChatCoze",
    "dappier": "from langchain_community.chat_models.dappier import ChatDappierAI",
    "databricks": "from langchain_community.chat_models.databricks import ChatMlflow",
    "deepinfra": "from langchain_community.chat_models.deepinfra import ChatDeepInfra",
    "edenai": "from langchain_community.chat_models.edenai import ChatEdenAI",
    "ernie": "from langchain_community.chat_models.ernie import ErnieBotChat",
    "everlyai": "from langchain_community.chat_models.everlyai import ChatOpenAI",
    "fake": "from langchain_community.chat_models.fake import SimpleChatModel",
    "fireworks": "from langchain_community.chat_models.fireworks import ChatFireworks",
    "friendli": "from langchain_community.chat_models.friendli import ChatFriendli",
    "gigachat": "from langchain_community.chat_models.gigachat import GigaChat",
    "google_palm": "from langchain_community.chat_models.google_palm import ChatGooglePalm",
    "gpt_router": "from langchain_community.chat_models.gpt_router import GPTRouter",
    "huggingface": "from langchain_community.chat_models.huggingface import ChatHuggingFace",
    "human": "from langchain_community.chat_models.human import HumanInputChatModel",
    "hunyuan": "from langchain_community.chat_models.hunyuan import ChatHunyuan",
    "javelin_ai_gateway": "from langchain_community.chat_models.javelin_ai_gateway import ChatJavelinAIGateway",
    "jinachat": "from langchain_community.chat_models.jinachat import JinaChat",
    "kinetica": "from langchain_community.chat_models.kinetica import ChatKinetica",
    "konko": "from langchain_community.chat_models.konko import ChatOpenAI",
    "litellm": "from langchain_community.chat_models.litellm import ChatLiteLLM",
    "litellm_router": "from langchain_community.chat_models.litellm_router import ChatLiteLLMRouter",
    "llama_edge": "from langchain_community.chat_models.llama_edge import LlamaEdgeChatService",
    "llamacpp": "from langchain_community.chat_models.llamacpp import ChatLlamaCpp",
    "maritalk": "from langchain_community.chat_models.maritalk import ChatMaritalk",
    "minimax": "from langchain_community.chat_models.minimax import MiniMaxChat",
    "mlflow": "from langchain_community.chat_models.mlflow import ChatMlflow",
    "mlflow_ai_gateway": "from langchain_community.chat_models.mlflow_ai_gateway import ChatMLflowAIGateway",
    "mlx": "from langchain_community.chat_models.mlx import ChatMLX",
    "moonshot": "from langchain_community.chat_models.moonshot import MoonshotChat",
    "naver": "from langchain_community.chat_models.naver import ChatClovaX",
    "oci_data_science": "from langchain_community.chat_models.oci_data_science import ChatOCIModelDeploymentVLLM",
    "oci_generative_ai": "from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI",
    "octoai": "from langchain_community.chat_models.octoai import ChatOpenAI",
    "ollama": "from langchain_community.chat_models.ollama import ChatOllama",
    "openai": "from langchain_community.chat_models.openai import ChatOpenAI",
    "outlines": "from langchain_community.chat_models.outlines import ChatOutlines",
    "pai_eas_endpoint": "from langchain_community.chat_models.pai_eas_endpoint import PaiEasChatEndpoint",
    "perplexity": "from langchain_community.chat_models.perplexity import ChatPerplexity",
    "premai": "from langchain_community.chat_models.premai import ChatPremAI",
    "promptlayer_openai": "from langchain_community.chat_models.promptlayer_openai import PromptLayerChatOpenAI",
    "reka": "from langchain_community.chat_models.reka import ChatReka",
    "sambanova": "from langchain_community.chat_models.sambanova import ChatSambaStudio",
    "snowflake": "from langchain_community.chat_models.snowflake import ChatSnowflakeCortex",
    "solar": "from langchain_community.chat_models.solar import SolarChat",
    "sparkllm": "from langchain_community.chat_models.sparkllm import ChatSparkLLM",
    "symblai_nebula": "from langchain_community.chat_models.symblai_nebula import ChatNebula",
    "tongyi": "from langchain_community.chat_models.tongyi import ChatTongyi",
    "vertexai": "from langchain_community.chat_models.vertexai import ChatVertexAI",
    "volcengine_maas": "from langchain_community.chat_models.volcengine_maas import VolcEngineMaasChat",
    "writer": "from langchain_community.chat_models.writer import ChatWriter",
    "yandex": "from langchain_community.chat_models.yandex import ChatYandexGPT",
    "yi": "from langchain_community.chat_models.yi import ChatYi",
    "yuan2": "from langchain_community.chat_models.yuan2 import ChatYuan2",
    "zhipuai": "from langchain_community.chat_models.zhipuai import ChatZhipuAI"
}

class ChatModelProvider(Enum):
    ANTHROPIC = "anthropic"
    ANYSCALE = "anyscale"
    AZURE_OPENAI = "azure_openai"
    AZUREML_ENDPOINT = "azureml_endpoint"
    BAICHUAN = "baichuan"
    BAIDU_QIANFAN_ENDPOINT = "baidu_qianfan_endpoint"
    BEDROCK = "bedrock"
    CLOUDFLARE_WORKERSAI = "cloudflare_workersai"
    COHERE = "cohere"
    COZE = "coze"
    DAPPIER = "dappier"
    DATABRICKS = "databricks"
    DEEPINFRA = "deepinfra"
    EDENAI = "edenai"
    ERNIE = "ernie"
    EVERLYAI = "everlyai"
    FAKE = "fake"
    FIREWORKS = "fireworks"
    FRIENDLI = "friendli"
    GIGACHAT = "gigachat"
    GOOGLE_PALM = "google_palm"
    GPT_ROUTER = "gpt_router"
    HUGGINGFACE = "huggingface"
    HUMAN = "human"
    HUNYUAN = "hunyuan"
    JAVELIN_AI_GATEWAY = "javelin_ai_gateway"
    JINACHAT = "jinachat"
    KINETICA = "kinetica"
    KONKO = "konko"
    LITELLM = "litellm"
    LITELLM_ROUTER = "litellm_router"
    LLAMA_EDGE = "llama_edge"
    LLAMACPP = "llamacpp"
    MARITALK = "maritalk"
    MINIMAX = "minimax"
    MLFLOW = "mlflow"
    MLFLOW_AI_GATEWAY = "mlflow_ai_gateway"
    MLX = "mlx"
    MOONSHOT = "moonshot"
    NAVER = "naver"
    OCI_DATA_SCIENCE = "oci_data_science"
    OCI_GENERATIVE_AI = "oci_generative_ai"
    OCTOAI = "octoai"
    OLLAMA = "ollama"
    OPENAI = "openai"
    OUTLINES = "outlines"
    PAI_EAS_ENDPOINT = "pai_eas_endpoint"
    PERPLEXITY = "perplexity"
    PREMAI = "premai"
    PROMPTLAYER_OPENAI = "promptlayer_openai"
    REKA = "reka"
    SAMBANOVA = "sambanova"
    SNOWFLAKE = "snowflake"
    SOLAR = "solar"
    SPARKLLM = "sparkllm"
    SYMBLAI_NEBULA = "symblai_nebula"
    TONGYI = "tongyi"
    VERTEXAI = "vertexai"
    VOLCENGINE_MAAS = "volcengine_maas"
    WRITER = "writer"
    YANDEX = "yandex"
    YI = "yi"
    YUAN2 = "yuan2"
    ZHIPUAI = "zhipuai"