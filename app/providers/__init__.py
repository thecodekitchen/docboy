from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.document_loaders import BaseLoader
from langchain_core.vectorstores import VectorStore
from langchain_core.stores import BaseStore as ChainStore
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools.base import BaseToolkit
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore as GraphStore
from typing import Callable

import importlib

from chat_model import *
from doc_loader import *
from checkpoint_saver import *
from vector_store import *
from embeddings import *
from llm import *
from retriever import *
from graph_store import *
from chain_store import *
from toolkit import *

def create_lazy_loader(import_statement)->Callable:
    """Create a function that will import the class when called."""
    def loader():
        parts = import_statement.split(' import ')
        module_path = parts[0].replace('from ', '')
        class_name = parts[1]
        
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    
    return loader

def get_doc_loader_class_instance(
    type: DocumentLoaderType, 
    **loader_kwargs
) -> BaseLoader:
    """
    Get an instance of a document loader class for a given file type.
    
    Args:
        type: The document loader type enum value
        **loader_kwargs: Additional keyword arguments to pass to the loader constructor
        
    Returns:
        An instance of the BaseLoader class from langchain_core.document_loaders
        
    Raises:
        ValueError: If the type is not supported
        ImportError: If there's an issue importing the type module
    """
    type_str = type.value
    
    if type_str not in doc_loader_imports:
        available = list(doc_loader_imports.keys())
        raise ValueError(
            f"type '{type_str}' not supported. Available types: {available}"
        )
    
    try:
        # Get the class using the lazy loader
        loader_class = create_lazy_loader(doc_loader_imports[type_str])()
        
        # Create and return an instance with the provided kwargs
        return loader_class(**loader_kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import {type_str} module: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error creating {type_str} loader instance: {str(e)}")

def get_chat_model_class_instance(
    provider: ChatModelProvider, 
    **model_kwargs
) -> BaseChatModel:
    """
    Get an instance of a chat model class for the given provider.
    
    Args:
        provider: The chat model provider enum value
        **model_kwargs: Additional keyword arguments to pass to the model constructor
        
    Returns:
        An instance of the BaseChatModel class from langchain_core.language_models.chat_models
        
    Raises:
        ValueError: If the provider is not supported
        ImportError: If there's an issue importing the provider module
    """
    provider_str = provider.value
    
    if provider_str not in chat_model_imports:
        available = list(chat_model_imports.keys())
        raise ValueError(
            f"Provider '{provider_str}' not supported. Available providers: {available}"
        )
    
    try:
        # Get the class using the lazy loader
        model_class = create_lazy_loader(chat_model_imports[provider_str])()
        
        # Create and return an instance with the provided kwargs
        return model_class(**model_kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import {provider_str} module: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error creating {provider_str} model instance: {str(e)}")

def get_checkpoint_saver_class_instance(
    provider: CheckpointSaverProvider, 
    **saver_kwargs
) -> BaseCheckpointSaver:
    """
    Get an instance of a checkpoint saver class for the given provider.
    
    Args:
        provider: The checkpoint saver provider enum value
        **saver_kwargs: Additional keyword arguments to pass to the saver constructor
        
    Returns:
        An instance of the BaseCheckpointSaver class from langgraph.checkpoint
        
    Raises:
        ValueError: If the provider is not supported
        ImportError: If there's an issue importing the provider module
    """
    provider_str = provider.value
    
    if provider_str not in checkpoint_saver_imports:
        available = list(checkpoint_saver_imports.keys())
        raise ValueError(
            f"Provider '{provider_str}' not supported. Available providers: {available}"
        )
    
    try:
        # Get the class using the lazy loader
        saver_class = create_lazy_loader(checkpoint_saver_imports[provider_str])()
        
        # Create and return an instance with the provided kwargs
        return saver_class(**saver_kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import {provider_str} module: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error creating {provider_str} saver instance: {str(e)}")
    
def get_vector_store_class_instance(
    provider: VectorStoreProvider, 
    **vector_store_kwargs
) -> VectorStore:
    """
    Get an instance of a vector store class for the given provider.
    
    Args:
        provider: The vector store provider enum value
        **vector_store_kwargs: Additional keyword arguments to pass to the vector store constructor
        
    Returns:
        An instance of the VectorStore class from langchain_core.vectorstores
        
    Raises:
        ValueError: If the provider is not supported
        ImportError: If there's an issue importing the provider module
    """
    provider_str = provider.value
    
    if provider_str not in vector_store_imports:
        available = list(vector_store_imports.keys())
        raise ValueError(
            f"Provider '{provider_str}' not supported. Available providers: {available}"
        )
    
    try:
        # Get the class using the lazy loader
        vector_store_class = create_lazy_loader(vector_store_imports[provider_str])()
        
        # Create and return an instance with the provided kwargs
        return vector_store_class(**vector_store_kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import {provider_str} module: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error creating {provider_str} vector store instance: {str(e)}")
    
def get_embeddings_class_instance(
    provider: VectorStoreProvider, 
    **embeddings_kwargs
) -> Embeddings:
    """
    Get an instance of a vector store class for the given provider.
    
    Args:
        provider: The vector store provider enum value
        **embeddings_kwargs: Additional keyword arguments to pass to the vector store constructor
        
    Returns:
        An instance of the Embeddings class from langchain_core.embeddings
        
    Raises:
        ValueError: If the provider is not supported
        ImportError: If there's an issue importing the provider module
    """
    provider_str = provider.value
    
    if provider_str not in embeddings_imports:
        available = list(embeddings_imports.keys())
        raise ValueError(
            f"Provider '{provider_str}' not supported. Available providers: {available}"
        )
    
    try:
        # Get the class using the lazy loader
        embeddings_class = create_lazy_loader(embeddings_imports[provider_str])()
        
        # Create and return an instance with the provided kwargs
        return embeddings_class(**embeddings_kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import {provider_str} module: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error creating {provider_str} vector store instance: {str(e)}")
    
def get_llm_class_instance(
    provider: VectorStoreProvider, 
    **llm_kwargs
) -> BaseLLM:
    """
    Get an instance of a vector store class for the given provider.
    
    Args:
        provider: The vector store provider enum value
        **llm_kwargs: Additional keyword arguments to pass to the vector store constructor
        
    Returns:
        An instance of the BaseLLM class from langchain_core.language_models.llms
        
    Raises:
        ValueError: If the provider is not supported
        ImportError: If there's an issue importing the provider module
    """
    provider_str = provider.value
    
    if provider_str not in llm_imports:
        available = list(llm_imports.keys())
        raise ValueError(
            f"Provider '{provider_str}' not supported. Available providers: {available}"
        )
    
    try:
        # Get the class using the lazy loader
        llm_class = create_lazy_loader(llm_imports[provider_str])()
        
        # Create and return an instance with the provided kwargs
        return llm_class(**llm_kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import {provider_str} module: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error creating {provider_str} vector store instance: {str(e)}")
    
def get_retriever_class_instance(
    provider: VectorStoreProvider, 
    **retriever_kwargs
) -> BaseRetriever:
    """
    Get an instance of a vector store class for the given provider.
    
    Args:
        provider: The vector store provider enum value
        **retriever_kwargs: Additional keyword arguments to pass to the vector store constructor
        
    Returns:
        An instance of the BaseRetriever class from langchain_core.retrievers
        
    Raises:
        ValueError: If the provider is not supported
        ImportError: If there's an issue importing the provider module
    """
    provider_str = provider.value
    
    if provider_str not in retriever_imports:
        available = list(retriever_imports.keys())
        raise ValueError(
            f"Provider '{provider_str}' not supported. Available providers: {available}"
        )
    
    try:
        # Get the class using the lazy loader
        retriever_class = create_lazy_loader(retriever_imports[provider_str])()
        
        # Create and return an instance with the provided kwargs
        return retriever_class(**retriever_kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import {provider_str} module: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error creating {provider_str} vector store instance: {str(e)}")
    
def get_graph_store_class_instance(
    provider: GraphStoreProvider, 
    **graph_store_kwargs
) -> GraphStore:
    """
    Get an instance of a graph store class for the given provider.
    
    Args:
        provider: The graph store provider enum value
        **graph_store_kwargs: Additional keyword arguments to pass to the graph store constructor
        
    Returns:
        An instance of the GraphStore class from langgraph.store.base
        
    Raises:
        ValueError: If the provider is not supported
        ImportError: If there's an issue importing the provider module
    """
    provider_str = provider.value
    
    if provider_str not in graph_store_imports:
        available = list(graph_store_imports.keys())
        raise ValueError(
            f"Provider '{provider_str}' not supported. Available providers: {available}"
        )
    
    try:
        # Get the class using the lazy loader
        graph_store_class = create_lazy_loader(graph_store_imports[provider_str])()
        
        # Create and return an instance with the provided kwargs
        return graph_store_class(**graph_store_kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import {provider_str} module: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error creating {provider_str} graph store instance: {str(e)}")
    
def get_chain_store_class_instance(
    provider: ChainStoreProvider, 
    **chain_store_kwargs
) -> ChainStore:
    """
    Get an instance of a chain store class for the given provider.
    
    Args:
        provider: The chain store provider enum value
        **chain_store_kwargs: Additional keyword arguments to pass to the chain store constructor
        
    Returns:
        An instance of the ChainStore class from langchain_core.stores
        
    Raises:
        ValueError: If the provider is not supported
        ImportError: If there's an issue importing the provider module
    """
    provider_str = provider.value
    
    if provider_str not in chain_store_imports:
        available = list(chain_store_imports.keys())
        raise ValueError(
            f"Provider '{provider_str}' not supported. Available providers: {available}"
        )
    
    try:
        # Get the class using the lazy loader
        chain_store_class = create_lazy_loader(chain_store_imports[provider_str])()
        
        # Create and return an instance with the provided kwargs
        return chain_store_class(**chain_store_kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import {provider_str} module: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error creating {provider_str} graph store instance: {str(e)}"
)
    
def get_toolkit_class_instance(
    provider: Toolkit, 
    **toolkit_kwargs
) -> BaseToolkit:
    """
    Get an instance of a toolkit class for the given provider.
    
    Args:
        provider: The toolkit provider enum value
        **toolkit_kwargs: Additional keyword arguments to pass to the toolkit constructor
        
    Returns:
        An instance of the BaseToolkit class from langchain_core.tools.base
        
    Raises:
        ValueError: If the provider is not supported
        ImportError: If there's an issue importing the provider module
    """
    provider_str = provider.value
    
    if provider_str not in toolkit_imports:
        available = list(toolkit_imports.keys())
        raise ValueError(
            f"Provider '{provider_str}' not supported. Available providers: {available}"
        )
    
    try:
        # Get the class using the lazy loader
        toolkit_class = create_lazy_loader(toolkit_imports[provider_str])()
        
        # Create and return an instance with the provided kwargs
        return toolkit_class(**toolkit_kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import {provider_str} module: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error creating {provider_str} graph store instance: {str(e)}")