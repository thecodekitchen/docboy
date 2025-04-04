from enum import Enum

text_splitter_imports = {
    "TextSplitter": "from langchain_text_splitters.base import TextSplitter",
    "TokenTextSplitter": "from langchain_text_splitters.base import TokenTextSplitter",
    "CharacterTextSplitter": "from langchain_text_splitters.character import CharacterTextSplitter",
    "RecursiveCharacterTextSplitter": "from langchain_text_splitters.python import RecursiveCharacterTextSplitter",
    "HTMLHeaderTextSplitter": "from langchain_text_splitters.html import HTMLHeaderTextSplitter",
    "HTMLSectionSplitter": "from langchain_text_splitters.html import HTMLSectionSplitter",
    "HTMLSemanticPreservingSplitter": "from langchain_text_splitters.html import HTMLSemanticPreservingSplitter",
    "RecursiveJsonSplitter": "from langchain_text_splitters.json import RecursiveJsonSplitter",
    "JSFrameworkTextSplitter": "from langchain_text_splitters.jsx import JSFrameworkTextSplitter",
    "KonlpyTextSplitter": "from langchain_text_splitters.konlpy import KonlpyTextSplitter",
    "LatexTextSplitter": "from langchain_text_splitters.latex import LatexTextSplitter",
    "ExperimentalMarkdownSyntaxTextSplitter": "from langchain_text_splitters.markdown import ExperimentalMarkdownSyntaxTextSplitter",
    "MarkdownHeaderTextSplitter": "from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter",
    "MarkdownTextSplitter": "from langchain_text_splitters.markdown import MarkdownTextSplitter",
    "NLTKTextSplitter": "from langchain_text_splitters.nltk import NLTKTextSplitter",
    "PythonCodeTextSplitter": "from langchain_text_splitters.python import PythonCodeTextSplitter",
    "SentenceTransformersTokenTextSplitter": "from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter",
    "SpacyTextSplitter": "from langchain_text_splitters.spacy import SpacyTextSplitter"
}

class TextSplitterType(Enum):
    TextSplitter = "TextSplitter"
    TokenTextSplitter = "TokenTextSplitter"
    CharacterTextSplitter = "CharacterTextSplitter"
    RecursiveCharacterTextSplitter = "RecursiveCharacterTextSplitter"
    HTMLHeaderTextSplitter = "HTMLHeaderTextSplitter"
    HTMLSectionSplitter = "HTMLSectionSplitter"
    HTMLSemanticPreservingSplitter = "HTMLSemanticPreservingSplitter"
    RecursiveJsonSplitter = "RecursiveJsonSplitter"
    JSFrameworkTextSplitter = "JSFrameworkTextSplitter"
    KonlpyTextSplitter = "KonlpyTextSplitter"
    LatexTextSplitter = "LatexTextSplitter"
    ExperimentalMarkdownSyntaxTextSplitter = "ExperimentalMarkdownSyntaxTextSplitter"
    MarkdownHeaderTextSplitter = "MarkdownHeaderTextSplitter"
    MarkdownTextSplitter = "MarkdownTextSplitter"
    NLTKTextSplitter = "NLTKTextSplitter"
    PythonCodeTextSplitter = "PythonCodeTextSplitter"
    SentenceTransformersTokenTextSplitter = "SentenceTransformersTokenTextSplitter"
    SpacyTextSplitter = "SpacyTextSplitter"
    # Add more text splitters as needed