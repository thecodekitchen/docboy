import pkgutil
import importlib
import inspect
import json
from langchain_core.retrievers import BaseRetriever
from langchain_community import retrievers

retriever_imports = []

for _, name, _ in pkgutil.iter_modules(retrievers.__path__):
    if not name.startswith('_') and not name.endswith('_utils'):
        try:
            # Import the module
            module = importlib.import_module(f"langchain_community.retrievers.{name}")
            # Inspect all classes in the module
            for class_name, class_obj in inspect.getmembers(module, inspect.isclass):
                # Check if this class is a subclass of BaseChatModel but not BaseChatModel itself
                if issubclass(class_obj, BaseRetriever) and class_obj != BaseRetriever:
                    import_statement = f"from langchain_community.retrievers.{name} import {class_name}"
                    retriever_imports.append((name, import_statement))
        except (ImportError, AttributeError) as e:
            print(f"Could not process module {name}: {e}")

print(f"Found {len(retriever_imports)} retriever classes:")
import_dict = {}
for class_name, import_statement in retriever_imports:
    print(f"- {class_name}: {import_statement}")
    import_dict[class_name] = import_statement

with open("./import_maps/retriever_imports.json", "w") as f:
    import_dict_str = json.dumps(import_dict, indent=4)
    f.write(import_dict_str)
    print(f"Imports saved to retriever_imports.json")