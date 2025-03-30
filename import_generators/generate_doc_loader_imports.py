import pkgutil
import importlib
import inspect
import json
from langchain_community import document_loaders
from langchain_core.document_loaders import BaseLoader

doc_loader_imports = []

for _, name, _ in pkgutil.iter_modules(document_loaders.__path__):
    if not name.startswith('_') and not name.endswith('_utils'):
        try:
            # Import the module
            module = importlib.import_module(f"langchain_community.document_loaders.{name}")
            # Inspect all classes in the module
            for class_name, class_obj in inspect.getmembers(module, inspect.isclass):
                # Check if this class is a subclass of BaseChatModel but not BaseChatModel itself
                if issubclass(class_obj, BaseLoader) and class_obj != BaseLoader:
                    import_statement = f"from langchain_community.document_loaders.{name} import {class_name}"
                    doc_loader_imports.append((name, import_statement))
        except (ImportError, AttributeError) as e:
            print(f"Could not process module {name}: {e}")

print(f"Found {len(doc_loader_imports)} chat model classes:")
import_dict = {}
for class_name, import_statement in doc_loader_imports:
    print(f"- {class_name}: {import_statement}")
    import_dict[class_name] = import_statement

with open("./import_maps/doc_loader_imports.json", "w") as f:
    import_dict_str = json.dumps(import_dict, indent=4)
    f.write(import_dict_str)
    print(f"Imports saved to doc_loader_imports.json")