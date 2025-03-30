import pkgutil
import importlib
import inspect
import json
from langchain_core.embeddings import Embeddings
from langchain_community import embeddings

embeddings_imports = []

for _, name, _ in pkgutil.iter_modules(embeddings.__path__):
    if not name.startswith('_') and not name.endswith('_utils'):
        try:
            # Import the module
            module = importlib.import_module(f"langchain_community.embeddings.{name}")
            # Inspect all classes in the module
            for class_name, class_obj in inspect.getmembers(module, inspect.isclass):
                # Check if this class is a subclass of BaseChatModel but not BaseChatModel itself
                if issubclass(class_obj, Embeddings) and class_obj != Embeddings:
                    import_statement = f"from langchain_community.embeddings.{name} import {class_name}"
                    embeddings_imports.append((name, import_statement))
        except (ImportError, AttributeError) as e:
            print(f"Could not process module {name}: {e}")

print(f"Found {len(embeddings_imports)} chat model classes:")
import_dict = {}
for class_name, import_statement in embeddings_imports:
    print(f"- {class_name}: {import_statement}")
    import_dict[class_name] = import_statement

with open("./import_maps/embeddings_imports.json", "w") as f:
    import_dict_str = json.dumps(import_dict, indent=4)
    f.write(import_dict_str)
    print(f"Imports saved to embeddings_imports.json")