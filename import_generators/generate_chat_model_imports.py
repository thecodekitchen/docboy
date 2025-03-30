import pkgutil
import importlib
import inspect
import json
from langchain_community import chat_models
from langchain.chat_models.base import BaseChatModel

# List to store valid chat model classes and their import paths
chat_model_imports = []

# Iterate through all modules in chat_models
for _, name, _ in pkgutil.iter_modules(chat_models.__path__):
    if not name.startswith('_') and not name.endswith('_utils'):
        try:
            # Import the module
            module = importlib.import_module(f"langchain_community.chat_models.{name}")
            # Inspect all classes in the module
            for class_name, class_obj in inspect.getmembers(module, inspect.isclass):
                # Check if this class is a subclass of BaseChatModel but not BaseChatModel itself
                if issubclass(class_obj, BaseChatModel) and class_obj != BaseChatModel:
                    import_statement = f"from langchain_community.chat_models.{name} import {class_name}"
                    chat_model_imports.append((name, import_statement))
        except (ImportError, AttributeError) as e:
            print(f"Could not process module {name}: {e}")

# Print the results
print(f"Found {len(chat_model_imports)} chat model classes:")
import_dict = {}
for class_name, import_statement in chat_model_imports:
    print(f"- {class_name}: {import_statement}")
    import_dict[class_name] = import_statement

with open("./import_maps/chat_model_imports.json", "w") as f:
    import_dict_str = json.dumps(import_dict, indent=4)
    f.write(import_dict_str)
    print(f"Imports saved to chat_model_imports.json")