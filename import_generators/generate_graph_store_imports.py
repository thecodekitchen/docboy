import pkgutil
import importlib
import inspect
import json

from langgraph.store.base import BaseStore
from langgraph import store

graph_store_imports = []

for _, name, _ in pkgutil.iter_modules(store.__path__):
    if not name.startswith('_') and not name.endswith('_utils'):
        try:
            # Import the module
            module = importlib.import_module(f"langgraph.store.{name}")
            # Inspect all classes in the module
            for class_name, class_obj in inspect.getmembers(module, inspect.isclass):
                # Check if this class is a subclass of BaseChatModel but not BaseChatModel itself
                if issubclass(class_obj, BaseStore) and class_obj != BaseStore:
                    import_statement = f"from langgraph.store.{name} import {class_name}"
                    if "Async" in class_name:
                        graph_store_imports.append((name+"_async", import_statement))
                    else:
                        graph_store_imports.append((name, import_statement))
        except (ImportError, AttributeError) as e:
            print(f"Could not process module {name}: {e}")

print(f"Found {len(graph_store_imports)} chat model classes:")
import_dict = {}
for class_name, import_statement in graph_store_imports:
    print(f"- {class_name}: {import_statement}")
    import_dict[class_name] = import_statement

with open("./import_maps/graph_store_imports.json", "w") as f:
    import_dict_str = json.dumps(import_dict, indent=4)
    f.write(import_dict_str)
    print(f"Imports saved to graph_store_imports.json")