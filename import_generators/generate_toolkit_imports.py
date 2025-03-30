import pkgutil
import importlib
import inspect
import json
from langchain_core.tools.base import BaseToolkit
from langchain_community import agent_toolkits

toolkit_imports = []

for _, name, _ in pkgutil.iter_modules(agent_toolkits.__path__):
    if not name.startswith('_') and not name.endswith('_utils'):
        print(name)
        try:
            # Import the module
            module = importlib.import_module(f"langchain_community.agent_toolkits.{name}")
            is_submodule = False
            if hasattr(module, '__path__'):
                # If it's a package, we can use pkgutil to find submodules
                submodules = [name for _, name, is_pkg in pkgutil.iter_modules(module.__path__)]
                print("Submodules: ", submodules)
                if "toolkit" in submodules:
                    is_submodule = True
                    module = importlib.import_module(f"langchain_community.agent_toolkits.{name}.toolkit")
            
            # Inspect all classes in the module
            for class_name, class_obj in inspect.getmembers(module, inspect.isclass):
                # Check if this class is a subclass of BaseChatModel but not BaseChatModel itself
                if issubclass(class_obj, BaseToolkit) and class_obj != BaseToolkit:
                    import_statement = f"from langchain_community.agent_toolkits.{name} import {class_name}"
                    if is_submodule:
                        import_statement = f"from langchain_community.agent_toolkits.{name}.toolkit import {class_name}"
                    toolkit_imports.append((name, import_statement))
        except (ImportError, AttributeError) as e:
            print(f"Could not process module {name}: {e}")

print(f"Found {len(toolkit_imports)} toolkit classes:")
import_dict = {}
for class_name, import_statement in toolkit_imports:
    print(f"- {class_name}: {import_statement}")
    import_dict[class_name] = import_statement

with open("./import_maps/toolkit_imports.json", "w") as f:
    import_dict_str = json.dumps(import_dict, indent=4)
    f.write(import_dict_str)
    print(f"Imports saved to toolkit_imports.json")