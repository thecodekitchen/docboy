import pkgutil
import importlib
import inspect
import json
import langchain_text_splitters
text_splitter_imports = {
    "TextSplitter": "from langchain_text_splitters.base import TextSplitter",
}

for _, name, _ in pkgutil.iter_modules(langchain_text_splitters.__path__):
    if not name.startswith('_') and not name.endswith('_utils'):
        try:
            # Import the module
            module = importlib.import_module(f"langchain_text_splitters.{name}")
            # Inspect all classes in the module
            for class_name, class_obj in inspect.getmembers(module, inspect.isclass):
                # Check if this class is a subclass of BaseChatModel but not BaseChatModel itself
                if "Splitter" in class_name and class_name != "TextSplitter":
                    import_statement = f"from langchain_text_splitters.{name} import {class_name}"
                    text_splitter_imports[class_name] = import_statement

        except (ImportError, AttributeError) as e:
            print(f"Could not process module {name}: {e}")

print(f"Found {len(text_splitter_imports)} text splitter classes:")


with open("./import_maps/text_splitter_imports.json", "w") as f:
    import_dict_str = json.dumps(text_splitter_imports, indent=4)
    f.write(import_dict_str)
    print(f"Imports saved to text_splitter_imports.json")