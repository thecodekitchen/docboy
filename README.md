# DocBoy
## A FastAPI Langchain/Langgraph Starter Pack

## Overview

This repository aims to provide starting points for a number of basic agentic architectures in LangGraph applications.
It provides patterns to expose agents on pipeline routes with built-in streaming functionality so that you can start assembling your graphs quickly.
It also currently provides some helper functions for document loading, particularly Python files.
A wider feature set is planned, but the current focus is on achieving stability and test coverage for the current functionality.

## Current Features

- **OIDC Auth**: Authenticate , identify, and authorize users for personalized pipelines. Supports generic OIDC configurations.
- **Kubernetes Backend Configs**: Deploy supporting tools flexibly in any Kubernetes environment, including locally.
- **Python Document Parsing**: Parse, split, embed, and store Python files in Postgres-backed vector collections
- **Easy ReAct Agent Streaming**: Provide a simple spec with a model and a set of tools, let our helper functions do the rest.
- **Automatic Document Awareness**: By default, if a collection name is provided to a ReAct agent, it will generate a retriever tool to query the collection in the Postgres vector store.

## Caveats

- **Work in Progress**: This project is still under development. Please do not yet attempt to implement this in production environments without thorough vetting and adjustment for your professional needs.
- **No Testing**: Features are not yet tested and could contain bugs. Tests are currently in development.
- **Documentation**: Comprehensive documentation is not yet available, but is on the roadmap once the full feature set is fleshed out.
- **Backend Requires Config**: Some important automations, such as migrations for the Postgres tables used by the store components, have yet to be implemented. 

## Getting Started

### Requirements To Run Backend Locally
- **Docker**: Must be installed locally
- **Kubernetes Driver**: I like [microk8s](https://microk8s.io/)

Instructions to run local backend:

1. [Install Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
2. Enable Microk8s GPU Addon
```
sudo microk8s enable gpu
```
3. Clone repo
```
git clone https://github.com/thecodekitchen/docboy.git
```
4. Apply the master manifest
```
cd docboy
# Use the provided script to create the database secrets. 
# Be sure to replace the placeholder arguments with secure values and note them elsewhere.
# If not using Keycloak, omit third argument
source ./k8s/create_secrets.sh <redis-password> <postgres-root-password> <keycloak-temporary-password>
# If not using Keycloak, run
sudo microk8s kubectl apply -f ./k8s/master_manifest.yml
# Otherwise, run
sudo microk8s kubectl apply -f ./k8s/master_manifest_full.yml
```
5. Run port forwards in separate terminals to expose services
```
sudo microk8s kubectl port-forward svc/postgres 5432:5432
sudo microk8s kubectl port-forward svc/redis 6379:6379
sudo microk8s kubectl port-forward svc/ollama 11434:11434
# If using keycloak
sudo microk8s kubectl port-forward svc/keycloak 80:5555
```
5.5. (If using Keycloak) Follow instructions [here](https://www.keycloak.org/getting-started/getting-started-docker#_log_in_to_the_admin_console) to set up client
- Make sure you uncheck "confidential client" and check both "standard flow" and "implicit flow" to enable the OIDC hybrid flow implemented here.
6. Fill out and rename .env.example to .env
- Fill out whichever values are relevant to your integration stack.
- Values prefixed with AZ are intended to denote Azure values, KC denotes Keycloak. Add other providers as needed.
7. Start the app and play with it
```
cd app
python3 -m venv docboy_venv
source ./docboy_venv/Scripts/activate
pip install -r requirements.txt
fastapi dev
```

Front end example coming soon!

## Contributing

Contributions are welcome! Please submit issues or pull requests to help improve the project.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.