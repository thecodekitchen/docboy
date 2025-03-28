#!/bin/bash
kubectl create secret generic redis-secret --from-literal=redis-password="$1"
kubectl create secret generic postgres-secret --from-literal=postgres-password="$2"
if [ -n "$3" ]; then
    kubectl create secret generic keycloak-secret --from-literal=keycloak-password="$3"
fi