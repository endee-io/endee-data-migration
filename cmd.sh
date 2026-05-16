#!/bin/bash
docker network create vector-net 2>/dev/null || true
docker build --network=host -t vector-migration:latest .
docker compose up 
