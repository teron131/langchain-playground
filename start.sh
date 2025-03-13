#!/bin/bash

cd deployment
docker compose down
langgraph build -t universal-chain
docker compose up