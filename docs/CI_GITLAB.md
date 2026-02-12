# GitLab CI

Goals:

- same quality gates as GitHub Actions
- build and push Docker images to Docker Hub

## Required CI variables

- DOCKERHUB_USERNAME
- DOCKERHUB_TOKEN
- IMAGE_NAME_API
- IMAGE_NAME_UI

## Pipeline stages

stages:

- lint
- test
- build
- push

Notes:

- Use docker-in-docker (dind) or a runner with Docker available.
- For unit tests, do not require GPU.
- Keep inference tests optional unless using a self-hosted GPU runner.
