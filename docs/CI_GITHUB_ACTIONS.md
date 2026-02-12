# GitHub Actions CI

Goals:

- lint + typecheck + unit tests
- build Docker images
- push images to Docker Hub on tags/releases

## Required secrets

- DOCKERHUB_USERNAME
- DOCKERHUB_TOKEN (or password/token)

## Recommended workflows

1. ci.yml (on push / PR)

- install deps
- ruff (lint)
- mypy (typecheck)
- pytest (unit tests)

2. docker.yml (on tags)

- docker build api image
- docker build ui image
- docker push to Docker Hub with version tags:
  - latest
  - vX.Y.Z
