# BitNet Infrastructure Assets

This directory collects reproducible container definitions and infrastructure-as-code
artifacts that align with the guidance in `docs/deployment.md`.

- `docker/bitnet-runtime.Dockerfile` builds a production-ready image containing the
  compiled llama.cpp binaries and the packaged Python runtime helpers.  The Dockerfile
  is reproducible thanks to pinned base images and an isolated build stage that
  produces wheels before copying artefacts into the final image.
- `docker/bitnet-edge.Dockerfile` targets edge deployments where a Python virtual
  environment is required for offline upgrades.  It follows the same multi-stage
  pattern but keeps the venv intact for device-level package management.
- `terraform/` holds Kubernetes manifests expressed in Terraform, providing a
  turnkey deployment that configures namespaces, secrets, ConfigMaps and a
  deployment + service pairing.

These assets are exercised by CI to guarantee that container builds and Terraform
formatting remain stable.
