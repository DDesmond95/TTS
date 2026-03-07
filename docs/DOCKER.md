# 🐳 Docker Deployment

OmniVoice Studio provides a flexible Docker architecture supporting both lightweight development and production-ready "all-in-one" deployments.

## 🚀 Image Variants

We maintain two primary variants for the API:

1.  **Lite (`:lite` / `:latest`)**: Contains the engine code and dependencies. Requires you to mount your `models/` directory externally.
2.  **Full (`:full`)**: A complete image with all AI weights pre-loaded. Perfect for stateless cloud deployments where you don't want to manage external storage.

## 🛠️ Docker Compose (Local Development)

The easiest way to run OmniVoice Studio locally is using `docker-compose`. Ensure you have an `.env` file configured.

```bash
cd docker
docker-compose up -d
```

This will start:

- **API**: On port `8001` (Lite mode, mounting local `models/`).
- **UI**: On port `7860`.

## 🔋 GPU Support

To utilize high-speed neural synthesis, you **must** have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on your host.

The `docker-compose.yml` is pre-configured to reserve all available GPUs.

## 📥 Building Images Manually

### Build Lite API (Default)

```bash
docker build -t omnivoice-studio-api:lite -f docker/Dockerfile.api .
```

### Build Full API (Pre-loaded Models)

```bash
docker build --build-arg PRELOAD_MODELS=true -t omnivoice-studio-api:full -f docker/Dockerfile.api .
```

## 💓 Health & Monitoring

Both the API and UI images include built-in `HEALTHCHECK` instructions. orchestration tools like Kubernetes or Docker Swarm will automatically detect if the service is ready or struggling:

- **API Health**: `curl -f http://localhost:8001/health`
- **UI Health**: `curl -f http://localhost:7860/`

## 🎙️ VTuber & Host Audio Note

When running in Docker, it is recommended to run the **Streaming Bridge** (Python/Batch) on your **Host OS** while the API runs in Docker. This simplifies routing audio to virtual cables (VAC) for OBS or VSeeFace.
