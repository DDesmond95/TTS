# OmniVoice Studio 🎙️✨

**Your Complete AI Voice Production Studio**

OmniVoice Studio is a professional-grade, local voice AI platform supporting **Qwen3-TTS**, **TCSinger2**, **VoiceSculptor**, and **MeanVC**. Designed for creators and developers, it provides studio-quality voice synthesis, cloning, and conversion with complete privacy and local control.

## ✨ Key Features

- **Multi-Model Support**: Qwen3-TTS (TTS/Clone), TCSinger2 (Singing), VoiceSculptor (Design/Edit), MeanVC (Conversion).
- **Professional Quality**: Studio-grade audio output with advanced processing.
- **Unified Interface**: Access all features via Python API, HTTP REST API, WebSocket Streaming, or the Web Studio UI.
- **Local Control**: All processing is done locally on your hardware.
- **Streaming Ready**: Ultra-low latency streaming for VTubers and real-time interactive apps.

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/OmniVoiceStudio/omnivoice-studio.git
cd omnivoice-studio
pip install -e .

# Run the Studio UI
omnivoice run-ui
```

## 📚 Documentation

Explore our comprehensive documentation in the `docs/` folder:

- **[Quickstart Guide](docs/QUICKSTART.md)** - Get up and running in minutes.
- **[Models Overview](docs/MODELS.md)** - Learn about the supported AI models.
- **[API Reference](docs/API.md)** - Integrate OmniVoice into your own apps.
- **[VTuber Integration](docs/VTUBER.md)** - Setup for OBS and live streaming.
- **[Documentation Index](docs/README.md)** - See all guides.

## 🧹 Code Quality

We maintain high standards for our codebase. Developers should use the following commands regularly:

- `make lint` / `make format`: Quick linting with Ruff.
- `make pylint`: Static analysis of code health.
- `make pylint-report`: Generate a Pylint report to a file.
- `make type`: Check types with MyPy.
- `make test`: Run the test suite.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**OmniVoice Studio** - Where Your Voice Comes to Life 🎙️✨
