# Graph Report - .  (2026-08-24)

## Corpus Check
- 56 files · ~0 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 619 nodes · 1373 edges · 32 communities (24 shown, 8 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Community 30
- Community 31

## God Nodes (most connected - your core abstractions)
1. `TTSEngine` - 76 edges
2. `RunResult` - 48 edges
3. `UIState` - 40 edges
4. `get_engine()` - 26 edges
5. `TTSTaskRunnerMixin` - 22 edges
6. `create_ui()` - 22 edges
7. `RunResponse` - 18 edges
8. `Task` - 18 edges
9. `CustomVoiceRequest` - 18 edges
10. `VoiceProfile` - 17 edges

## Surprising Connections (you probably didn't know these)
- `create_app()` --references--> `AppConfig`  [EXTRACTED]
  src/omnivoice_studio/api/app.py → src/omnivoice_studio/config.py
- `cmd_run_api()` --calls--> `create_app()`  [EXTRACTED]
  src/omnivoice_studio/cli.py → src/omnivoice_studio/api/app.py
- `get_voice_route()` --references--> `VoiceProfile`  [EXTRACTED]
  src/omnivoice_studio/api/http_routes.py → src/omnivoice_studio/voices/schema.py
- `save_voice_route()` --references--> `VoiceProfile`  [EXTRACTED]
  src/omnivoice_studio/api/http_routes.py → src/omnivoice_studio/voices/schema.py
- `tts_custom_voice()` --references--> `CustomVoiceRequest`  [EXTRACTED]
  src/omnivoice_studio/api/http_routes.py → src/omnivoice_studio/tasks/custom_voice.py

## Import Cycles
- None detected.

## Communities (32 total, 8 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (82): delete, Exception, FastAPI, FileResponse, get, post, create_app(), Path (+74 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (41): Blocks, AppConfig, Configuration management for OmniVoice Studio, supporting YAML and ENV…, Root configuration object containing all sub-configs., Encodes audio using a tokenizer model. Args: audio: Path to the audio file.…, Decodes codes using a tokenizer model. Args: codes_json_path: Path to the JSON…, Any, BaseModel (+33 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (29): Core orchestration engine for OmniVoice Studio TTS models., Retrieves a voice profile by its ID., create_voices_page(), Voice management package., CloneConfig, DesignTemplateConfig, BaseModel, Metadata schema for voice profiles in OmniVoice Studio. (+21 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (40): Namespace, cmd_convert(), cmd_download_models(), cmd_list_models(), cmd_list_voices(), cmd_run_api(), cmd_run_ui(), cmd_synthesize() (+32 more)

### Community 4 - "Community 4"
Cohesion: 0.13
Nodes (23): ABC, Pydantic schemas for OmniVoice Studio API requests and responses., Mixin for TTS task dispatching in OmniVoice Studio., InferenceError, Raised when an inference task fails, Base utilities for synthesis pipelines in OmniVoice Studio., Long-form text-to-speech pipeline for OmniVoice Studio., Script reading pipeline for dialogue-based multi-speaker synthesis. (+15 more)

### Community 5 - "Community 5"
Cohesion: 0.08
Nodes (20): dtype, Loaded, Any, Path, Lists all discovered models and their metadata., Lists all available voice profiles., Saves or updates a voice profile., Infers the best torch device and dtype from config. (+12 more)

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (26): Runs a long-form synthesis pipeline. Args: text: Large text to synthesize.…, Runs a script reading pipeline (multi-speaker dialogue). Args: script_text:…, Runs an audiobook generation pipeline. Args: chapter_paths: List of paths to…, AudiobookPipeline, AudiobookRequest, Any, Audiobook generation pipeline for OmniVoice Studio., Configuration for an audiobook generation run. (+18 more)

### Community 7 - "Community 7"
Cohesion: 0.12
Nodes (16): Configuration for the model inference runtime., Returns the appropriate torch dtype string., RuntimeConfig, Initializes the TTSEngine., OutputStore, Any, ndarray, Path (+8 more)

### Community 8 - "Community 8"
Cohesion: 0.10
Nodes (19): Streams a voice design task. Args: text: Text to synthesize. language: Target…, DesignThenCloneRequest, DesignThenCloneTask, Any, BaseModel, Request schema for design-then-clone workflow., Executes a two-step generation: first design a voice, then clone it immediately., Validate the incoming request parameters. (+11 more)

### Community 9 - "Community 9"
Cohesion: 0.17
Nodes (19): get_engine(), Any, Depends, ndarray, WebSocket routes for real-time TTS streaming., WebSocket endpoint for custom voice streaming., WebSocket endpoint for voice design streaming., WebSocket endpoint for voice clone streaming. (+11 more)

### Community 10 - "Community 10"
Cohesion: 0.12
Nodes (15): ndarray, Streams a custom voice generation task. Args: text: Text to synthesize.…, Runs a custom voice generation task. Args: text: Text to synthesize. language:…, Any, Executes the subtitle pipeline. Args: engine: The TTSEngine instance. request:…, CustomVoiceRequest, CustomVoiceTask, Any (+7 more)

### Community 11 - "Community 11"
Cohesion: 0.14
Nodes (12): Any, Performs voice design and then clones that voice for a text synthesis. Args:…, Runs a voice conversion task. Args: source_audio: Path to source audio.…, Runs a singing synthesis task. Args: lyrics: Lyrics text. score: Musical score…, Runs a voice sculpting task (zero-shot editing). Args: instruction: Text…, Runs a batch NPC lines generation pipeline. Args: csv_path: Path to CSV with…, Mixin providing high-level run and stream methods for the TTSEngine., Runs a subtitle dubbing pipeline. Args: srt_path: Path to the SRT file.… (+4 more)

### Community 12 - "Community 12"
Cohesion: 0.14
Nodes (12): PipelineHelper, Any, Prepares a task and its request based on a speaker configuration., Helper methods for pipelines to avoid code duplication., Simple sentence-based chunking., Helper to save artifacts and finalize a run., Runs a task and loads its audio output into memory., Any (+4 more)

### Community 13 - "Community 13"
Cohesion: 0.15
Nodes (12): Runs a voice cloning task. Args: text: Text to synthesize. language: Target…, Streams a voice cloning task. Args: text: Text to synthesize. language: Target…, Any, BaseModel, ndarray, Streaming for voice cloning (sentence-by-sentence)., Request schema for voice clone generation., Generates audio using Qwen3-TTS in 'voice_clone' mode. (+4 more)

### Community 14 - "Community 14"
Cohesion: 0.16
Nodes (11): Model management package for OmniVoice Studio., ModelInfo, ModelRegistry, Path, Model registry for managing and categorizing local model files., Contains metadata about a local model., Simple local registry: scans models_dir for subfolders and categorizes by name., Initializes the ModelRegistry. Args: models_dir: The directory containing local… (+3 more)

### Community 15 - "Community 15"
Cohesion: 0.14
Nodes (10): RequestT, ResultT, Any, ndarray, Path, Validate the request., Execute the task (non-streaming)., Execute the task (streaming) - optional. (+2 more)

### Community 16 - "Community 16"
Cohesion: 0.19
Nodes (11): NPCLine, NPCPackPipeline, NPCPackRequest, Any, BaseModel, NPC pack pipeline for batch vocal performance generation., Reads dialogue lines from a CSV file., Represents a single dialogue line for an NPC. (+3 more)

### Community 17 - "Community 17"
Cohesion: 0.22
Nodes (8): Any, BaseModel, Request schema for MeanVC voice conversion., Voice conversion using MeanVC models., Validate the incoming request parameters., Execute the voice conversion task., VoiceConversionRequest, VoiceConversionTask

### Community 18 - "Community 18"
Cohesion: 0.22
Nodes (8): Any, BaseModel, Request schema for singing synthesis., Singing synthesis using TCSinger models., Validate the incoming request parameters., Execute the singing synthesis task., SingingSynthesisRequest, SingingSynthesisTask

### Community 19 - "Community 19"
Cohesion: 0.20
Nodes (9): Test script for the OmniVoice Studio HTTP API., Tests the /voices endpoint to list available voice profiles., Tests the /tts/custom_voice endpoint for custom voice generation., Tests the /tts/voice_design endpoint for voice design generation., Tests the /tts/voice_clone endpoint., test_generate_clone(), test_generate_custom(), test_generate_design() (+1 more)

### Community 20 - "Community 20"
Cohesion: 0.25
Nodes (6): Any, BaseModel, Request schema for voice sculpting., Validate the incoming request parameters., Execute the voice sculpting task., VoiceSculptRequest

### Community 21 - "Community 21"
Cohesion: 0.29
Nodes (5): Paths, Path, Filesystem path management for OmniVoice Studio., Container for essential application paths., Creates a Paths instance from configuration strings, resolving relative paths.…

### Community 22 - "Community 22"
Cohesion: 0.40
Nodes (4): BaseModel, Represents a single line from a script with a speaker tag., Parses a text script into individual speaker rows., ScriptRow

### Community 23 - "Community 23"
Cohesion: 0.40
Nodes (4): Caption, BaseModel, Parses an SRT file into a list of Caption objects., Represents a single subtitle caption with timing and text.

## Knowledge Gaps
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TTSEngine` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 8`, `Community 9`, `Community 10`, `Community 11`, `Community 13`, `Community 25`?**
  _High betweenness centrality (0.281) - this node is a cross-community bridge._
- **Why does `RunResult` connect `Community 4` to `Community 1`, `Community 2`, `Community 6`, `Community 7`, `Community 8`, `Community 10`, `Community 11`, `Community 12`, `Community 13`, `Community 16`, `Community 17`, `Community 18`, `Community 20`?**
  _High betweenness centrality (0.135) - this node is a cross-community bridge._
- **Why does `Task` connect `Community 4` to `Community 24`, `Community 15`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.05399625768511093 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.060882800608828 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.07716701902748414 - nodes in this community are weakly interconnected._
- **Should `Community 3` be split into smaller, more focused modules?**
  _Cohesion score 0.07665505226480836 - nodes in this community are weakly interconnected._