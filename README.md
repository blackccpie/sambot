# sambot

Running an audio chatbot on a 2Gb VRAM GPU

The chatbot is based on a 3 steps pipeline:

* STT using [Whisper-small](https://huggingface.co/openai/whisper-small) model
* LLM interaction through [HuggingFace Inference API](https://huggingface.co/docs/inference-providers/providers/hf-inference)
* TTS using [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)

The UI is made using Gradio, with automatic VAD managed on the frontend using [vad-web](https://github.com/ricky0123/vad).