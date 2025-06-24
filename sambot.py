
# The MIT License

# Copyright (c) 2025 Albert Murienne

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import logging
import numpy as np

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

from huggingface_hub import InferenceClient

from kokoro import KPipeline
# Or for onnx version:
# from kokoro_onnx import Kokoro

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# INITIALIZE MODELS

# Load Whisper model and processor
#modelcard="openai/whisper-tiny"
#modelcard="openai/whisper-base"
modelcard="openai/whisper-small"
processor = WhisperProcessor.from_pretrained(modelcard)
model = WhisperForConditionalGeneration.from_pretrained(modelcard)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="transcribe")

# Set up Hugging Face InferenceClient (for LLM like llama)
hf = InferenceClient(
    model="Qwen/Qwen2.5-14B-Instruct", 
    provider="featherless-ai",
    api_key=os.environ.get("HF_API_KEY"))  # remote LLM

# Load Kokoro
tts_pipeline = KPipeline(
    repo_id='hexgrad/Kokoro-82M',
    lang_code="f")  # french

# DEFINE CALLBACKS

def transcribe(audio_path):
    
    logging.info(f"audio path: {audio_path}")

    # load and resample local WAV file to 16kHz mono
    audio_array, sampling_rate = librosa.load(audio_path, sr=16000, mono=True)

    # process audio
    input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features

    # generate token ids
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    logging.info(f"transcription: {transcription[0]}")

    return transcription[0]

def chat_with_llm(query, history):
    # Prepare messages in OpenAI-style format
    messages = [{"role": "system", "content": "tu es un assistant francophone. Réponds en une phrase courte adaptée pour la synthèse vocale."}]
    for i, (role, content) in enumerate(history):
        messages.append({"role": "user" if role == "You" else "assistant", "content": content})
    messages.append({"role": "user", "content": query})

    logging.info(f"user queried: {query}")

    answer = hf.chat_completion(messages=messages, max_tokens=512).choices[0].message.content

    logging.info(f"bot answered: {answer}")

    return answer

def synthesize(text, voice="ff_siwis"):
    gen = tts_pipeline(text, voice=voice)
    _, _, audio = next(gen)
    # Convert to numpy if it's a tensor
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    elif not isinstance(audio, np.ndarray):
        audio = np.array(audio)

    logging.info(f"voice synthesis ready")
    
    return (24000, audio)

# BUILD THE GRADIO UI

import gradio as gr

with gr.Blocks() as demo:
    chatbot_state = gr.State(value=[])
    
    audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Speak")
    chat_out = gr.Chatbot(label="Conversation", type="messages")
    tts_player = gr.Audio(label="TTS Response", interactive=False, autoplay=True)
    
    def run_step(audio_path, chat_history):
        user_text = transcribe(audio_path)  # now using faster-whisper
        chat_history.append(("You", user_text))

        # LLM and TTS logic unchanged:
        bot_text = chat_with_llm(user_text, chat_history)
        chat_history.append(("Bot", bot_text))
        audio_tuple = synthesize(bot_text)

        # Convert to Gradio format: list of dicts with 'role' and 'content'
        gradio_history = [
            {"role": "user" if role == "You" else "assistant", "content": content}
            for role, content in chat_history
        ]

        return gradio_history, audio_tuple
    
    audio.upload(
        run_step,
        inputs=[audio, chatbot_state],
        outputs=[chat_out, tts_player],
    )

    audio.stop_recording(
        run_step,
        inputs=[audio, chatbot_state],
        outputs=[chat_out, tts_player],
    )
    
demo.launch()
