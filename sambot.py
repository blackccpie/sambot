
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

import argparse
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
    #model="Qwen/Qwen2.5-14B-Instruct", 
    #provider="featherless-ai",
    model="meta-llama/Llama-3.3-70B-Instruct",
    #model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    provider="hf-inference",
    api_key=os.environ.get("HF_API_KEY"))  # remote LLM

# Load Kokoro
tts_pipeline = KPipeline(
    repo_id='hexgrad/Kokoro-82M',
    lang_code="f")  # french

# DEFINE CALLBACKS

def transcribe(audio_path):
    """
    Transcribe audio file to text using Whisper model.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        str: Transcribed text.
    """

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
    """
    Interact with the LLM using the provided query and conversation history.
    Args:
        query (str): User's query.
        history (list): Conversation history as a list of messages.
    Returns:
        str: LLM's response.
    """

    # Prepare messages in OpenAI-style format
    messages = [
        {"role": "system", "content": \
        """tu es un assistant francophone destiné aux enfants de 8 ans, qui s'appelle Sam.
        Réponds en une ou deux phrases courtes adaptées pour la synthèse vocale."""},
        *history,
    ]

    logging.info(f"user queried: {query}")

    answer = hf.chat_completion(messages=messages, max_tokens=512).choices[0].message.content

    logging.info(f"bot answered: {answer}")

    return answer

def synthesize(text, voice="ff_siwis"):
    """
    Synthesize text to speech using Kokoro TTS pipeline.
    Args:
        text (str): Text to synthesize.
        voice (str): Voice model to use for synthesis.
    Returns:
        tuple: Sampling rate and audio data as a numpy array.
    """

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

from vad_js import js, js_reset

import gradio as gr

from dataclasses import dataclass, field

@dataclass
class AppState:
    conversation: list = field(default_factory=list)

with gr.Blocks(js=js) as demo:
    
    state = gr.State(value=AppState())
    
    gr.Image("images/sam.png", height=300)

    input_audio = gr.Audio(
        sources=["microphone"],
        label="Speak",
        type="filepath",
        waveform_options=gr.WaveformOptions(waveform_color="#DB7FBF")
    )
    chatbot = gr.Chatbot(
        label="Conversation",
        type="messages",
        visible=False
    )
    output_audio = gr.Audio(
        label="TTS Response",
        autoplay=True,
        visible=True, 
        elem_id="streaming_out"
    )
    
    def run_step(state: AppState, audio_path,):
        """
        Process a single step in the conversation.
        Args:
            state (AppState): Current application state.
            audio_path (str): Path to the recorded audio file.
        Yields:
            AppState: Updated application state.
            list: Conversation history.
            tuple: Audio tuple for TTS response.
        """

        if not input_audio:
            return AppState()

        user_text = transcribe(audio_path)  # now using faster-whisper
        state.conversation.append({"role": "user", "content": user_text})

        yield state, state.conversation, None

        # LLM and TTS logic unchanged:
        bot_text = chat_with_llm(user_text, state.conversation)
        state.conversation.append({"role": "assistant", "content": bot_text})
        audio_tuple = synthesize(bot_text)

        yield state, state.conversation, audio_tuple

    stream = input_audio.start_recording(
        lambda audio, state: (audio, state),
        [input_audio, state],
        [input_audio, state],
    )
    respond = input_audio.stop_recording(
        run_step,
        [state, input_audio],
        [state, chatbot, output_audio]
    )
    restart = respond.then(
        lambda state: None, [state], [input_audio]).then(
            lambda state: state, state, state, js=js_reset
        )

    cancel = gr.Button("Restart Conversation", variant="stop")
    cancel.click(
        lambda: (AppState(), gr.Audio(recording=False)),
        None,
        [state, input_audio],
        cancels=[respond, restart],
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to serve Gradio app on")
    args = parser.parse_args()

    # demo.launch(
    #         server_name=args.ip,
    #         server_port=7860,
    #         share=True)

    demo.launch(
        server_name=args.ip,
        server_port=7860,
        share=False,
        ssl_certfile='cert.pem',
        ssl_keyfile='key.pem',
        ssl_verify=False)

# openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes