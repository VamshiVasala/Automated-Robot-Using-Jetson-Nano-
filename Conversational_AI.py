import grpc
import pyaudio
import numpy as np
import itertools
import threading
from riva.client.proto import riva_asr_pb2, riva_asr_pb2_grpc
from riva.client.proto import riva_tts_pb2, riva_tts_pb2_grpc
import riva.client.proto.riva_audio_pb2 as audio_pb2
from openai import OpenAI
import re

# Configuration
SERVER = "grpc.nvcf.nvidia.com:443"
USE_SSL = True
AUTH_TOKEN = "Bearer api-key"
FUNCTION_ID_ASR = "func-id-asr-key"
FUNCTION_ID_TTS = "func-id-tts-key"

LANGUAGE_CODE = "en-US"
SAMPLE_RATE = 16000
CHUNK_SIZE = 4096
WAKE_WORD = "jack"
VOICE_NAME = "English-US.Male-1"



# Setup gRPC connection
creds = grpc.ssl_channel_credentials() if USE_SSL else None
channel = grpc.secure_channel(SERVER, creds) if USE_SSL else grpc.insecure_channel(SERVER)
asr_client = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(channel)
tts_client = riva_tts_pb2_grpc.RivaSpeechSynthesisStub(channel)

# OpenAI (Gemma) Client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="api-key"
)
# Interrupt flag
interrupt_flag = threading.Event()

def audio_stream():
    """Stream microphone audio to Riva ASR."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)
    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            yield riva_asr_pb2.StreamingRecognizeRequest(audio_content=data)
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

def detect_wake_word():
    """Listens for the wake word to interrupt or start conversation."""
    metadata = (('function-id', FUNCTION_ID_ASR), ('authorization', AUTH_TOKEN))
    config = riva_asr_pb2.RecognitionConfig(
        encoding=audio_pb2.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=SAMPLE_RATE,
        language_code=LANGUAGE_CODE,
        max_alternatives=1,
        enable_automatic_punctuation=False
    )
    streaming_config = riva_asr_pb2.StreamingRecognitionConfig(config=config)
    request_iterator = itertools.chain(
        iter([riva_asr_pb2.StreamingRecognizeRequest(streaming_config=streaming_config)]),
        audio_stream()
    )

    print(f"Listening for wake word: '{WAKE_WORD}'...")

    for response in asr_client.StreamingRecognize(request_iterator, metadata=metadata):
        if response.results:
            transcript = response.results[0].alternatives[0].transcript.lower()
            print(f"Detected: {transcript}")
            if WAKE_WORD in transcript:
                print("Wake word detected! Interrupting response...")
                interrupt_flag.set()  # Set flag to stop speech immediately
                return True
    return False


def transcribe_speech():
    """Converts speech to text using Riva ASR."""
    metadata = (('function-id', FUNCTION_ID_ASR), ('authorization', AUTH_TOKEN))
    config = riva_asr_pb2.RecognitionConfig(
        encoding=audio_pb2.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=SAMPLE_RATE,
        language_code=LANGUAGE_CODE,
        max_alternatives=1,
        enable_automatic_punctuation=True
    )
    streaming_config = riva_asr_pb2.StreamingRecognitionConfig(config=config)
    request_iterator = itertools.chain(
        iter([riva_asr_pb2.StreamingRecognizeRequest(streaming_config=streaming_config)]),
        audio_stream()
    )

    for response in asr_client.StreamingRecognize(request_iterator, metadata=metadata):
        if response.results:
            transcript = response.results[0].alternatives[0].transcript
            return transcript  
    return ""

# Store conversation history
conversation_history = []

def chat_with_gemma(user_input):
    """Send user input to Gemma (OpenAI API) and maintain conversation history."""
    global conversation_history
    
    # Append user input to history
    conversation_history.append({"role": "user", "content": user_input})
    
    # Keep only the last 5 interactions to maintain relevance
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]
    
    try:
        completion = client.chat.completions.create(
            model="google/gemma-2-2b-it",
            messages=conversation_history,  # Send history for context
            temperature=0.3,
            top_p=0.8,
            max_tokens=100,
            stop=["\n"],
            stream=False
        )

        response_text = completion.choices[0].message.content.strip()

        if not response_text:
            return "Sorry, I couldn't generate a response."

        # Add AI response to history
        conversation_history.append({"role": "assistant", "content": response_text})

        print(f"Assistant: {response_text}")
        return response_text

    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, I couldn't process that."



def monitor_interrupt():
    """ Listens for the wake word during speech playback and interrupts when detected. """
    global interrupt_flag
    while True:
        if detect_wake_word():
            interrupt_flag.set()  # Set the flag to stop speech playback
            print("🛑 Wake word detected! Interrupting response...")
            return  # Exit thread immediately, allowing the system to take new input

def generate_speech(text):
    """Convert text to speech and play, but allow interruption."""
    if not text or text.isspace():
        print("Error: TTS received empty input, skipping synthesis.")
        return None

    global interrupt_flag
    interrupt_flag.clear()
    
    # Remove * and # symbols from the response
    clean_text = re.sub(r'[*#]', '', text)

    metadata = (('function-id', FUNCTION_ID_TTS), ('authorization', AUTH_TOKEN))

    request = riva_tts_pb2.SynthesizeSpeechRequest(
        text=clean_text,  # Use cleaned text
        language_code=LANGUAGE_CODE,
        encoding=audio_pb2.AudioEncoding.LINEAR_PCM,  
        sample_rate_hz=22050,
        voice_name=VOICE_NAME
    )

    response = tts_client.Synthesize(request, metadata=metadata)
    
    if not response.audio:
        print("🔴 No audio data received from Riva TTS!")
        return

    print(f"🟢 Received TTS response with {len(response.audio)} bytes of audio data.")

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=22050, output=True)

    audio_data = np.frombuffer(response.audio, dtype=np.int16)
    chunk_size = 1024

    # Start wake word detection in a thread
    wake_word_thread = threading.Thread(target=monitor_interrupt)
    wake_word_thread.daemon = True
    wake_word_thread.start()

    for i in range(0, len(audio_data), chunk_size):
        if interrupt_flag.is_set():  # If wake word is detected, stop speaking and listen for query
            print("🛑 Speech interrupted! Listening for new input immediately...")
            stream.stop_stream()
            stream.close()
            p.terminate()

            # 🔹 Immediately start listening for the new query
            transcript = transcribe_speech()
            if transcript.strip():
                response = chat_with_gemma(transcript)
                generate_speech(response)  # Respond without requiring another wake word
            return  # Exit function

        stream.write(audio_data[i:i+chunk_size].tobytes())

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("✅ TTS playback finished.")

def transcribe_live():
    """Main loop for listening, responding, and handling interruptions."""
    while True:
        print(f"🎤 Waiting for wake word: '{WAKE_WORD}'...")
        if detect_wake_word():
            print("🟢 Wake word detected! Start speaking your query...")
            transcript = transcribe_speech()

            if not transcript.strip():
                print("No valid input detected, continuing...")
                continue

            # Get AI response
            assistant_response = chat_with_gemma(transcript)
            if not assistant_response:
                print("AI returned an empty response, skipping TTS...")
                continue

            # Convert AI response to speech (can be interrupted)
            generate_speech(assistant_response)


# Run live transcription and interaction
if __name__ == "__main__":
    transcribe_live()
