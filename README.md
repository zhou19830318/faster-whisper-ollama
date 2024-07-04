# faster-whisper-ollama
A real-time voice chatbot based on foster whisper and ollama

```markdown
# Local Audio Recording and Transcription

This script provides a local audio recording and transcription service without relying on cloud services, making it suitable for offline use.

## Dependencies

Before running the script, you need to install the following dependencies:

```bash
pip3 install pyaudio webrtcvad faster-whisper
```

## Running the Script

To run the script, use the following command:

```bash
python3 faster_whisper_demo.py
```

## Script Overview

This script records audio from the microphone, transcribes it using a local model, and provides real-time transcription services. Below is an overview of the main components:

### 1. `Queues` Class

- Manages two queues: `audio` for storing audio data and `text` for storing transcribed text.

### 2. `Transcriber` Class

- Utilizes the FasterWhisper model for audio transcription.
- Initializes the model with specified parameters such as model size, device, and compute type.
- Processes audio data and yields transcribed text segments.

### 3. `AudioRecorder` Class

- Records audio using the PyAudio library and processes it with WebRTC VAD (Voice Activity Detection).
- Initializes the audio stream and VAD, and manages the recording and buffering of audio frames.
- Detects speech activity and manages the transition between recording and silence.

### 4. `Chat` Class

- Handles communication with the Ollama chatbot.
- Sends transcribed text to the chatbot and receives responses.

### 5. `main` Function

- Initializes and starts the audio recorder, transcriber, and chat components.
- Manages the lifecycle of these components and handles exceptions.

## Usage

1. **Audio Recording**:
   - The script continuously records audio from the microphone.
   - When speech is detected, it starts recording, and when silence is detected, it stops and processes the audio.

2. **Audio Transcription**:
   - The recorded audio is sent to the `Transcriber` class, which uses the FasterWhisper model to transcribe the audio into text.

3. **Chat Interaction**:
   - The transcribed text is sent to the `Chat` class, which communicates with the Ollama chatbot and retrieves responses.

## Logging

- The script uses the `logging` module to log various stages of the recording, transcription, and chat processes.
- Log messages provide information on the status and any errors encountered during the execution.

## Example Output

- Transcribed text is logged in real-time, and responses from the chatbot are printed to the console.

## Error Handling

- The script includes error handling to manage exceptions during initialization, audio recording, transcription, and chat communication.

## Environment Variable

- Sets the environment variable `KMP_DUPLICATE_LIB_OK` to "TRUE" to resolve potential issues with library conflicts.

## Notes

- Ensure that your microphone is properly configured and accessible by the script.
- The script is designed to work with specific models and configurations. Adjust the parameters as needed for your environment.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
