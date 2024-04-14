from faster_whisper import WhisperModel
import os
import pyaudio
import wave

last_result: str = ''

def load_model(model_size: str) -> None:
    global MODEL
    MODEL = WhisperModel(model_size, device='cuda', compute_type='float16')

def remove_repeat_result(result: str) -> str:
    global last_result
    if last_result == result:
        result = ''
    last_result = result
    return result

def model_generate(file_path: str, task: str, detect_language: str) -> str:
    global MODEL
    segments, info = MODEL.transcribe(file_path, beam_size=5, task=task, language=detect_language)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription

def record_chunk(p: pyaudio, stream, file_path: str, fs: int, chunk: int, seconds: int = 1):
    frames = []
    for _ in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

def start_recording(chunk: int, fs: int, channels: int, seconds: int, task: str, detect_language: str):
    global keep_recording
    keep_recording = True

    p: pyaudio = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=fs, input=True, frames_per_buffer=chunk)

    accumulated_transcription = ''
    try:
        while keep_recording:
            chunk_file = 'temp_chunk.wav'
            record_chunk(p, stream, chunk_file, fs, chunk, seconds)
            transcription = model_generate(chunk_file, task, detect_language)
            result = remove_repeat_result(transcription)

            accumulated_transcription += result + '\n'
            os.remove(chunk_file)
            yield accumulated_transcription

    except KeyboardInterrupt:
        print('stopping...')
        os.remove(chunk_file)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def stop_recording():
    global keep_recording
    keep_recording = False