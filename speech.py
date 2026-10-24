import io
from faster_whisper import WhisperModel
from config import WHISPER_MODEL

model = None


def transcribe(audio_bytes, language=None):
    global model
    if model is None:
        # loaded on the first request, the first time it also downloads the model
        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    segments, info = model.transcribe(io.BytesIO(audio_bytes), language=language,
                                      beam_size=5, vad_filter=True)
    text = " ".join(s.text.strip() for s in segments).strip()
    return text, info.language
