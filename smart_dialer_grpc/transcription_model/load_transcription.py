import torch
import numpy as np
from whisper import load_model
from torch.utils.data import DataLoader, Dataset
from config import frame_length, frame_asr_offset

def load_transcription_model():
    try:
        transcription_model = load_model("small.en")
    except FileNotFoundError:
        raise FileNotFoundError(f"Model not found at the specified path")

    transcription_model.eval()
    transcription_model = transcription_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    return transcription_model

# Simple data layer to pass audio signal
class AudioDataLayer(Dataset):
    def __init__(self, signal):
        self.signal = signal.astype(np.float32) / 32768.0  # Normalize the signal
        self.signal_shape = self.signal.shape

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.as_tensor(self.signal, dtype=torch.float32), torch.as_tensor(self.signal_shape, dtype=torch.int64)

# Inference method for audio signal (single instance)
def infer_signal(model, signal):
    data_layer = AudioDataLayer(signal)
    data_loader = DataLoader(data_layer, batch_size=1)
    audio_signal, audio_signal_len = next(iter(data_loader))
    audio_signal, audio_signal_len = audio_signal.to(model.device), audio_signal_len.to(model.device)
    result = model.transcribe(audio_signal.cpu().numpy()[0])
    return result['text']

# Class for streaming frame-based ASR
class FrameASR:
    def __init__(self, sample_rate, frame_len=frame_length, frame_overlap=1, offset=frame_asr_offset):
        self.sample_rate = sample_rate
        self.frame_len = frame_len
        self.frame_overlap = frame_overlap
        self.offset = offset
        self.n_frame_len = int(frame_len * sample_rate)
        self.n_frame_overlap = int(frame_overlap * sample_rate)
        self.buffer = np.zeros(shape=3 * self.n_frame_overlap + self.n_frame_len, dtype=np.float32)
        self.reset()

    def transcribe(self, model, buffer, prev_char, merge=True):
        unmerged = infer_signal(model, buffer)
        if not merge:
            return unmerged, unmerged, prev_char
        text, prev_char = self.greedy_merge(unmerged, prev_char)
        return text, buffer, prev_char

    def reset(self):
        self.buffer = np.zeros(shape=self.buffer.shape, dtype=np.float32)
        self.prev_char = ''

    @staticmethod
    def greedy_merge(s, prev_char):
        s_merged = ''
        for char in s:
            if char != prev_char:
                prev_char = char
                s_merged += char
        return s_merged, prev_char