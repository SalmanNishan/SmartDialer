import gc
import scipy.signal as sps
from config import *
from .load_transcription import np, torch, load_transcription_model, FrameASR

resampler_ratio = int(transcription_model_sample_rate / input_stream_sample_rate)
class TranscriptionClips:
    def __init__(self):
        self.empty_counter = 0
        self.frame_len = frame_length
        self.n_frame_len = int(frame_length * transcription_model_sample_rate)
        self.n_frame_overlap = int(frame_asr_overlap * transcription_model_sample_rate)
        self.frame_asr_buffer = np.zeros(shape=3 * self.n_frame_overlap + self.n_frame_len, dtype=np.float32)
        self.frame_asr_prev_char = ''
        self.processed_packet = -2

    def add_clip(self, audio_data):
        ''' Convert binary data to numpy array and resample to match Whisper Sample Rate '''
        data_arr = np.frombuffer(audio_data, dtype=np.int16)
        if resampler_ratio != 1:
            frame = sps.resample(data_arr, len(data_arr) * resampler_ratio).astype(np.int16)
        else:
            frame = data_arr

        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.int16)
        elif len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
        elif len(frame) > self.n_frame_len:
            frame = frame[:self.n_frame_len]
        assert len(frame) == self.n_frame_len
        self.frame_asr_buffer[:-self.n_frame_len] = self.frame_asr_buffer[self.n_frame_len:]
        self.frame_asr_buffer[-self.n_frame_len:] = frame
        self.test_frame = data_arr

    def transcribe(self, frame_asr, asr_model):
        text, self.empty_counter, self.frame_asr_buffer, self.frame_asr_prev_char = transcribe_stream(frame_asr, asr_model, \
                                                                                                    self.empty_counter, self.frame_asr_buffer, self.frame_asr_prev_char)
        return text

def transcribe_stream(frame_asr, asr_model, empty_counter, buffer, prev_char):
    text, buffer, prev_char = frame_asr.transcribe(asr_model, buffer, prev_char)
    if len(text):
        empty_counter = frame_asr.offset
        return text, empty_counter, buffer, prev_char
    elif empty_counter > 0:
        empty_counter -= 1
        if empty_counter == 0:
            return ' ', empty_counter, buffer, prev_char
    return text, empty_counter, buffer, prev_char

def reset_asr(asr):
    asr.reset()

def create_frame_asr():
    asr_model = load_transcription_model()
    asr = FrameASR(sample_rate=transcription_model_sample_rate, frame_len=frame_length, frame_overlap=frame_asr_overlap, offset=frame_asr_offset)
    torch.cuda.empty_cache()
    gc.collect()
    return asr, asr_model
