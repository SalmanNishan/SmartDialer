import os
import sys
import numpy as np
import base64
import grpc
import json
import logging
import wave
import torch
import gc
import scipy.signal as sps
from thefuzz import process, fuzz
from concurrent import futures
from whisper import load_model
from torch.utils.data import DataLoader, Dataset

import helloworld_pb2
import helloworld_pb2_grpc
from flashtext_files.flashtext_matching import predict_label
from config import *

resampler_ratio = int(transcription_model_sample_rate / input_stream_sample_rate)
accumulated_frame = []

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
        text, self.empty_counter, self.frame_asr_buffer, self.frame_asr_prev_char = transcribe_stream(frame_asr, asr_model, self.empty_counter, self.frame_asr_buffer, self.frame_asr_prev_char)
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

frame_asr, asr_model = create_frame_asr()
on_going_calls = {}

class CallStatusLogger:
    def __init__(self, bridge_id, insurance, call_progress='None', text=[]):
        self.transcription_manager = TranscriptionClips()
        self.bridge_id = bridge_id
        self.frame_buffer = b''
        self.call_progress = call_progress
        self.insurance = insurance
        self.prev_extracted_keyword = None
        self.text = text
        self.bag_of_words = self._load_bag_of_words()
        self.fuzzy_corpus = self._load_fuzzy_corpus()
        self.dial_plan = self._load_dial_plan()
        self.packet_count = 0
        self.packet_count_1 = 0

    def _load_bag_of_words(self):
        with open(os.path.join(entity_extraction_dir, "insurance_bag_of_words_grpc_old_copy.json"), 'r') as f:
            bag_of_words = f.read()
            bag_of_words = json.loads(bag_of_words)
            f.close()
        return bag_of_words

    def _load_fuzzy_corpus(self):
        with open(os.path.join(entity_extraction_dir, "common_dial_plan_flashtext_grpc_old_copy.json"), 'r') as f:
            common_text_matching = f.read()
            common_text_matching = json.loads(common_text_matching)
            f.close()
        return common_text_matching

    def _load_dial_plan(self):
        with open(os.path.join(entity_extraction_dir, "dial_plans_grpc_old_copy.json"), 'r') as f:
            dial_plans = f.read()
            dial_plans = json.loads(dial_plans)
            f.close()
        return dial_plans

class Greeter(helloworld_pb2_grpc.GreeterServicer):
    def SayHello(self, request, context):
        insurance_name = request.insurance_name
        audio_data_base64 = request.audio_data_base64
        bridge_id = request.bridge_id

        if bridge_id not in on_going_calls:
            call_status_logger = CallStatusLogger(bridge_id, insurance_name, 'None', [])
            on_going_calls[bridge_id] = call_status_logger
        else:
            call_status_logger = on_going_calls[bridge_id]

        audio_binary_stream = base64.b64decode(audio_data_base64)
        call_status_logger.transcription_manager.add_clip(audio_binary_stream)
        call_status_logger.packet_count += 1
        print(f"1- Packet Count For'{insurance_name}': {call_status_logger.packet_count}")
        print("")

        if len(accumulated_frame) < frame_limit:
            accumulated_frame.extend(call_status_logger.transcription_manager.test_frame)
        else:
            accumulated_frame_np = np.array(accumulated_frame)
            wf = wave.open('packet_order_test.wav', 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)  # setting sample width to 2 bytes
            wf.setframerate(8000)  # setting sample rate to 8000 Hz
            wf.writeframes(accumulated_frame_np.tobytes())  # converting to binary string and writing it to file
            wf.close()

        call_status_logger.text.append(call_status_logger.transcription_manager.transcribe(frame_asr, asr_model))
        insurance_transcript = [call_status_logger.text[i] for i in range(len(call_status_logger.text)) if (i + 1) % 7 == 0]
        # print("insurance_transcript", insurance_transcript)
        # print("*********************************************************************************************")
        localText = ''.join(insurance_transcript)
        print("localText", localText)

        extracted_keyword, action, action_callback, call_progress = self.extract_entities(call_status_logger)
        if extracted_keyword == "None":
            call_status_logger.prev_extracted_keyword = "None"
        else:
            if call_status_logger.prev_extracted_keyword != extracted_keyword:
                call_status_logger.prev_extracted_keyword = extracted_keyword
            else:
                extracted_keyword = "None"
                action = "None"
                action_callback = "None"

        call_status_logger.packet_count_1 += 1  # Increment packet counter
        print(f"2- Packet Count For '{insurance_name}': {call_status_logger.packet_count_1}")
        print("******************************************************************************")

        return helloworld_pb2.HelloReply(
            extracted_keyword='Keyword : %s' % extracted_keyword,
            action='Action : %s' % action,
            action_callback='Callback : %s' % action_callback,
            call_progress='Call Progress : %s' % call_progress,
            transcript='Transcript %s' % localText
        )

    def extract_entities(self, call_status_logger):
        text = ''.join(call_status_logger.text[-8:])
        hold_text = text
        num_of_tokens = text.split()
        text.replace(" ", "")

        insurance_name = call_status_logger.insurance
        print("")
        print("Insurance_Name", insurance_name)

        common_text_matching = call_status_logger.fuzzy_corpus
        bag_of_words = call_status_logger.bag_of_words
        dial_plans = call_status_logger.dial_plan

        keywords_given_string = dict((value.replace(" ", ""), key) for key, value_ls in common_text_matching.items() for value in value_ls)
        contiguous_sentences_plan = list(value.replace(" ", "") for key, value_ls in common_text_matching.items() for value in value_ls)
        desired_plan_plan = dial_plans[insurance_name]

        matched_string, score = process.extractOne(text, contiguous_sentences_plan, scorer=fuzz.partial_ratio)

        if call_status_logger.call_progress == 'Active':
            return 'Active', 'None', 'None', call_status_logger.call_progress

        elif call_status_logger.call_progress == 'Hold':
            active_flag, _, _ = predict_label(hold_text)
            if active_flag:
                call_status_logger.call_progress = 'Active'
                return 'Active', 'None', 'None', call_status_logger.call_progress
            else:
                return 'None', 'None', 'None', call_status_logger.call_progress

        elif score > 75 and len(num_of_tokens) >= 5:
            extracted_keyword = keywords_given_string[matched_string]
            print("Extracted_Keyword : ", extracted_keyword)
            if extracted_keyword == 'Hold':
                call_status_logger.call_progress = 'Hold'
                return extracted_keyword, 'None', 'None', call_status_logger.call_progress
            elif call_status_logger.prev_extracted_keyword == 'Hold':
                call_status_logger.call_progress = 'Hold'
                return extracted_keyword, 'None', 'None', call_status_logger.call_progress
            else:
                call_status_logger.call_progress = 'Inprogress'

            if extracted_keyword not in bag_of_words[insurance_name]:
                return 'None', 'None', 'None', call_status_logger.call_progress

            action = desired_plan_plan[extracted_keyword]
            print("Action", action)

            if action["excel"]["key"] == "":
                print("Action_1" ,extracted_keyword, action["sending_data"], action["callback"], call_status_logger.call_progress)
                return extracted_keyword, action["sending_data"], action["callback"], call_status_logger.call_progress
            else:
                print("Action_2", extracted_keyword, 'DB', action["callback"], call_status_logger.call_progress)
                return extracted_keyword, 'DB', action["callback"], call_status_logger.call_progress
        else:
            return 'None', 'None', 'None', call_status_logger.call_progress

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('0.0.0.0:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()