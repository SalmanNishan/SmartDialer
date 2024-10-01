import os
import json
from .transcription_clips import TranscriptionClips
from config import entity_extraction_dir

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