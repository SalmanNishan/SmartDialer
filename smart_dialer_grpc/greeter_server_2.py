import numpy as np
import base64
import grpc
import logging
import wave
from thefuzz import process, fuzz
from concurrent import futures

import helloworld_pb2
import helloworld_pb2_grpc
from flashtext_files.flashtext_matching import predict_label
from transcription_model.transcription_clips import create_frame_asr
from transcription_model.call_status import CallStatusLogger
from config import frame_limit

accumulated_frame = []
frame_asr, asr_model = create_frame_asr()
on_going_calls = {}

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
    server.add_insecure_port('0.0.0.0:50052')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()