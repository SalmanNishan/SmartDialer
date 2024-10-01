from flashtext import KeywordProcessor
from config import flashtext_list_ivr, flashtext_list_operator


with open(flashtext_list_operator, 'r') as handle:
    keyword_text = handle.read()
    
with open(flashtext_list_ivr, 'r') as handle:
    ivr_list = handle.read().split('\n')

operator_list = keyword_text.split('\n')
keyword_processor = KeywordProcessor()
keyword_processor.add_keywords_from_list(operator_list)
keyword_processor.add_keywords_from_dict({'ivr': ivr_list})

def predict_label(text):
    '''
    Predicts whether an input text is spoken by an IVR system or if it is spoken by a human agent
    Uses keyword matching for prediction

    text: (str) The text to predict the label for

    Returns:
    True if text is detected to be from human operator
    False otherwise
    '''
    word_found = False
    keywords_found = keyword_processor.extract_keywords(text)
    if len(keywords_found) > 0:
        if 'ivr' in keywords_found:
            word_found = False
        else:
            word_found = True
    
    return word_found, text, keywords_found
