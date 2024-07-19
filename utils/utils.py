import json
import os

from deep_translator import GoogleTranslator
from langdetect import detect


def load_metadata(metadata_dir):
    metadata_dict = {}
    for subdir in ['id', 'en']:
        full_dir_path = os.path.join(metadata_dir, subdir)
        for filename in os.listdir(full_dir_path):
            if filename.endswith('.json'):
                filepath = os.path.join(full_dir_path, filename)
                with open(filepath, 'r') as f:
                    metadata_list = json.load(f)
                    for item in metadata_list:
                        metadata_dict[item['file_id']] = item
    return metadata_dict

def ensure_metadata_fields(metadata, required_fields):
    return {field: metadata.get(field, "") for field in required_fields}

translator_en = GoogleTranslator(source='auto', target='en')
translator_id = GoogleTranslator(source='auto', target='id')

def is_language_indonesian(query):
    # This function assumes only laguage is either
    # Indonesian or English
    detected_lang = detect(query)
    return detected_lang == 'id'

def translate(lang, query):
    if lang == "id":
        return translator_id.translate(query)
    return translator_en.translate(query)

if __name__ == "__main__":
  scanned_pdf = 'data_pool/pbi/pbi-2_15_pbi_2000-20032004-bank_indonesia_regulation_no2_15_pbi_2000_dated_june_12_2000___bank_indonesia_regulation_no2_16_pbi_2000___concerning_amendment_to_decree_of_the_board_of_managing_directors_of_bank_indonesia_no31_150_kep_dir_dated_uxl.pdf'
  true_pdf = 'data_pool/pbi/pbi-6_3_pbi_2004-16022004-bank_indonesia_regulation_nr_6_3_pbi_2004_issuance_sale_and_purchase_and_administration_of_government_securities_nk8.pdf'


#   is_scanned = is_scanned_pdf(true_pdf)
#   print("result", is_scanned)