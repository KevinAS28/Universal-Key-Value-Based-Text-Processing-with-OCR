import json

from ocrfw.postprocessing import *
from ocrfw.ocr import *
from ocrfw.misc import *


with open('ktp_config.json', 'r') as kc_f:
    KTP_CONFIG = json.loads(kc_f.read())

def parse_ktp(
    img=None,
    img_full_path=None,
    ktp_str=None,
    output_orders=['success', 'result', 'all_error_messages', 'warning', 'all_error_codes', 'model_result'],
    result_orders=['PROVINSI', 'KABUPATEN/KOTA', 'NIK', 'Nama', 'Tempat', 'Tanggal Lahir', 'Jenis kelamin', 'GolDarah', 'Alamat', 'RT/RW', 'Kel/Desa', 'Kecamatan', 'Agama', 'Status Perkawinan', 'Pekerjaan', 'Kewarganegaraan', 'Berlaku Hingga'],
    config_source=1,
    use_final_evaluator=True
    ):

    if config_source==1:
        configs = get_ktp_line_configs()
    elif config_source==2:
        configs = config_source
    elif config_source==0:
        # default config
        configs = KTP_CONFIG

    to_extracts_ktp = [

        dict(extractor=get_str_keys_values, extractor_args=dict(key_str_list_list=[['PROVINSI']], max_distances=configs['provinsi']['max_distances'], min_accuracies=configs['provinsi']['min_accuracies'], separators=[':'], re_chars_filter=[['[a-z|A-Z|0-9]']],alias_names=['PROVINSI']), evaluators=[provinsi_evaluator], multi_line_value=False, alias_names=['PROVINSI'], tolerant_not_exists=configs['provinsi']['tolerant_not_exists']), 

        dict(extractor=get_all, extractor_args=dict(alias_names=['KABUPATEN/KOTA']), evaluators=[kab_kota_evaluator], multi_line_value=False, alias_names=['KABUPATEN/KOTA'], tolerant_not_exists=configs['kabupatenkota']['tolerant_not_exists']), 

        dict(extractor=get_str_keys_values, extractor_args=dict(key_str_list_list=[['NIK']], max_distances=configs['nik']['max_distances'], min_accuracies=configs['nik']['min_accuracies'], separators=[':'], re_chars_filter=[['[a-z|A-Z|0-9]', '[\\:]']], alias_names=['NIK']), evaluators=[nik_evaluator], multi_line_value=False, alias_names=['NIK'], tolerant_not_exists=[False]),

        dict(extractor=get_str_keys_values, extractor_args=dict(key_str_list_list=[['Nama']], max_distances=configs['nama']['max_distances'], min_accuracies=configs['nama']['min_accuracies'], separators=[':'], re_chars_filter=[['[a-z|A-Z|0-9]']], alias_names=['Nama']), evaluators=[letters_evaluator], multi_line_value=True, alias_names=['Nama'], tolerant_not_exists=configs['nama']['tolerant_not_exists']), 

        dict(extractor=get_str_keys_values, extractor_args=dict(key_str_list_list=['Tempat Tgl Lahir'.split(' ')], max_distances=configs['tempattgllahir']['max_distances'], min_accuracies=configs['tempattgllahir']['min_accuracies'], separators=[':'], re_chars_filter=[['[a-z|A-Z|\/|0-9]']*3], alias_names=['Tempat/Tgl Lahir'], wo_space=True), evaluators=[ttl_evaluator], multi_line_value=False, alias_names=['Tempat/Tgl Lahir'], tolerant_not_exists=configs['tempattgllahir']['tolerant_not_exists']), 

        dict(extractor=get_str_keys_values, extractor_args=dict(key_str_list_list=['jenis kelamin'.split(' '), 'gol darah'.split(' ')], max_distances=configs['jeniskelamin|goldarah']['max_distances'], min_accuracies=configs['jeniskelamin|goldarah']['min_accuracies'], separators=[':', '.'], re_chars_filter=[['[a-z|A-Z|\-|0-9]']*3, ['[a-z|A-Z|\-|0-9]']*3], alias_names=['Jenis kelamin', 'GolDarah'], line_preprocessing=lambda x: x.lower(), wo_space=True), evaluators=[jk_evaluator, darah_evaluator], multi_line_value=False, alias_names=['Jenis kelamin', 'GolDarah'], tolerant_not_exists=configs['jeniskelamin|goldarah']['tolerant_not_exists']), 

        dict(extractor=get_str_keys_values, extractor_args=dict(key_str_list_list=[['Alamat']], max_distances=configs['alamat']['max_distances'], min_accuracies=configs['alamat']['min_accuracies'], separators=[':'], re_chars_filter=[['[a-z|A-Z|0-9]']], alias_names=['Alamat']), evaluators=[lambda x: [True, nodigits_typo(letters_evaluator(x)[1])]], multi_line_value=True, alias_names=['Alamat'], tolerant_not_exists=configs['alamat']['tolerant_not_exists']), 
    
        dict(extractor=get_str_keys_values, extractor_args=dict(key_str_list_list=[['RTRW']], max_distances=configs['rtrw']['max_distances'], min_accuracies=configs['rtrw']['min_accuracies'], separators=[':'], re_chars_filter=[['[a-z|A-Z|0-9|\=|\/]'], ['[a-z|A-Z|0-9|\=|\/]']], alias_names=['RT/RW'], line_preprocessing=lambda x: x.upper(), wo_space=True), evaluators=[rtrw_evaluator], multi_line_value=True, alias_names=['RT/RW'], tolerant_not_exists=configs['rtrw']['tolerant_not_exists']), 

        dict(extractor=get_str_keys_values, extractor_args=dict(key_str_list_list=['KelDesa'.split(' ')], max_distances=configs['keldesa']['max_distances'], min_accuracies=configs['keldesa']['min_accuracies'], separators=[':'], re_chars_filter=[['[a-z|A-Z|0-9]']*3], alias_names=['Kel/Desa']), evaluators=[lambda x: letters_evaluator(x.upper())], multi_line_value=False, alias_names=['Kel/Desa'], tolerant_not_exists=configs['keldesa']['tolerant_not_exists']),    

        dict(extractor=get_str_keys_values, extractor_args=dict(key_str_list_list=[['Kecamatan']], max_distances=configs['kecamatan']['max_distances'], min_accuracies=configs['kecamatan']['min_accuracies'], separators=[':'], re_chars_filter=[['[a-z|A-Z|0-9]']], alias_names=['Kecamatan']), evaluators=[letters_evaluator], multi_line_value=True, alias_names=['Kecamatan'], tolerant_not_exists=configs['kecamatan']['tolerant_not_exists']), 

        dict(extractor=get_str_keys_values, extractor_args=dict(key_str_list_list=[['Agama']], max_distances=configs['agama']['max_distances'], min_accuracies=configs['agama']['min_accuracies'], separators=[':'], re_chars_filter=[['[a-z|A-Z|0-9]']], alias_names=['Agama']), evaluators=[agama_evaluator], multi_line_value=True, alias_names=['Agama'], tolerant_not_exists=configs['agama']['tolerant_not_exists']), 

        dict(extractor=get_str_keys_values, extractor_args=dict(key_str_list_list=['Status Perkawinan'.split(' ')], max_distances=configs['statusperkawinan']['max_distances'], min_accuracies=configs['statusperkawinan']['min_accuracies'], separators=[':'], re_chars_filter=[['[a-z|A-Z|\/|0-9]']*2], alias_names=['Status Perkawinan']), evaluators=[kawin_evaluator], multi_line_value=False, alias_names=['Status Perkawinan'], tolerant_not_exists=configs['statusperkawinan']['tolerant_not_exists']),
                 
        dict(extractor=get_str_keys_values, extractor_args=dict(key_str_list_list=[['Pekerjaan']], max_distances=configs['pekerjaan']['max_distances'], min_accuracies=configs['pekerjaan']['min_accuracies'], separators=[':'], re_chars_filter=[['[a-z|A-Z|0-9]']], alias_names=['Pekerjaan']), evaluators=[pekerjaan_evaluator], multi_line_value=True, alias_names=['Pekerjaan'], tolerant_not_exists=configs['pekerjaan']['tolerant_not_exists']),         

        dict(extractor=get_str_keys_values, extractor_args=dict(key_str_list_list=[['Kewarganegaraan']], max_distances=configs['kewarganegaraan']['max_distances'], min_accuracies=configs['kewarganegaraan']['min_accuracies'], separators=[':'], re_chars_filter=[['[a-z|A-Z|0-9]']], alias_names=['Kewarganegaraan']), evaluators=[kn_evaluator], multi_line_value=True, alias_names=['Kewarganegaraan'], tolerant_not_exists=configs['kewarganegaraan']['tolerant_not_exists']),         

        dict(extractor=get_str_keys_values, extractor_args=dict(key_str_list_list=['Berlaku Hingga'.split(' ')], max_distances=configs['berlakuhingga']['max_distances'], min_accuracies=configs['berlakuhingga']['min_accuracies'], separators=[':'], re_chars_filter=[['[a-z|A-Z|\/|0-9]']*2], alias_names=['Berlaku Hingga'], wo_space=True), evaluators=[berlaku_evaluator], multi_line_value=False, alias_names=['Berlaku Hingga'], tolerant_not_exists=configs['berlakuhingga']['tolerant_not_exists']),        
        
        ]
    
    to_extracts_ktp_nik = [
        dict(extractor=get_all, extractor_args=dict(alias_names=['NIK_1'], pattern=r'.*'), evaluators=[nik_evaluator], multi_line_value=False, alias_names=['NIK_1'], tolerant_not_exists=[True], tess_config=f'--tessdata-dir {CURRENT_PATH} --psm 11 -l ktpnik3 -c tessedit_do_invert=0 ' + ONLY_CONFIG(string.digits)),         
    ]


    if img_full_path:
        img = cv2.imread(img_full_path)
        doc_parsers = {
            parse_doc: {'doc_str': ktp_str, 'to_extracts': to_extracts_ktp},
            simple_parse_one_all: {'to_extract': to_extracts_ktp_nik, 'universal_img_preprocessing': whimg}
        }

    elif not (img is None):
        img = img
        doc_parsers = {
            parse_doc: {'doc_str': ktp_str, 'to_extracts': to_extracts_ktp},
            simple_parse_one_all: {'to_extract': to_extracts_ktp_nik, 'universal_img_preprocessing': whimg}
        }

    elif ktp_str:
        doc_parsers = {
            parse_doc: {'doc_str': ktp_str, 'to_extracts': to_extracts_ktp},
        }
        to_extracts_ktp.insert(2, dict(extractor=get_str_keys_values, extractor_args=dict(key_str_list_list=[['NIK']], max_distances=configs['nik']['max_distances'], min_accuracies=configs['nik']['min_accuracies'], separators=[':'], re_chars_filter=[['[a-z|A-Z|0-9]', '[\:]']], alias_names=['NIK']), evaluators=[nik_evaluator], multi_line_value=False, alias_names=['NIK'], tolerant_not_exists=configs['nik']['tolerant_not_exists']))
    else:
        raise Exception(f'One of img_full_path, img, ktp_str should be filled.')
    
    if use_final_evaluator:
        final_evaluator = ktp_final_evaluator
    else:
        final_evaluator = lambda x: x


    result = blended_parse_doc(doc_parsers=doc_parsers, final_evaluator=final_evaluator, img=img)
    result['warning'] = get_warning_image(img)

    result['result'] = {k: result['result'][k] for k in result_orders if k in result['result']}

    return {k: result[k] for k in output_orders if k in result}
