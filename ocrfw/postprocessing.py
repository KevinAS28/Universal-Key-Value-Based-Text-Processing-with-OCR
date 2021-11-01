import string
import re
import copy
import numpy as np
from nltk.metrics import edit_distance, accuracy
from itertools import combinations, permutations
import math
from ocr import *
from preprocessing import *

def get_re_pattern(string0, max_mistakes=1):
    replacements0 = dict(zip(list(string.printable), list(string.printable)))
    replacements1 = {
        '_': '\s*',
        ' ': '\s*',
        'l': '[i|l|1]',
        '1': '[i|l|1]',
        'i': '[i|l|1]',
        '0': '[o|O|0]',
        'o': '[o|O|0]',
        'O': '[o|O|0]',
        'A': '[A|4]',
        '4': '[A|4]',
        '?': '[7|\?]',
        '7': '[7|\?]',
        'E': '[E|3]',
        '3': '[E|3]',
        'E': '[E|3]',
        'S': '[S|5|s]',
        's': '[S|5|s]',
        '5': '[S|5|s]',
    }

    replacements = {**replacements0, **replacements1}

    aiueo = list('euU')
    for c in aiueo:
        replacements[c] = '.{{,{mm}}}'.format(mm=max_mistakes)
    
    specials = list('[]{}|()$^&*-+=.!/')
    for c in specials:
        replacements[c] = '\s{{,{mm}}}'.format(mm=max_mistakes) + '\\' + c + '\s{{,{mm}}}'.format(mm=max_mistakes)

    return ''.join([replacements[c] for c in string0])


# %%
def find_closest(to_find, points=[]):
    diff = list(map(lambda x: abs(x-to_find), points))
    return points[diff.index(min(diff))]


# %%
def split_index(string, splitters=[' ']):
    splitted_string = []
    index_list = []
    temp = ''
    i0 = 0
    for ci in range(len(string)):
        c = string[ci]
        if c in splitters:
            if not re.match(r'^\s*$', temp):
                splitted_string.append(temp)
                index_list.append((i0, ci))
            i0 = ci+1
            temp = ''
        else:
            temp += c
    if not re.match(r'^\s*$', temp):
        splitted_string.append(temp)
        index_list.append((i0, ci+1))
    
    return splitted_string, index_list


# %%
def get_str_keys_values(
        line,
        key_str_list_list,
        max_distances,
        min_accuracies,
        separators=[':'],
        re_chars_filter=None, #Set to None to make it all \w, set it to a re pattern to make it all the same
        alias_names = [],
        whitespace = ' ',
        line_preprocessing = lambda x: x,
        use_re_support = False,
        wo_space = False
    ):

    #without space key combinations
    if wo_space:
        key_str_list_list.extend([[i] for i in [''.join(i) for i in key_str_list_list]])
        max_distances.extend([[i] for i in [sum(i) for i in max_distances]])
        min_accuracies*=2
        alias_names*=2
        re_chars_filter*=2
        
    #re_chars_filter
    the_re_chars_filter = copy.copy(re_chars_filter)
    if (re_chars_filter is None) or (type(re_chars_filter)==str):
        re_chars_filter = []
        for k0 in key_str_list_list:
            if the_re_chars_filter is None:
                re_chars_filter.append([r'\w']*len(k0))
            else:
                re_chars_filter.append([the_re_chars_filter]*len(k0))
    
    #max_distances
    max_distances_list = []
    for i0, mm0 in enumerate(max_distances):
        mm_list = []
        for i1, mm1 in enumerate(mm0):
            if type(mm1)==float:
                the_key = key_str_list_list[i0][i1]
                mm_list.append(int(find_closest(mm1, np.arange(0, 1, 1/len(the_key)))/(1/len(the_key))))
            elif type(mm1)==int:
                mm_list.append(mm1)
            else:
                raise ValueError(f'max_distances should be 2 dimensional list containing float/int, found: ({str(type(mm1))}) {str(mm1)}')
        max_distances_list.append(mm_list)
    
    line_preprocessed = line_preprocessing(line)
    line_splitted, line_splitted_index = split_index(line_preprocessed, [whitespace, *separators])
    key_names = []

    #=Find all keys=
    all_found_keys = dict()
    founded_index = []
    log_verb = []

    for list_key_index in range(len(key_str_list_list)):

        if list_key_index in founded_index:
            log_verb.append(f'Skipping: {list_key_index}')
            continue

        the_list_key = key_str_list_list[list_key_index]

        log_verb.append(f'the_list_key: {the_list_key}')

        if list_key_index < len(alias_names):
            the_key_name = alias_names[list_key_index]
        else:
            the_key_name = tuple(the_list_key)
        
        key_names.append(the_key_name)

        if len(the_list_key) > len(line_splitted):
            all_found_keys[the_key_name] = [False, f'Lenght of key {str(the_list_key)}={str(len(the_list_key))} > Lenght of line {str(line_splitted)}={str(len(line_splitted))}']
            continue
        
        for sliced_key_index in range((len(line_splitted)-len(the_list_key))+1):
            success = 1 #0=success, 1=keep going, 2=fail

            start_index_key = sliced_key_index
            stop_index_key = start_index_key + len(the_list_key)
            sliced_line = line_splitted[start_index_key:stop_index_key]
            sliced_line = [''.join(re.findall(re_chars_filter[list_key_index][i], sliced_line[i])) for i in range(len(sliced_line))]
            joined_sliced_line = whitespace.join(sliced_line)
            joined_list_key = whitespace.join(the_list_key)
            overall_scores = []

            #regex first
            if (success==1)  and (use_re_support):
                pattern0 = '\s*'.join(['({})'.format(get_re_pattern(i, 2)) for i in the_list_key])
                result = re.search(pattern0, joined_sliced_line)
                if result:
                    overall_scores.append(50)
                    sliced_line = list(result.groups())
                    joined_sliced_line = whitespace.join(sliced_line)
                    log_verb.append(f'Success: regex {pattern0} <-> {joined_sliced_line}')
                else:
                    success = 2
                    log_verb.append(f'Failed: regex {pattern0} <-> {joined_sliced_line}')

            #check distances count
            if (success == 1):
                the_max_distances = max_distances_list[list_key_index]
                
                distances_result = [edit_distance(the_list_key[lki], sliced_line[lki]) for lki in range(len(the_list_key))]
                distance_scores = [distances_result[i] for i in range(len(distances_result)) if distances_result[i] <= the_max_distances[i]]
                if (len(distance_scores) < len(the_max_distances)):
                    if (not (the_key_name in all_found_keys)):
                        all_found_keys[the_key_name] = [False, f'distances beyond maximum: {distances_result}']
                    elif (the_key_name in all_found_keys) and (not all_found_keys[the_key_name][0]):
                        all_found_keys[the_key_name] = [False, f'distances beyond maximum: {distances_result}']
                    log_verb.append(f'Failed: {the_list_key} <-> {sliced_line} distances beyond maximum: {distances_result}')
                    success = 2
                else:
                    log_verb.append(f'Success: distance {the_list_key} <-> {sliced_line} ')
                    overall_scores.append((1-(sum(distance_scores)/len(joined_list_key)))*100)

            #check accuracy
            if success == 1:
                min_acc = min_accuracies[list_key_index]
                #To check the accuracy, the string lenghts should be equal
                if len(joined_sliced_line) < len(joined_list_key):
                    line_accuracy = joined_sliced_line + (whitespace*(len(joined_list_key)-len(joined_sliced_line)))
                elif len(joined_sliced_line) > len(joined_list_key):
                    line_accuracy = joined_sliced_line[:len(joined_list_key)]
                else:
                    line_accuracy = joined_sliced_line
                the_acc = accuracy(joined_list_key, line_accuracy)
                
                if the_acc < min_acc:
                    if (not (the_key_name in all_found_keys)):
                        all_found_keys[the_key_name] = [False, f'Accuracy below minimum']
                    elif (the_key_name in all_found_keys) and (not all_found_keys[the_key_name][0]):
                        all_found_keys[the_key_name] = [False, f'Accuracy below minimum']
                    log_verb.append(f'Failed: accuracy {joined_list_key} <-> {line_accuracy} Accuracy below minimum')
                    success = 2
                else:
                    log_verb.append(f'Success: accuracy {joined_list_key} <-> {line_accuracy}: {the_acc}')
                    success = 0
                    overall_scores.append(the_acc*100)
            
            
            if success == 0:
                log_verb.append(f'Success {joined_list_key} <-> {sliced_line} <-> {sliced_key_index}')
                overall_scores_num = sum(overall_scores)/len(overall_scores)

                founded_index.append(list_key_index)
                
                if the_key_name in all_found_keys:
                    if all_found_keys[the_key_name][0]:
                        if overall_scores_num > all_found_keys[the_key_name][2]:
                            all_found_keys[the_key_name] = [True, sliced_line, overall_scores_num, overall_scores, line_splitted_index[start_index_key:stop_index_key]]
                    else:
                        all_found_keys[the_key_name] = [True, sliced_line, overall_scores_num, overall_scores, line_splitted_index[start_index_key:stop_index_key]]
                else:
                    all_found_keys[the_key_name] = [True, sliced_line, overall_scores_num, overall_scores, line_splitted_index[start_index_key:stop_index_key]]
        
    #=End find all keys=

    #=Find all values=#
    
    result = dict()
    all_found_keys_list = [[name, *all_found_keys[name]] for name in all_found_keys if all_found_keys[name][0]]
    all_found_keys_list.append(['', None, None, None, None, [[len(line), len(line)]]])
    all_found_keys_list.insert(0, ['', None, None, None, None, [[0, 0]]])   
    for keys_index in range(1, len(all_found_keys_list[1:-1])+1):
        current_key_indexes = all_found_keys_list[keys_index][-1]
        next_key_indexes = all_found_keys_list[keys_index+1][-1]
        key_result = all_found_keys_list[keys_index][0]
        value_result = line[current_key_indexes[-1][-1]:next_key_indexes[0][0]]
        result[key_result] = value_result
    
    #=End find all values=#
    
    return [True, result]


# %%
def get_all(line, alias_names=[], pattern='.*'):
    result = dict()
    for an in alias_names:
        result[an] = line
    if re.search(pattern, line):
        return [True, result]
    return [False, result]

# %% [markdown]
# # Evaluator functions
# - To clean and validate the values

# %%
def letters_evaluator(s0, allowed_chars=['a-z', 'A-Z', '0-9']):
    typo_map = {
        '[4]': 'a',
        '[5]': 's',
        '[?]': '7',
        '[8]': 'b',
        '[0]': 'o',
        '[1]': 'i',
        '[\+]': 't',
        '[6|9]': 'g',
        '[3]': 'e',
    }

    allowed_chars_regex = '[{}]'.format('|'.join(['^'+i for i in allowed_chars]))
    string_splitted = [i for i in re.split(allowed_chars_regex, s0) if len(i) > 0]
    string_result = []
    for word in string_splitted:
        string_temp = ''
        for c in word:
            found = False
            for pattern in typo_map:
                if re.match(pattern, c):
                    found = True
                    string_temp += typo_map[pattern]
                    break
            if not found:
                string_temp += c
        string_result.append(string_temp)

    return [True, ' '.join(string_result)]


# %%
def todigits_typo(digtypo, ignore_not_number=True):
    mapping = {
        tuple('1liI!'): '1',
        tuple('2P'): '2',
        tuple('3eE'): '3',
        tuple('4AaY'): '4',
        tuple('5Ss'): '5',
        tuple('6b'): '6',
        tuple('7?'): '7',
        tuple('8B'): '8',
        tuple('9g'): '9',
        tuple('0DoO'): '0'
    }

    digits = ''
    for c in digtypo:
        found = False
        for k, v in mapping.items():
            if c in k:
                digits += v
                found = True
                break
        if (not found) and (not ignore_not_number):
            digits += c

    return digits


# %%
def nodigits_typo(nod):
    b = re.findall(r'[N|n][O|o|0]\s*\.{0,1}\s*[^\s]*', nod)
    for no in b:
        nodig = re.search(r'([N|n][O|o|0]\s*\.{0,1}\s*)([^\s]*)', no).groups()
        nod = nod.replace(no, f'NO. {todigits_typo(nodig[1])}')
    return nod


# %%
def ktp_final_evaluator(result_dict):
    keys = list(result_dict.keys())

    if (('NIK' in keys) or ('NIK_1' in keys)):
        scoring_functions = [
            lambda nik: len(nik)==16,
            
        ]
        if  ('Tanggal Lahir' in keys) and ('Jenis kelamin' in keys):
            def _tl(tl):
                tl = str(tl)
                if len(tl) == 0:
                    return '00'
                elif len(tl) == 1:
                    return '0' + tl
                return tl
            
            tl = [str(i) for i in result_dict['Tanggal Lahir']]
            jk = result_dict['Jenis kelamin']
            tl1 = ''.join([_tl(i) for i in [int(tl[0]) +
                                            (40 if 'PEREMPUAN' in jk else 0), tl[1], tl[2][-2:]]])

            scoring_functions.append(lambda nik: tl1 in nik)

        all_nik_ktp = []
        all_nik_keys = ['NIK', 'NIK_1']
        for nik_key in all_nik_keys:
            if nik_key in result_dict:
                all_nik_ktp.extend([i[1][0] for i in result_dict[nik_key]])

        scores_nik = dict()
        
        for nik in all_nik_ktp:
            
            score = 0
            for fun in scoring_functions:
                if fun(nik):
                    score+=1

            if score in scores_nik:
                scores_nik[score].append(nik)
            else:
                scores_nik[score] = [nik]
        
        top_3_nik_scores = sorted(list(scores_nik.keys()))[::-1][:3]

        scores_nik_list = [nik for score in top_3_nik_scores for nik in scores_nik[score]]
        
        result_dict['NIK'] = scores_nik_list
        if len(scores_nik) > 0:
            return {'success': True, 'result': result_dict}
        else:
            return {'success': False, 'result': result_dict, 'all_error_codes': [1], 'all_error_messages': [f'Final evaluator: there is no valid NIK']}

    else:
        return {'success': False, 'result': result_dict, 'all_error_codes': [2], 'all_error_messages': [f'Final evaluator: there is no NIK or Tanggal Lahir or Jenis Kelamin or PROVINSI']}

# %% [markdown]
# # Parser functions

# %%
def parse_doc(img, to_extracts, ocr=ocr0, doc_str=None, final_evaluator=None):
    final_success = True

    # Final result
    results_json = dict()
    
    #Just to test post-processing?
    if doc_str:
        real_all_lines = doc_str.split('\n')
        list_all_lines = copy.copy(real_all_lines)
    else:
        real_all_lines = ocr(preprocessing3(img)).split('\n')
        list_all_lines = copy.copy(real_all_lines)
    
    all_error_messages = []
    all_error_codes = []

    for te in to_extracts:
        line_index = 0
        temp_result = None
        error_level = 0
        success_inserted = 0
        error_result = []
        

        for line in list_all_lines:
            #No blank space allowed
            if re.match(r'^\s*$', line):
                continue

            extractor = te['extractor']
            extractor_args = te['extractor_args']
            extractor_args['line'] = line

            line_result = extractor(**extractor_args)
            if line_result[0]: #Extractor success
                success = False
                temp_result = line_result[1]
                all_required_exist = True

                names_to_check = dict()
                for ani in range(len(te['alias_names'])):
                    an = te['alias_names'][ani]

                    if (not (an in temp_result)) and (not (te['tolerant_not_exists'][ani])):
                        all_required_exist = False
                        success = False
                        if error_level <= 0:
                            error_result = [False, f'{an} is not exist while its not tolerant to not exist', line]
                            error_level = 0
                        break
                    elif (not (an in temp_result)) and ((te['tolerant_not_exists'][ani])):
                        names_to_check[an] = False
                    else:
                        names_to_check[an] = True

                if all_required_exist:
                    for alias_index in range(len(te['alias_names'])):
                        a_name = te['alias_names'][alias_index]
                        if alias_index < len(te['evaluators']): #There is an evaluator for this value
                            
                            evaluator = te['evaluators'][alias_index]

                            if not names_to_check[a_name]: #Not exist and it's tolerant to not exist
                                continue

                            temp_result = line_result[1]
                            eval_result = evaluator(temp_result[a_name])
                            
                            if eval_result[0]: #Passed evaluator
                                success = True
                                success_inserted += 1
                                if type(eval_result[1]) == dict:
                                    results_json = {**results_json, **eval_result[1]}
                                else:
                                    results_json[a_name] = eval_result[1]           
                            else:
                                if error_level <= 2:
                                    error_result = [False, f'{a_name} not passed evaluator', temp_result, eval_result, line,]
                                    error_level = 2
                        
                        else: #No evaluator
                            
                            success = True
                            if not names_to_check[a_name]: #Not exist and it's tolerant to not exist
                                if error_level < 1:
                                    error_result = [False, f'{a_name} not exist and its not tolerant to not exist', line]
                                    error_level = 1                                
                                continue                            
                            success_inserted += 1
                            
                            temp_result = line_result[1]
                            results_json[a_name] = temp_result[a_name]

                if success:
                    del list_all_lines[list_all_lines.index(line)]
                    break
                else:
                    if type(temp_result)==list:
                        temp_result.insert(1, 'Not success')
                        if error_result:
                            error_result.insert(1, 'Not success')
                        else:
                            error_result = [False, 'Not success']
                    else:
                        temp_result['status'] = 'Not success'            
                    
                
            else:
                pass
            line_index += 1
            

        must_exists_count = sum([1 for i in te['tolerant_not_exists'] if not i])
        if success_inserted!=len(te['alias_names']):
            if success_inserted < must_exists_count:
                all_error_codes.append(0)
                all_error_messages.append(f'One or more keys in {te["alias_names"]} not found. Must exists: {must_exists_count}, founded: {success_inserted} <=> {error_result}')
                final_success = False


    if final_evaluator:
        final_result = final_evaluator(results_json)
        
        if not final_result['success']:
            all_error_codes += final_result['error_codes']
            all_error_messages += final_result['error_messages']

            del final_result['error_codes']
            del final_result['error_messages']

        if final_success and (not final_result['success']):
            final_success = False
    else:
        final_result = {'result': results_json}
    
    if not final_success:
        final_result['all_error_codes'] = all_error_codes
        final_result['all_error_messages'] = all_error_messages

    final_result['model_result'] = {f'pd{i}': line for i, line in enumerate(real_all_lines)}
    final_result['success'] = final_success

    return final_result


# %%
def parse_doc_new0(img, to_extracts, final_evaluator=None, keep_looking=False):

    # Final result
    results_json = dict()

    img1 = preprocessing3(img)
    boxes = list(get_line_boxes(img1, scale=0.5).keys())

    for te in to_extracts:
        temp_result = []
        error_level = 0
        success_inserted = 0
        error_result = []
        tess_config = te['tess_config']
        box_index = -1
        success = False
        for box in boxes:
            box = [i*2 for i in box]
            if success and (not keep_looking):
                break
            box_index += 1
            extra_lines = ocr2(img1, box, config=tess_config).split('\n')
            for line in extra_lines:
                if re.match(r'^\s*$', line):
                    continue

                extractor = te['extractor']
                extractor_args = te['extractor_args']
                extractor_args['line'] = line
                line_result = extractor(**extractor_args)

                if line_result[0]: #Extractor success
                    temp_result = line_result[1]
                    all_required_exist = True

                    names_to_check = dict()
                    for ani in range(len(te['alias_names'])):
                        an = te['alias_names'][ani]

                        if (not (an in temp_result)) and (not (te['tolerant_not_exists'][ani])):
                            all_required_exist = False
                            success = False
                            if error_level <= 0:
                                error_result = [False, f'{an} is not exist while its not tolerant to not exist', line]
                                error_level = 0
                            break
                        elif (not (an in temp_result)) and ((te['tolerant_not_exists'][ani])):
                            names_to_check[an] = False
                        else:
                            names_to_check[an] = True

                    if all_required_exist:
                        for alias_index in range(len(te['alias_names'])):
                            a_name = te['alias_names'][alias_index]
                            if alias_index < len(te['evaluators']): #There is an evaluator for this value
                                
                                evaluator = te['evaluators'][alias_index]

                                if not names_to_check[a_name]: #Not exist and it's tolerant to not exist
                                    continue

                        
                                temp_result = line_result[1]
                                eval_result = evaluator(temp_result[a_name])
                                
                                if eval_result[0]: #Passed evaluator
                                    success = True
                                    success_inserted += 1
                                    if type(eval_result[1]) == dict:
                                        results_json = {**results_json, **eval_result[1]}
                                    else:
                                        results_json[a_name] = eval_result[1]           
                                else:
                                    if error_level <= 2:
                                        error_result = [False, f'{a_name} not passed evaluator', temp_result, eval_result, line,]
                                        error_level = 2
                            
                            else: #No evaluator
                                success = True
                                if not names_to_check[a_name]: #Not exist and it's tolerant to not exist
                                    if error_level < 1:
                                        error_result = [False, f'{a_name} not exist and its not tolerant to not exist', line]
                                        error_level = 1                                
                                    continue                            
                                success_inserted += 1
                                
                                temp_result = line_result[1]
                                results_json[a_name] = temp_result[a_name]
                
                if success:
                    
                    if len(extra_lines) <= 1:
                        del boxes[box_index]
                else:
                    if type(temp_result)==list:
                        temp_result.insert(1, ['Not success', te['alias_names'], extra_lines])
                        if error_result:
                            pass
                        else:
                            error_result = [False, 'Not success']
                    else:
                        temp_result['status'] = 'Not success'
                
            else:
                pass
            
        must_exists_count = sum([1 for i in te['tolerant_not_exists'] if not i])
        if success_inserted!=len(te['alias_names']):
            if success_inserted < must_exists_count:
                return {'success': False, 'error_code': 0, 'error_message': f'One or more keys in {te["alias_names"]} not found. Must exists: {must_exists_count}, founded: {success_inserted} <=> {error_result}', 'result': results_json, 'line_result': line_result, 'to extracts': te}

    if final_evaluator:
        final_result = final_evaluator(results_json)
    else:
        final_result = {'success': True, 'result': results_json}
    return final_result


# %%
def simple_parse_one_all(img, to_extract,ocr=ocr0, universal_img_preprocessing=lambda img: img, img_preprocessings=[preprocessing3, preprocessing1]):
    if type(to_extract)==list:
        to_extract = to_extract[0]
        
    all_result = []
    all_real_lines = dict()
    extractor = to_extract['extractor']
    extractor_args = to_extract['extractor_args']

    for preprocessing in img_preprocessings:
        pre_img = preprocessing(universal_img_preprocessing(img))
        real_lines = ocr(pre_img, config=to_extract['tess_config']).split('\n')
        all_real_lines[preprocessing.__name__] = real_lines
        for line in real_lines:
            extractor_args['line'] = line
            line_result = extractor(**extractor_args)
            
            if line_result[0]: 
                new_result = line_result[1]
                if type(new_result)==dict:
                    new_result = [v for _, v in new_result.items()]
                if len(to_extract['evaluators'])  > 0:
                    for evaluator in to_extract['evaluators']:
                        eval_result = evaluator(new_result)
                        if eval_result[0]:
                            all_result.extend(eval_result[1])
                        else:
                            pass
                
                else:
                    all_result.append(new_result)
            
    return {'result': {to_extract['alias_names'][0]: all_result}, 'model_result': all_real_lines, 'success': True if len(all_result)>0 else False}

# if __name__=='__main__':
#     print(nik_evaluator())
#     te = dict(extractor=get_all, extractor_args=dict(alias_names=['NIK_1'], pattern=r'^[\d]{14,16}$'), evaluators=[nik_evaluator], multi_line_value=False, alias_names=['NIK_1'], tolerant_not_exists=[True], tess_config=f'--tessdata-dir {CURRENT_PATH} --psm 11 -l ktpnik3 -c tessedit_do_invert=0 ' + ONLY_CONFIG(string.digits))
#     print(simple_parse_one_all(cv2.imread('ktp_data/ktp.JPG'), te))


# %%
def blended_parse_doc(doc_parsers, final_evaluator=None, **kwargs):
    fields_to_update = {'result': dict(), 'model_result': dict()}

    result = {field: dict() for field in fields_to_update.keys()}
    success = True
    for parser in doc_parsers:
        args = {**kwargs, **doc_parsers[parser]}
        new_result = parser(**args)

        for field in fields_to_update.keys():
            if not (field in new_result):
                continue        
            for k0 in new_result[field].keys():
                if k0 in fields_to_update[field].keys():
                    continue
                fields_to_update[field][k0] = new_result[field][k0]

        
        result = {**result, **new_result}
        if not new_result['success']:
            success = False
            
    result = {**result, **fields_to_update}

    result = {**result, **final_evaluator(result['result'])}

    if not success:
        result['success'] = False
        
    
    return result
