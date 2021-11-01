# %%
from ocrfw.postprocessing import *

# %%
def provinsi_evaluator(provinsi, max_mistakes=5, min_accuracy=0.5):
    province_names = dict(zip([re.sub(r'\s', '', i.upper())
                               for i in PROVINCIES_LIST], PROVINCIES_LIST))
    provinsi = re.sub(r'\s', '', provinsi.upper())
    scores_names = dict()

    for p_n in province_names:
        # To check accuracy, both string length should be equal
        if len(provinsi) < len(p_n):
            prov = provinsi + (' '*(len(p_n)-len(provinsi)))
        elif len(provinsi) > len(p_n):
            prov = provinsi[:len(p_n)]
        else:
            prov = provinsi

        the_accuracy = accuracy(prov, p_n)
        mistakes = edit_distance(p_n, provinsi)

        if (mistakes <= max_mistakes) and (the_accuracy >= min_accuracy):
            scores = [100-(mistakes*(100/max_mistakes)), the_accuracy*100]
            scores = sum(scores)/len(scores)
            scores_names[scores] = province_names[p_n]

    if len(scores_names) == 0:
        return [False, provinsi, f'No province name found: {provinsi}', scores_names]
    else:
        closest_scores = dict()
        def set_key_value(key, value): closest_scores[key] = value
        [set_key_value(i[0], i[1]) for i in [[abs(100-i), i]
                                             for i in list(scores_names.keys())]]

        return [True, scores_names[closest_scores[sorted(closest_scores)[0]]], scores_names]


# %%
def nik_evaluator(nik, max_distance=1):
    nik_lenght = 16
    all_nik = [re.sub(r'[^0-9]', '', i)
               for i in todigits_typo(nik, False).split(' ')]
    
    possible_nik = dict()
    for i in range(1, len(all_nik)+1):
        nik_comb = [''.join(map(str, j)) for j in combinations(all_nik, i)]
        for nc in nik_comb:
            len_nc = len(nc)
            nik_dist = abs(nik_lenght-len_nc)
            if nik_dist <= max_distance:
                prov_code_valid = False
                for i in range(2):
                    if nc[i:i+2] in PROV_CITY_KEC:
                        prov_code_valid = True
                        break
                if prov_code_valid:
                    if not (len_nc in possible_nik):
                        possible_nik[nik_dist] = []
                    possible_nik[nik_dist].append(nc)
    
    result = {k: possible_nik[k] for k in sorted(possible_nik)}
    
    if len(possible_nik) > 0:
        result_list = [[k, v] for k, v in result.items()]
        return [True, result_list]
    else:
        possible_nik_list = [[k, v] for k, v in possible_nik.items()]
        return [False, possible_nik_list]


# %%
def ttl_evaluator(ttl):

    similar_digits = {
        '1': '7',
        '7': '1',
        '6': '8',
        '8': '6',
        '9': '5',
        '2': '3',
        '3': '2',
        '4': '9',
        '9': '4',
        '5': '0',
        '0': '5'
    }

    ttl_p = r'(.*)(\d{2,2}).*(\-*).*(\d{2,2}).*(\-*).*(\d{4,4})'
    ttl_re = re.search(ttl_p, ttl)
    if ttl_re:
        ttl_re = ttl_re.groups()
        tempat = letters_evaluator(ttl_re[0].rstrip().lstrip())[1]
        tgllahir = [int(''.join(re.findall(r'\d', i))) for i in list(ttl_re[1:]) if not re.match(r'^\s*$', i)]

        new_tgllahir = []
        temp_tgllahir = ''
        if tgllahir[0] > 31:
            
            digit_0, _ = str(tgllahir[0])
            if int(digit_0) > 3:
                temp_tgllahir += similar_digits[digit_0]
            else:
                temp_tgllahir += digit_0
        else:
            temp_tgllahir += str(tgllahir[0])

        new_tgllahir.append(temp_tgllahir)
        temp_tgllahir = ''
        
        if tgllahir[1] > 12:
            digit_0, digit_1 = str(tgllahir[1])
            if int(digit_0) > 1:
                temp_tgllahir += similar_digits[digit_0]
            else:
                temp_tgllahir += digit_0
            
            if int(digit_1) > 2:
                temp_tgllahir += similar_digits[digit_1]
            else:
                temp_tgllahir += digit_1
        else:
            temp_tgllahir += str(tgllahir[1])

        new_tgllahir.append(temp_tgllahir)
        temp_tgllahir = ''
        
        if tgllahir[2] < 1920:
            digit_0, digit_1, digit_2, digit_3 = str(tgllahir[2])
            if int(digit_0) < 1:
                temp_tgllahir += similar_digits[digit_0]
            else:
                temp_tgllahir += digit_0

            if int(digit_1) < 9:
                temp_tgllahir += similar_digits[digit_1]
            else:
                temp_tgllahir += digit_1

            temp_tgllahir += digit_2 + digit_3
        else:
            temp_tgllahir += str(tgllahir[2])
        
        new_tgllahir.append(temp_tgllahir)

        return [True, {'Tempat': tempat, 'Tanggal Lahir': new_tgllahir}]
    else:
        return [False, ttl, f'ttl not match with pattern: {ttl_p} ']


# %%
def kab_kota_evaluator(city, max_mistakes=5, min_accuracy=0.5):
    
        
    city_names = dict(zip([re.sub(r'\s', '', i.upper())
                           for i in CITIES_LIST], CITIES_LIST))
    city = re.sub(r'\s', '', city.upper())
    if len(city) <= 3:
        return [False, f'Lenght of city should be > 3, found: {len(city)}']
    scores_names = dict()

    for c_n in city_names:
        # To check accuracy, both string length should be equal
        if len(city) < len(c_n):
            city0 = city + (' '*(len(c_n)-len(city)))
        elif len(city) > len(c_n):
            city0 = city[:len(c_n)]
        else:
            city0 = city

        the_accuracy = accuracy(city0, c_n)
        mistakes = edit_distance(c_n, city)

        if (mistakes <= max_mistakes) and (the_accuracy >= min_accuracy):
            scores = [100-(mistakes*(100/max_mistakes)), the_accuracy*100]
            scores = sum(scores)/len(scores)
            scores_names[scores] = city_names[c_n]

    if len(scores_names) == 0:
        return [True, city, f'No kab/kota name found: {city}', scores_names]
    else:
        closest_scores = dict()
        def set_key_value(key, value): closest_scores[key] = value
        [set_key_value(i[0], i[1]) for i in [[abs(100-i), i]
                                             for i in list(scores_names.keys())]]

        return [True, scores_names[closest_scores[sorted(closest_scores)[0]]], scores_names]


# %%
def jk_evaluator(jk, max_distances=4):
    jk = jk.upper()
    to_removes_re = [r'[^a-z|^A-Z|^0-9]']
    for trr in to_removes_re:
        jk = re.sub(trr, '', jk)
    key_val = {
        'LAKILAKI': 'LAKI-LAKI',
        'PEREMPUAN': 'PEREMPUAN'
    }
    result = dict()
    for k in key_val:
        r = edit_distance(k, jk)
        result[r] = key_val[k]
    if len(result) > 0:
        result_sorted = sorted(result)
        if result_sorted[0] <= max_distances:
            return [True, result[result_sorted[0]]]
        return [False, result, result_sorted]
    else:
        return [False, result]


# %%
def darah_evaluator(darah):
    posibilities = {
        '[4|A]': 'A',
        '[B|8|9]': 'B',
        '[4|A][B|8|9]': 'AB',
        '[o|O|0]': 'O',
    }

    if len(darah) == 0:
        return [True, '-']
    for c in darah:
        for p in posibilities:
            pattern = '{}'.format(p)
            if re.match(pattern, c):
                return [True, posibilities[p]]
    
    if re.search(r'.*[\-|\_]+.*', darah):
        return [True, '-']

    return [False, f'No darah match: {darah}']
    


# %%
def rtrw_evaluator(rtrw):
    replacements0 = {
        'l': '1',
        'i': '1',
        'I': '1',
        'o': '0',
        'O': '0',
        'A': '4',
        '?': '7',
        'E': '3',
        'S': '5',
        's': '5',
        '/': '/',
        '\\': '\\',
    }
    rtrw = todigits_typo(rtrw, ignore_not_number=False)
    replacements0 = {**replacements0, **
                     (dict(zip(tuple(string.digits), tuple(string.digits))))}
    rtrw0 = ''.join([replacements0[c] for c in rtrw if c in replacements0])
    re_result = re.search(
        r'[^\d]*(\d{1,})[^\d]*[\/|\\]+[^\d]*(\d{1,})[^\d]*', rtrw0)
    if re_result:
        return [True, tuple(re_result.groups()), rtrw0]

    nums = ''.join(re.findall(r'[\d]', rtrw0))
    return [True, [nums[:int(len(nums)//2)], nums[int(len(nums)//2):]]]


# %%
from math import ceil

def rs_p(char_lists=[]):
    p = ''
    
    if len(char_lists)==0:
        return p
    first_chars = [c for cl in char_lists[1:] for c in cl if c not in char_lists[0]]
    not_first_chars_p = '[{}]'.format('|'.join(['^'+c for c in first_chars]))

    p+=not_first_chars_p

    for cl in char_lists[:-1]:
        not_next_chars = ''#'[{}]*'.format('|'.join(['^'+i for i in cl]))
        p+='([{}]*){}'.format('|'.join(cl), not_next_chars)
    
    cl = char_lists[-1]
    not_next_chars = ''#'[{}]*'.format('|'.join(['^'+i for i in cl]))#'[{}]*'.format('|'.join(['^'+i for i in cl]))
    p+='([{}]*){}'.format('|'.join(cl), not_next_chars)
    return p

def agama_evaluator(agama):
    agama = agama.upper()
    to_removes_re = [r'[^a-z|^A-Z|^0-9|^\+|^?]']
    for trr in to_removes_re:
        agama = re.sub(trr, '', agama)

    key_val = [
        ('iIl1 sS5 li1I 4A mM'.split(), "ISLAM"),
        ('Kk 4A'.split() + [['T', '7', r'\+', r'\?']] + 'H oO0 LI1 Kk'.split(), 'KATHOLIK'),
        ('Kk R il1I 5s'.split() + [['T', '7', r'\+', r'\?']] + 'E3 NM'.split(), 'KRISTEN'),
        ('H ilI1 NM D U'.split(), 'HINDU'),
        ('B86 U D D H A4'.split(), 'BUDDHA'),
        ('Kk Oo0 NM G H U C U'.split(), 'KONGHUCU'),
    ]

    result = dict()
    for k, v in key_val:
        the_chars = [i for c in k for i in c]
        txt = ''.join([c for c in agama if c in the_chars])
        p = rs_p(k)
        try:
            groups = [i for i in re.search(p, txt).groups() if len(i)>0]
        except AttributeError:
            continue
        if len(groups)>=(ceil(len(k)/2)):
            result[len(groups)] = v

    if len(result) > 0:
        result_sorted = sorted(result)
        return [True, result[result_sorted[-1]]]
    else:
        return [False, result]


# %%
def kawin_evaluator(kawin, max_distances=5):
    kawin = kawin.upper()
    to_removes_re = [r'[^a-z|^A-Z|^0-9]']
    for trr in to_removes_re:
        kawin = re.sub(trr, '', kawin)
    key_val = {
        'KAWIN': 'KAWIN',
        'BELUMKAWIN': 'BELUM KAWIN',
        'CERAIHIDUP': 'CERAI HIDUP',
        'CERAIMATI': 'CERAI MATI',
    }
    result = dict()
    for k in key_val:
        if len(kawin) >= len(k):
            for i in range((len(kawin)-len(k))+1):
                s = i
                e = i+len(k)
                r = edit_distance(k, kawin[s:e])
                result[r] = key_val[k]
        else:
            r = edit_distance(k, kawin)
            result[r] = key_val[k]

    if len(result) > 0:
        result_sorted = sorted(result)
        if result_sorted[0] <= max_distances:
            return [True, result[result_sorted[0]]]
        return [False, result, result_sorted]
    else:
        return [False, result]


# %%
def pekerjaan_evaluator(job, recommended_distance=1, max_distance=0.25):
    job0 = letters_evaluator(job)[1].upper()

    smallest_dist = np.inf
    smallest_name = False
    for j in JOBS:
        j0 = j
        j = letters_evaluator(j)[1].upper()
        md = math.floor(len(j)*max_distance)

        if len(job0) > len(j):
            job = job0[:len(j)]
        else:
            job = job0

        distance = edit_distance(j, job)
        if distance <= recommended_distance:
            return [True, j0, job]

        if (distance <= md) and (distance < smallest_dist):
            smallest_dist = distance
            smallest_name = j0

    if smallest_name:
        return [True, smallest_name, smallest_dist, job0]

    return [False, smallest_name, smallest_dist, job0]


# %%
def berlaku_evaluator(berlaku, max_distances=6):
    berlaku = re.sub('\s', '', berlaku.upper())

    #Tanggal?
    replacements0 = {
        '_': '',
        ' ': '',
        'l': '1',
        'i': '1',
        'I': '1',
        'o': '0',
        'O': '0',
        'A': '4',
        '?': '7',
        'E': '3',
        'S': '5',
        's': '5'
    }

    br = ''.join(
        i if not i in replacements0 else replacements0[i] for i in berlaku)
    t_p = r'(.*)(\d{2,2}).*(\-*).*(\d{2,2}).*(\-*).*(\d{4,4})'
    re_result = re.search(t_p, br)
    if re_result:
        groups = re_result.groups()
        return [True, [groups[1], groups[3], groups[5]]]

    # SEUMUR HIDUP?
    replacements1 = {
        '1': 'I',
        '3': 'E',
        '0': 'O',
        '4': 'A',
        '5': 'S',
    }

    sh = 'SEUMURHIDUP'
    br = ''.join(
        i if not i in replacements1 else replacements1[i] for i in berlaku)
    distance = edit_distance(sh, ''.join(re.findall('[A-Z]', br)))
    if distance <= max_distances:
        return [True, 'SEUMUR HIDUP']

    return [False, berlaku, 'Not match with pattern or SEUMUR HIDUP']


# %%
def kn_evaluator(kn, recommended_distance=1, max_distance=0.25):
    kn0 = letters_evaluator(kn)[1].upper()

    # Most likely WNI
    if 'WNI' in kn0:
        return [True, 'WNI']
    if 'WM' in kn0:
        return [True, 'WNI']

    # Not WNI?
    smallest_dist = np.inf
    smallest_name = False
    for ct in COUNTRIES:
        ct = ct.upper()
        md = math.floor(len(ct)*max_distance)

        distance = edit_distance(ct, kn0)
        if distance <= recommended_distance:
            return [True, ct, kn0]

        if (distance <= md) and (distance < smallest_dist):
            smallest_dist = distance
            smallest_name = ct

    if smallest_name:
        return [True, smallest_name, smallest_dist, kn0]

    return [False, f'{kn0} is not a valid kewarganegaraan', smallest_name, smallest_dist, kn0]

