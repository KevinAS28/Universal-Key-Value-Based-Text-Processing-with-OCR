import os
import pytesseract
import copy
import cv2


TO_READ_IMGS_DIR = 'to_read'
if not os.path.isdir(TO_READ_IMGS_DIR):
    os.mkdir(TO_READ_IMGS_DIR)

OUT_DIR = 'output'
if not os.path.isdir(OUT_DIR):
    os.mkdir(OUT_DIR)

def ocr0(img, config='-l ind --psm 6 -c tessedit_do_invert=0'):
    return pytesseract.image_to_string(img, config=config)

def ocr2(final, box, config=' -l ind -c tessedit_do_invert=0 --psm 6 '):
    final = copy.deepcopy(final)
    imgbox = final[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
    imgbox = cv2.copyMakeBorder(imgbox, 50, 50, 50, 50, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
    imgbox =  cv2.resize(imgbox, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(imgbox, config=config)
    return text, imgbox

def get_line_boxes(img, scale=0.5, config=' --psm 6 -l ind -c tessedit_do_invert=0', with_line_values=False):
    '''
    Get all boxes for each line in a img with some filterr
    '''

    img = copy.copy(img)
    img =  cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)

    n_boxes = len(d['level'])
    all_boxes = []
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        all_boxes.append((x, y, w, h))


    #Filter the line boxes
    h, w = img.shape[0:2]
    box_lines = []
    line_contains = dict()
    for new_box in all_boxes:
        
        #not a line
        if (new_box[3] >= 0.2*h):
            continue

        found = 0
        index = 0
        tolerant_size = 5
        for old_box in box_lines:
            if (new_box[0] >= old_box[0]-tolerant_size) and (new_box[0]+new_box[2] <= (old_box[0]+old_box[2]+tolerant_size)) and (new_box[1] >= old_box[1]-tolerant_size) and (new_box[1]+new_box[3] <= (old_box[1]+old_box[3]+tolerant_size)):
                found = 1
                break
            elif (old_box[0] >= new_box[0]-tolerant_size) and (old_box[0]+old_box[2] <= (new_box[0]+new_box[2]+tolerant_size)) and (old_box[1] >= new_box[1]-tolerant_size) and (old_box[1]+old_box[3] <= (new_box[1]+new_box[3]+tolerant_size)):
                found = 2
                break        
            index += 1
        
        if not found:
            if True:#new_box[2] >= 700:
                box_lines.append(new_box)
                line_contains[tuple(new_box)] = []
        elif found==1: #new new_box inside old new_box
            line_contains[tuple(box_lines[index])].append(new_box)
            pass
        elif found==2: #old new_box inside new new_box
            old_box = box_lines[index]
            old_contains = copy.copy(line_contains[tuple(old_box)]) + [old_box]
            del line_contains[tuple(old_box)]
            line_contains[tuple(new_box)] = old_contains
            box_lines[index] = new_box
    
    img_line_contains = dict()
    for box_key in line_contains:
        img_line_contains[tuple(box_key)] = []
        if with_line_values:
            for box_val in line_contains[tuple(box_key)]:
                img_line_contains[tuple(box_key)].append(box_val)    
    
    return img_line_contains