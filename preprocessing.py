import copy
import os
import cv2
import numpy as np
import imutils
from statistics import mean as avg


whimg_index = 0
def whimg(img0, start_percent=1, writeimg=False):
    global whimg_index
    img = copy.deepcopy(img0)
    img[int(img.shape[0]*start_percent):] = (255, 255, 255)
    if writeimg:
        out_dir = 'whimg'
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        cv2.imwrite(os.path.join(out_dir, f'{whimg_index}.jpg'), img)
        whimg_index += 1
    return img

preprocessing_index = 0

def preprocessing1(img0, scale=2, manual_crop=True, write_img=False):
    if manual_crop:
        h, w = img0.shape[0:2]
        start_row = 0
        start_col = 0
        end_row = int(h*.85)
        end_col = int(w*.75)
        img0 = img0[start_row:end_row, start_col:end_col]            

    img2 = img0
    img2 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    thresh, img3 = cv2.threshold(img2, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img3 = cv2.addWeighted(img3, 2, np.zeros(img3.shape, img3.dtype), 0, 0)
    kernel = np.ones((1, 1), np.uint8)
    img4 = cv2.dilate(img3, kernel, iterations=1)
    img4 = cv2.erode(img4, kernel, iterations=1)
    img6 = cv2.fastNlMeansDenoising(img4,None, 50, 7, 21)
    img7 = cv2.resize(img6, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if write_img:
        global preprocessing_index
        cv2.imwrite(os.path.join(OUT_DIR, f'preprocessing1_{preprocessing_index}.jpeg'), img7)
        preprocessing_index += 1
    return img7

def get_contrast(img):
    return img.std()

def get_brightness(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.mean(cv2.split(img)[2])[0]

def increase_brightness1(img, brightness):
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            max = 255
        else:
            shadow = 0
            max = 255 + brightness
        alpha = (max - shadow) / 255
        gamma = shadow
        cal = cv2.addWeighted(img, alpha, img, 0, gamma)
    else:
        cal = img
    return cal

def increase_brightness1(img, brightness):
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            max = 255
        else:
            shadow = 0
            max = 255 + brightness
        alpha = (max - shadow) / 255
        gamma = shadow
        cal = cv2.addWeighted(img, alpha, img, 0, gamma)
    else:
        cal = img
    return cal


# %%
def increase_contrast1(img, contrast):
    alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
    gamma = 127 * (1 - alpha)
    cal = cv2.addWeighted(img, alpha, img, 0, gamma)
    return cal


# %%
def adjust_brightness_contrast1(img, optimal_brightness=(135, 225), optimal_contrast=(65, 70)):
    br0 = get_brightness(img)
    
    if br0 < optimal_brightness[0]:
        img = increase_brightness1(img, optimal_brightness[0]-br0)
    elif br0 > optimal_brightness[1]:
        img = increase_brightness1(img, optimal_brightness[1]-br0)

    cr0 = get_contrast(img)
    if cr0 < optimal_contrast[0]:
        img = increase_contrast1(img, (optimal_contrast[0]-cr0)*2)
    elif cr0 > optimal_contrast[1]:
        img = increase_contrast1(img, (optimal_contrast[1]-cr0)*2)

    return img


# %%
def get_warning_image(img, min_res = (1000, 600), range_cr = (25, 65), range_br=(130, 240)):
    min_tup = lambda t0, t1: not (False in [t1[i]>=t0[i] for i in range(len(t0))])
    range_v = lambda r, v: True if (v >= r[0]) and (v <= r[1]) else False

    warning = dict()
    
    h, w = img.shape[:2]
    crop_img = img[ int(h/10):int(h-int(h/10)),  int(w/10):int(w-(int(w/10)*3))]
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    br = round(get_brightness(hsv), 3)
    cr = round(get_contrast(gray), 3)

    if not min_tup(min_res, (w, h)):
        warning[0] = f'Resolution: {str((w, h))} not meet the minimum: {str(min_res)}'
    if not range_v(range_br, br):
        warning[1] = f'Brightness: {br} not meet the minimum: {str(range_br)}'
    if not range_v(range_cr, cr):
        warning[2] = f'Contrast: {cr} not meet the minimum: {str(range_cr)}'
        
    return warning


# %%
def adjust_gamma(img, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(img, table)


# %%
def preprocess(img):
    img = cv2.medianBlur(img, 3)
    return 255 - img


# %%
def postprocess(img):
    kernel = np.ones((3,3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img


# %%
def get_block_index(image_shape, yx, block_size): 
    y = np.arange(max(0, yx[0]-block_size), min(image_shape[0], yx[0]+block_size))
    x = np.arange(max(0, yx[1]-block_size), min(image_shape[1], yx[1]+block_size))
    return np.meshgrid(y, x)


# %%
def adaptive_median_threshold(img_in, delta):
    med = np.median(img_in)
    img_out = np.zeros_like(img_in)
    img_out[img_in - med < delta] = 255
    kernel = np.ones((3,3),np.uint8)
    img_out = 255 - cv2.dilate(255 - img_out,kernel,iterations = 2)
    return img_out


# %%
def block_image_process(img, block_size, delta):
    out_image = np.zeros_like(img)
    for row in range(0, img.shape[0], block_size):
        for col in range(0, img.shape[1], block_size):
            idx = (row, col)
            block_idx = get_block_index(img.shape, idx, block_size)
            out_image[block_idx] = adaptive_median_threshold(img[block_idx], delta)
    return out_image


# %%
def process_image(img, block_size, delta):
    image_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_in = preprocess(image_in)
    image_out = block_image_process(image_in, block_size, delta)
    image_out = postprocess(image_out)
    return image_out


# %%
def sigmoid(x, orig, rad):
    k = np.exp((x - orig) * 5 / rad)
    return k / (k + 1.)


# %%
def combine_block(img_in, mask):
    img_out = np.zeros_like(img_in)
    img_out[mask == 255] = 255
    fimg_in = img_in.astype(np.float32)

    idx = np.where(mask == 0)
    if idx[0].shape[0] == 0:
        img_out[idx] = img_in[idx]
        return img_out

    lo = fimg_in[idx].min()
    hi = fimg_in[idx].max()
    v = fimg_in[idx] - lo
    r = hi - lo

    img_in_idx = img_in[idx]
    ret3,th3 = cv2.threshold(img_in[idx],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    bound_value = np.min(img_in_idx[th3[:, 0] == 255])
    bound_value = (bound_value - lo) / (r + 1e-5)
    f = (v / (r + 1e-5))
    f = sigmoid(f, bound_value + 0.05, 0.2)

    
    img_out[idx] = (255. * f).astype(np.uint8)
    return img_out


# %%
def combine_block_image_process(img, mask, block_size):
    out_image = np.zeros_like(img)
    for row in range(0, img.shape[0], block_size):
        for col in range(0, img.shape[1], block_size):
            idx = (row, col)
            block_idx = get_block_index(img.shape, idx, block_size)
            out_image[block_idx] = combine_block(
                img[block_idx], mask[block_idx])
    return out_image


# %%
def combine_postprocess(img):
    return img


# %%
def combine_process(img, mask):
    image_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_out = combine_block_image_process(image_in, mask, 20)
    image_out = combine_postprocess(image_out)
    return image_out


# %%
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    lim = 255 - value
    v[v > lim] = 255
    
    m = np.min(v[v <= lim])
    if m+value < 0:
        value += abs(m+value)
    increase = value

    increase = np.uint8(increase)
    v[v <= lim] += increase

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


# %%
def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect


# %%
def four_point_transform(img, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	max_width = max(int(width_a), int(width_b))

	height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	max_height = max(int(height_a), int(height_b))
	dst = np.array([
		[0, 0],
		[max_width - 1, 0],
		[max_width - 1, max_height - 1],
		[0, max_height - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(img, M, (max_width, max_height))

	return warped


# %%
def add_border_from_bottom(img):
    row, col = img.shape[:2]
    bottom = img[row-2:row, 0:col]
    mean = avg(cv2.mean(bottom))

    bordersize = 200
    border = cv2.copyMakeBorder(
        img,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[mean, mean, mean]
    )

    return border


# %%
def scan(img, points=4, add_border=True, minimal=(300, 30)):
    original_image = copy.copy(img)

    if add_border:
        img = add_border_from_bottom(img)

    ratio = img.shape[0] / 500.0
    orig = img.copy()
    img = imutils.resize(img, height = 500)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    founded = []

    for c in cnts:    
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == points:
            founded.append(approx)

    if len(founded) > 0:
        warpeds = []
        for screenCnt in founded:
            warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
            h, w = warped.shape[:2]
            if (h < minimal[0]) or (w < minimal[1]):
                continue
            warpeds.append(warped)
        return original_image, warpeds
    else:
        return original_image, []


# %%
def ktp_censor(img):
    l_img = img.copy()    
    h, w = l_img.shape[:2]

    wh_h, wh_w = int(4*h/5), int(w/4)
    s_img = np.array([[[255, 255, 255, 255] for _ in range(wh_w)] for _ in range(wh_h)])

    y, x = int(h/5), w-(int(w/4))
    y1, y2 = y, y + s_img.shape[0]
    x1, x2 = x, x + s_img.shape[1]

    alpha_s = s_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                alpha_l * l_img[y1:y2, x1:x2, c])
    
    return l_img


# %%
def adjust_brightness_contrast(img):
    # img1 = adjust_gamma(img)
    img1 = copy.deepcopy(img)

    h, w = img1.shape[:2]
    crop_img = img1[ int(h/10):int(h-int(h/10)),  int(w/10):int(w-(int(w/10)*3))]
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    br0 = get_brightness(hsv) 

    srdb = 70
    stdb = 1
    dist_b = br0-srdb
    delta = (stdb*dist_b)/2
    
    mask = process_image(img1, 40, delta)

    img1 = combine_process(img1, mask)

    return  img1


# %%
def blur_score(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# %%

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


# %%

def blurring_sharping(img):
    sharpness = blur_score(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    brightness = get_brightness(img)
    if (brightness > 175) and (sharpness > 800):
        img1 = cv2.GaussianBlur(img,(1,5),0)
    elif (brightness > 175) and (sharpness > 400):
        img1 = unsharp_mask(img, (5, 5), 1, 2)
    elif (brightness <= 175) and (sharpness > 800):
        img1 = cv2.GaussianBlur(img,(1,5),0)
    elif (brightness <= 175) and (sharpness <= 400):
        img1 = unsharp_mask(img, (5, 5), 1, 2)
    elif (brightness > 175) and (sharpness <= 400):
        img1 = unsharp_mask(img, (5, 5), 1, 2)
    elif (brightness <= 100) and (sharpness >= 150):
        img1 = cv2.GaussianBlur(img,(1,5),0)
    else:
        img1 = img
    return img1


# %%
def preprocessing3(img0, manual_crop=False, seq=None, write_img=False):
    preprocessing_sequences = [
        # [scan],
        # [ktp_censor],
        [cv2.resize, (1000, 600)],
        [increase_brightness, 0-100],
        [adjust_gamma],
        [blurring_sharping],
        [adjust_brightness_contrast]
    ]

    if seq is None:
        seq = [i for i in range(len(preprocessing_sequences))]

    img1 = img0
    for prep_i in seq:
        prep = preprocessing_sequences[prep_i]
        fun = prep[0]
        del prep[0]

        img1 = fun(img1, *prep)

    if manual_crop:
        start_row = 0
        start_col = 0
        end_row = 600
        end_col = 750
        img1 = img1[start_row:end_row, start_col:end_col] 

    if write_img:
        global preprocessing_index
        cv2.imwrite(os.path.join(OUT_DIR, f'preprocessing3_{preprocessing_index}.jpeg'), img1)
        preprocessing_index += 1
    return img1
