#上下采样-----------------------------------------------------------------------
def down_sample(img, scale=5):

    shape = img.shape
    ori_image = Image.fromarray(img)
    image_thumb = ori_image.resize((shape[1] // scale, shape[0] // scale), Image.NEAREST)
    image_thumb = np.array(image_thumb).astype(np.uint8)
    return image_thumb


def up_sample(image, ori_shape):

    mask_thumb = Image.fromarray(image)
    marker = mask_thumb.resize((ori_shape[1], ori_shape[0]), Image.NEAREST)
    marker = np.array(marker).astype(np.uint8)
    return marker


#填洞-------------------------------------------------------------------------
def hole_fill(binary_image):
    ''' 孔洞填充 '''
    hole = binary_image.copy()  ## 空洞填充
    hole = cv2.copyMakeBorder(hole, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0])  # 首先将图像边缘进行扩充，防止空洞填充不完全
    hole2 = hole.copy()
    cv2.floodFill(hole, None, (0, 0), 255)  # 找到洞孔
    hole = cv2.bitwise_not(hole)
    binary_hole = cv2.bitwise_or(hole2, hole)[1:-1, 1:-1]
    return binary_hole


#打印进度条  tqdm---------------------------------------------------------------
def view_bar(message, id, total, end=''):
    rate = id / total
    rate_num = int(rate * 40)
    print('\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num,
                                    "=" * (40 - rate_num), np.round(rate * 100), id, total,), end=end)


#切图  拼图-------------------------------------------------------------------
def split(image, cut_size, overlap=100):

    shapes = image.shape
    x_nums = int(shapes[0] / (cut_size - overlap))
    y_nums = int(shapes[1] / (cut_size - overlap))
    img_list = []
    x_list = []
    y_list = []
    for x_temp in range(x_nums + 1):
        for y_temp in range(y_nums + 1):
            x_begin = max(0, x_temp * (cut_size - overlap))
            y_begin = max(0, y_temp * (cut_size - overlap))
            x_end = min(x_begin + cut_size, shapes[0])
            y_end = min(y_begin + cut_size, shapes[1])
            if x_begin == x_end or y_begin == y_end:
                continue
            i = image[x_begin: x_end, y_begin: y_end]
            # tifffile.imsave(os.path.join(outpath, file + '_' + str(shapes[0]) + '_' + str(shapes[1]) + '_' + str(x_begin) + '_' + str(y_begin) + '.tif'), i)  #, r'white_5000'r'20210326_other_crop'
            x_list.append(x_begin)
            y_list.append(y_begin)
            img_list.append(i)
    return img_list, x_list, y_list


def merge(label_list, x_list, y_list, shapes,  overlap = 100):

    if len(label_list) == 1:
        return label_list[0]

    if not isinstance(label_list, list):
        return label_list

    image = np.zeros((int(shapes[0]), int(shapes[1])), dtype=np.uint8)
    for index, temp_img in enumerate(label_list):
        info = [x_list[index], y_list[index]]
        h, w = temp_img.shape

        if overlap == 0:
            x_begin = int(info[0]) + overlap // 2
            y_begin = int(info[1]) + overlap // 2
            image[int(x_begin): int(x_begin) + h, int(y_begin): int(y_begin) + w] = temp_img
        else:
            # cor in merged image
            x_begin = int(info[0]) + overlap // 2 if int(info[0]) else 0
            y_begin = int(info[1]) + overlap // 2 if int(info[1]) else 0
            if x_begin == 0 or x_begin + h > shapes[0]:
                x_end = x_begin + h - overlap // 2
            else:
                x_end = x_begin + h - overlap

            if y_begin == 0 or y_begin + w > shapes[1]:
                y_end = y_begin + w - overlap // 2
            else:
                y_end = y_begin + w - overlap

            # cor in tile
            x_begin_tile = overlap // 2 if int(info[0]) else 0
            y_begin_tile = overlap // 2 if int(info[1]) else 0

            if x_begin_tile == 0 or x_begin + h > shapes[0]:
                x_end_tile = x_begin_tile + h - overlap // 2
            else:
                x_end_tile = x_begin_tile + h - overlap

            if y_begin_tile == 0 or y_begin + w > shapes[1]:
                y_end_tile = y_begin_tile + w - overlap // 2
            else:
                y_end_tile = y_begin_tile + w - overlap

            image[x_begin: x_end, y_begin: y_end] = temp_img[x_begin_tile: x_end_tile, y_begin_tile: y_end_tile]

    return image


#生成ouline------------------------------------------------------------------
def outline(image):
    image = np.where(image != 0, 1, 0).astype(np.uint8)
    edge = np.zeros((image.shape), dtype=np.uint8)
    contours, hierachy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    r = cv2.drawContours(edge, contours, -1, (255, 255, 255), 1)
    return r


#16bit转8bit------------------------------------------------------------------
def transfer_16bit_to_8bit(image_16bit):

    min_16bit = np.min(image_16bit)
    max_16bit = np.max(image_16bit)

    image_8bit = np.array(np.rint(255 * ((image_16bit - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)

    return image_8bit

#dice-------------------------------------------------------------------------
def dice(y_true, y_pred):
    # 这里的acc就是dice系数，因为保存最好模型的地方只能识别val_acc
    y_pred = (y_pred > 0).astype(np.uint8)
    y_true = (y_true > 0).astype(np.uint8)
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

#mian----------------------------------------------------------------------

#!/usr/bin/env python
#coding: utf-8
###### Import Modules ##########
import argparse
import sys
import os
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'waterSeg'))
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'postProcess'))
# print(sys.path)
import time
import numpy as np
import tissueCut_utils.tissue_seg_pipeline as pipeline


#########################################################
#########################################################
# tissue segmentation
#########################################################
#########################################################


usage = '''
     limin  %s
     Usage: %s imagePath outPath imageType(1:ssdna; 0:RNA)  method(1:deep; 0:other)
''' % ('2021-07-15', os.path.basename(sys.argv[0]))


def args_parse():
    ap = argparse.ArgumentParser(usage=usage)
    ap.add_argument('-i', '--img_path', action='store', help='image path')
    ap.add_argument('-o', '--out_path', action='store',  help='mask path')
    ap.add_argument('-t', '--img_type', dest='img_type', type=int, help='ssdna:1; rna:0', default=1)
    ap.add_argument('-m', '--seg_method', dest='seg_method', type=int, help='deep:1; intensity:0', default=1)
    return ap.parse_args()


def tissueSeg(img_path, out_path, type, deep):
    cell_seg_pipeline = pipeline.tissueCut(img_path, out_path, type, deep)
    ref = cell_seg_pipeline.tissue_seg()
    return ref


def tissue_segment_entry(args):
    args = vars(args)
    img_path = args['img_path']
    out_path = args['out_path']
    type = args['img_type']
    deep = args['seg_method']

    t0 = time.time()
    ref = tissueSeg(img_path, out_path, type, deep)
    t1 = time.time()
    print('running time:', t1 - t0)



def main():
    ######################### Phrase parameters #########################
    args = args_parse()
    print(args)

    # call segmentation
    tissue_segment_entry(args)

if __name__ == '__main__':
    main()

#图像增强------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def augument(image, mask):
    segmap = SegmentationMapsOnImage(mask, shape=image.shape)
    seq = iaa.Sequential([
        iaa.Crop(px=(0, 100)),     #对图像进行crop操作，随机在距离边缘的0到16像素中选择crop范围
    iaa.Fliplr(0.5),     #对百分之五十的图像进行做左右翻转
    iaa.Flipud(0.5),
    iaa.GaussianBlur((1,5)),     #在模型上使用0均值1方差进行高斯模糊
#          iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
        iaa.ContrastNormalization((1, 1.5)),
    iaa.Sharpen((0.0, 1.0)),       # sharpen the image
    iaa.Affine(rotate=(-45, 45))  # rotate by -45 to 45 degrees (affects segmaps)
    ], random_order = True)
    image_aug, segmap_aug = seq(image = image, segmentation_maps=segmap)#  
    return image_aug, segmap_aug

#mask转彩色-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def showConnectedComponents(binary_img):

    w, h = binary_img.shape
    color = []
    color.append((0, 0, 0))
    img_color = np.zeros((w, h, 3), dtype=np.uint8)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img)
    for num in range(1,retval):
        color_b = random.randint(0, 255)
        color_g = random.randint(0, 255)
        color_r = random.randint(0, 255)
        color.append((color_b, color_g, color_r))
    for x in range(w):
        for y in range(h):
            lable = labels[x,y]
            img_color[x,y,:] = color[int(lable)]
            
    cv2.imwrite(r'/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.user/liMin/cellSegMask_color.png', img_color)

#多值mask转二值，类似deecell的plant-----------------------------------------------------------------------------------

def to_binary(img):
    binary = np.zeros((img.shape), dtype=np.uint16)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for id in cell_id:
        if id == 0: continue
        temp = np.where(img==id, 255, 0).astype(np.uint8)
        temp_erode = cv2.erode(temp, kernel)
        binary += temp_erode
        
    binary = np.where(binary != 0, 255, 0)
    return binary

#画图----------------------------------------------------------------------------------------------------------------
data = {
    'fruits':
    ['苹果', '梨', '草莓', '西瓜', '葡萄', '香蕉'],
    '2015': [2, 1, 4, 3, 2, 4],
    '2016': [5, 3, 3, 2, 4, 6],
    '2017': [3, 2, 4, 4, 5, 3]
}
df = pd.DataFrame(data).set_index("fruits")
print(df)

p_bar = df.plot_bokeh.bar(
    ylabel="每斤的的价格 [￥]", 
    title="水果每年的价格", 
    stacked=True,
    alpha=0.6)


#######################################################################

data = {
    'fruits':
    ['苹果', '梨', '草莓', '西瓜', '葡萄', '香蕉'],
    '2015': [2, 1, 4, 3, 2, 4],
    '2016': [5, 3, 3, 2, 4, 6],
    '2017': [3, 2, 4, 4, 5, 3]
}
df = pd.DataFrame(data).set_index("fruits")

p_bar = df.plot_bokeh.bar(
    ylabel="每斤的的价格 [￥]", 
    title="水果每年的价格", 
    stacked=False,
    alpha=0.6, 
    show_figure=False)

p_barS = df.plot_bokeh.bar(
    ylabel="每斤的的价格 [￥]", 
    title="水果每年的价格", 
    stacked=True,
    alpha=0.6,
    show_figure=False)

p_barh = df.plot_bokeh.barh(
    ylabel="每斤的的价格 [￥]", 
    title="水果每年的价格", 
    stacked=False,
    alpha=0.6,
    show_figure=False)

p_barSh = df.plot_bokeh.barh(
    ylabel="每斤的的价格 [￥]", 
    title="水果每年的价格", 
    stacked=True,
    alpha=0.6,
    show_figure=False)


pandas_bokeh.plot_grid([[p_bar, p_barS], [p_barh, p_barSh]], plot_width=450)


#路径下查找文件-----------------------------------------------------------------------------------

def search_files(file_path, exts):
    files_ = list()
    for root, dirs, files in os.walk(file_path):
        if len(files) == 0:
            continue
        for f in files:
            fn, ext = os.path.splitext(f)
            if ext in exts:
                files_.append(os.path.join(root, f))
    return files_

#json写入和读取--------------------------------------------------------------------------

from json import loads, dumps, dump, load
class JSONObject:
    def __init__(self, d):
        self.__dict__ = d


def json_serialize(obj, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as fd:
        str_dct = dumps(obj, default=lambda o: o.__dict__)
        dump(loads(str_dct), fd, indent=2, ensure_ascii=False)


def json_2_dict(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as fd:
        return load(fd)


# about dict
def dict_deserialize(dct: dict) -> JSONObject:
    str_dct = dumps(dct)
    return loads(str_dct, object_hook=JSONObject)


# about json
def json_deserialize(file_path: str) -> JSONObject:
    dct = json_2_dict(file_path)
    return dict_deserialize(dct)

#局部分水岭-------------------------------------------------------------------------
def water_score(input_list):

    mask, image = input_list
    label = measure.label(mask, connectivity=2)
    props = measure.regionprops(label, intensity_image=image)
    shapes = mask.shape
    post_mask = np.zeros(shapes, dtype=np.uint8)
    color_mask_ori = np.zeros(shapes)
    score_list = []
    for idx, obj in enumerate(props):
        intensity_image = obj['intensity_image'] # white image
        bbox = obj['bbox']
        center = obj['centroid']
        count, xpts, ypts = find_maxima(intensity_image)
        if count > 1:
            distance = ndimage.distance_transform_edt(intensity_image)
            markers = np.zeros(intensity_image.shape, dtype=np.uint8)
            for i in range(count):
                markers[ypts[i], xpts[i]] = i + 1
            seg_result = segmentation.watershed(-distance, markers, mask=intensity_image, compactness=10,
                                                watershed_line=True)
            seg_result = np.where(seg_result != 0, 255, 0).astype(np.uint8)  #binary image
            post_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]] += seg_result
            label_seg = measure.label(seg_result, connectivity=1)
            props_seg = measure.regionprops(label_seg, intensity_image=intensity_image)
            color_seg = np.zeros(obj['image'].shape)
            for p in props_seg:
                score_temp = score_cell(p)
                bbox_p = p['bbox']
                center = p['centroid']
                color_mask_temp = p['image'] * score_temp * 100
                color_seg[bbox_p[0]: bbox_p[2], bbox_p[1]: bbox_p[3]] += color_mask_temp
                score_list.append([p['label'], center[0], center[1], score_temp])
            color_mask_ori[bbox[0]: bbox[2], bbox[1]: bbox[3]] += color_seg
        else:
            post_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]] += (obj['image'] * 255).astype(np.uint8)
            # area_list.append(mean_intensity)
            total_score = score_cell(obj)
            score_list.append([obj['label'], center[0], center[1], total_score])
            color_mask_temp = obj['image'] * total_score * 100
            color_mask_ori[bbox[0]: bbox[2], bbox[1]: bbox[3]] += color_mask_temp
    post_mask = np.where(post_mask != 0, 1, 0).astype(np.uint8)
    color_mask_ori = np.array(np.rint(color_mask_ori), dtype=np.uint8)
    return [post_mask, color_mask_ori]

def watershed_multi(input_list, processes):
    with mp.Pool(processes=processes) as p:
        post_img = p.map(water_score, input_list)
    return post_img
