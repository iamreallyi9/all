import numpy as np
import cv2
import time
import os
import os.path as osp

def avg_simplify_points(pts, avg_num):
    all_points = []
    for i in range(pts.shape[0] - 1):
        start_pt = pts[i]
        end_pt = pts[i + 1]
        points_on_line = np.linspace(start_pt, end_pt, int(np.linalg.norm(start_pt - end_pt)))

        all_points.append(start_pt)
        for j in range(1, len(points_on_line) - 1):
            all_points.append(points_on_line[j])
        all_points.append(end_pt)

    per_points = int(len(all_points) / avg_num)

    A = np.mat('1, 1;, {}, {}'.format(per_points, per_points + 1))
    b = np.mat('{}, {}'.format(avg_num, len(all_points))).T
    r = (np.linalg.solve(A, b) + 0.5).astype(np.int32)
    r = r.astype(np.int32)
    # print(r)

    res_points = []
    for i in range(r[0, 0]):
        pt = all_points[i * per_points]
        res_points.append(pt)
        # cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), cv2.FILLED)

    for i in range(r[1, 0]):
        pt = all_points[r[0, 0] * per_points + i * (per_points + 1)]
        res_points.append(pt)
        # cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), cv2.FILLED)

    pt = all_points[-1]
    res_points.append(pt)
    # cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), cv2.FILLED)
    res_points = np.array(res_points, dtype=np.float32)
    return res_points
# def land1k_to_240(landmarks_1k):
#     index_1k_to_240 = [211, 216, 222, 228, 233, 239, 245, 250, 256, 262, 267, 273, 279, 284, 290, 296, 301, 306, 0,  #左脸
#                        6, 11, 16, 22, 28, 33, 39, 45, 50, 56, 62, 67, 73, 79, 84, 90, 96, 101,  #右脸
#
#                        928, 931, 935, 938, 942, 945, 949, 952, 956, 960,  #左上眉毛
#                        964, 968, 972, 976, 979, 983, 986, 990, 993, 997,  #左下眉毛
#                        856, 860, 864, 868, 871, 874, 877, 881, 884, 888, 892,  #右上眉毛
#                        896, 899, 902, 906, 909, 913, 916, 920, 924,  #右下眉毛
#                        # 723, 720, 717, 714, 711, 708, 705, 702, 699, 696, 693, 691,  # 左上眼睛
#                        723, 720, 717, 715, 712, 709, 707, 704, 701, 699, 696, 693, 691,  # 左上眼睛
#                        753, 749, 747, 744, 741, 739, 736, 733, 731, 728, 725,  #左下眼睛
#                        792, 794, 797, 800, 802, 805, 808, 810, 813, 816, 818, 821,
#                        824, 826, 829, 832, 834, 837, 840, 842, 845, 848, 850, 853,
#
#                        548, 560, 575, 579,  #左鼻翼
#                        580, 581,  # 左鼻孔
#                        583, 584,  # 右鼻孔
#                        585, 589, 604, 616,  #右鼻翼
#                        653, 645, 637, 629, 621,  # 鼻梁
#                        312, 463, 458, 453, 448, 443,  # 外圈上嘴唇左边
#                        # 467, 463, 458, 453, 448, 443, 438,  #外圈上嘴唇左边
#                        438, 436, 434, 432,  #外圈上嘴唇左上
#
#                        430, 428, 426,  # 外圈上嘴唇右上
#
#                        421, 416, 411, 406, 401,   # 外圈上嘴唇右边
#
#                        396, 391, 386, 382, 377, 372, 368, 363, 358,  #外圈下嘴唇右边
#                        354, 349, 344, 340, 335, 330, 326, 321, 316,  # 外圈下嘴唇左边
#
#
#                        468, 546, 544, 542, 540, 538, 536, 534, 532, 530,528,
#                        526, 524, 522, 520, 518, 516, 514, 512, 510,
#                        508, 506, 504, 502, 500, 498, 496, 494, 492, 490, 488,
#                        486, 484, 482, 480, 478, 476, 474, 472, 470,
#
#                        # 112, 123, 134, 145, 156, 167, 178, 189, 200,  # 额头
#                        108, 113, 118, 123, 128, 133, 138, 143, 148, 153, 158, 163, 168, 173, 178, 183, 188, 193, 198, 203,# 额头
#                        654,  # 左眼中心
#                        755,  # 右眼中心
#                        ]
#     # print(len(index_1k_to_240))
#     landmarks_240 = landmarks_1k[index_1k_to_240, :]
#     return landmarks_240

def land1k_to_220(landmarks_1k):
    index_1k_to_220 = [211, 216, 222, 228, 233, 239, 245, 250, 256, 262, 267, 273, 279, 284, 290, 296, 301, 306, 0,  #左脸
                       6, 11, 16, 22, 28, 33, 39, 45, 50, 56, 62, 67, 73, 79, 84, 90, 96, 101,  #右脸

                       928, 931, 935, 938, 942, 945, 949, 952, 956, 960,  #左上眉毛
                       964, 968, 972, 976, 979, 983, 986, 990, 993, 997,  #左下眉毛
                       856, 860, 864, 868, 871, 874, 877, 881, 884, 888, 892,  #右上眉毛
                       896, 899, 902, 906, 909, 913, 916, 920, 924,  #右下眉毛
                       # 723, 720, 717, 714, 711, 708, 705, 702, 699, 696, 693, 691,  # 左上眼睛
                       723, 720, 717, 715, 712, 709, 707, 704, 701, 699, 696, 693, 691,  # 左上眼睛
                       753, 749, 747, 744, 741, 739, 736, 733, 731, 728, 725,  #左下眼睛
                       792, 794, 797, 800, 802, 805, 808, 810, 813, 816, 818, 821,
                       824, 826, 829, 832, 834, 837, 840, 842, 845, 848, 850, 853,

                       548, 560, 575, 579,  #左鼻翼
                       580, 581,  # 左鼻孔
                       583, 584,  # 右鼻孔
                       585, 589, 604, 616,  #右鼻翼
                       653, 645, 637, 629, 621,  # 鼻梁
                       312, 463, 458, 453, 448, 443,  # 外圈上嘴唇左边
                       # 467, 463, 458, 453, 448, 443, 438,  #外圈上嘴唇左边
                       438, 436, 434, 432,  #外圈上嘴唇左上

                       430, 428, 426,  # 外圈上嘴唇右上

                       421, 416, 411, 406, 401,   # 外圈上嘴唇右边

                       396, 391, 386, 382, 377, 372, 368, 363, 358,  #外圈下嘴唇右边
                       354, 349, 344, 340, 335, 330, 326, 321, 316,  # 外圈下嘴唇左边

                       468, 546, 544, 542, 540, 538, 536, 534, 532, 530,528,
                       526, 524, 522, 520, 518, 516, 514, 512, 510,
                       508, 506, 504, 502, 500, 498, 496, 494, 492, 490, 488,
                       486, 484, 482, 480, 478, 476, 474, 472, 470,
                       654,  # 左眼中心
                       755,  # 右眼中心
                       ]
    landmarks_220 = landmarks_1k[index_1k_to_220, :]
    return landmarks_220

def land240_to_96(land240pts):
    index_240_to_96 = [52, 55, 58, 61, 64, 66, 78, 70, 72, 0,              #左边脸 10
                       2,4,6,8,10,12,15,18,21,              #右边脸 9
                       74,76,78,80,82,156,     #左上眉毛 6
                       154,152,150,148,  #左下眉毛 4
                       157,85,87,89,91,93,   #右上眉毛 6
                       165,163,161,159,  #右下眉毛 4
                       107,108,110,112,114,116,117,   #左上眼睛 7
                       118,120,122,124,126,  #左下眼睛 5
                       127,128,130,132,134,136,137, #右上眼睛 7
                       138,140,142,144,146,  #右下眼睛 5
                       167,169,171,177, # 左鼻侧  4
                       98,99,101,102,  #鼻孔  5
                       178, 176,174,172,   #右鼻侧  4
                       97,  #鼻尖
                       179,182,185,186,187,190,193,  #外上嘴唇 7
                       195,198,201,204,206,   #外下嘴唇  5
                       208,212,216,220,223,   #内上嘴唇  5
                       227,230,234,   #内下嘴唇  3
                      ]
    landmarks_96 = land240pts[index_240_to_96, :]
    return landmarks_96

def land240_to_137(land240pts):
    index_240_to_137 = [0,3,6,9,12,15, 18,21,25,29,33,37, # 右脸   12
                        41,45,49,53,56,59,62,65,68,71,  #左脸  10
                        179,207,206,205,204,203,201,200,199,198,197,196,195,194,193,   #外下嘴唇  15
                        192,191,190,189,187,186,185,183,182,181,180,    #外上嘴唇   11
                        208,236,235,233,231,229,227,225,223,     #内下嘴唇  9
                        222,220,218,216,213,211,209,      #内上嘴唇  7
                        168,169,170,171,177,    #左鼻翼   5
                        98,99,100,101,102,   #鼻孔 5
                        178,176,175,174,173,    #右鼻翼5
                        103,104,105,106,   #上鼻孔 4
                        97,96,95,94,    #鼻梁 4
                        238, #左眼珠 1
                        117,116,115,113,112,111,109,108,107, #左上眼睛 9
                        126,125,123,122,121,119,118,   #左下眼睛 7
                        239 ,  #右眼珠
                        127,128,129,131,132,133,135,136,137, #右上眼睛 9
                        138,139,141,142,143,145,146, #右下眼睛7
                        157,85,88,91,93,#右上眉毛 5
                        165,162,159, #右下眉毛3
                        74,76,79,82,156, #左上眉毛 5
                        154,151,148, #左下眉毛 3
                      ]
    landmarks_137 = land240pts[index_240_to_137, :]
    return landmarks_137

def pts_1k_to_240(pts_1k):
    pts_240 = []

    # re_order = list(range(156, 312))    #左边脸
    # re_order.extend(list(range(0, 154)))   #右边脸
    # pts = pts_1k[re_order]
    re_order = list(range(0, 154))    #右边脸
    pts_r = pts_1k[re_order]
    pts_facecontour_r = avg_simplify_points(pts_r, 36)
    pts_240.append(pts_facecontour_r)

    re_order = list(range(156, 308))    #右边脸
    pts_l = pts_1k[re_order]
    pts_facecontour_l = avg_simplify_points(pts_l, 36)
    pts_240.append(pts_facecontour_l)

    re_order = list(range(928, 965))   #左上眉毛
    pts = pts_1k[re_order]
    # for pt in pts:
    #     cv2.circle(image, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), cv2.FILLED)
    pts_left_brown_up = avg_simplify_points(pts, 10)
    pts_left_brown_up = pts_left_brown_up[:-1]
    # for pt in pts_left_brown_up:
    #     cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), cv2.FILLED)

    pts_240.append(pts_left_brown_up)

    re_order = list(range(856, 893))  #右上眉毛
    pts = pts_1k[re_order]
    # for pt in pts:
    #     cv2.circle(image, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), cv2.FILLED)
    pts_right_brown_up = avg_simplify_points(pts, 10)
    pts_right_brown_up = pts_right_brown_up[1:]
    # for pt in pts_right_brown_up:
    #     cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), cv2.FILLED)

    pts_240.append(pts_right_brown_up)

    re_order = list(range(653, 620, -1))    #鼻梁
    pts = pts_1k[re_order]
    # for pt in pts:
    #     cv2.circle(image, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), cv2.FILLED)
    pts_nose_middle = avg_simplify_points(pts, 3)
    # for pt in pts_nose_middle:
    #     cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), cv2.FILLED)

    pts_240.append(pts_nose_middle)

    re_order = list(range(580, 585))   #鼻孔
    re_order.extend([617,618,619,620])
    pts_nose_bottom = pts_1k[re_order]
    # for pt in pts_nose_bottom:
    #     cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), cv2.FILLED)

    pts_240.append(pts_nose_bottom)

    re_order = list(range(723, 690, -1))   #左上眼睛
    pts = pts_1k[re_order]
    # for pt in pts:
    #     cv2.circle(image, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), cv2.FILLED)
    pts_left_eye_up = avg_simplify_points(pts, 10)

    # for pt in pts_left_eye_up:
    #     cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), cv2.FILLED)

    # pts_106.append(pts_left_eye_up[[0, 1, 3, 4]])
    pts_240.append(pts_left_eye_up)

    re_order = [691]
    re_order.extend(list(range(754, 722, -1)))  #左下眼睛
    pts = pts_1k[re_order]
    # for pt in pts:
    #     cv2.circle(image, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), cv2.FILLED)
    pts_left_eye_bottom = avg_simplify_points(pts, 10)
    pts_left_eye_bottom = pts_left_eye_bottom[1:-1]
    # for pt in pts_left_eye_bottom:
    #     cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), cv2.FILLED)

    # pts_106.append(pts_left_eye_bottom[[0, 2]])
    pts_240.append(pts_left_eye_bottom)

    re_order = list(range(792, 825))    #右上眼睛
    pts = pts_1k[re_order]
    # for pt in pts:
    #     cv2.circle(image, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), cv2.FILLED)
    pts_right_eye_up = avg_simplify_points(pts, 10)
    # for pt in pts_right_eye_up:
    #     cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), cv2.FILLED)

    # pts_106.append(pts_right_eye_up[[0, 1, 3, 4]])
    pts_240.append(pts_right_eye_up)

    re_order = list(range(824, 855))    #右下眼睛
    re_order.extend([792])
    pts = pts_1k[re_order]
    # for pt in pts:
    #     cv2.circle(image, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), cv2.FILLED)
    pts_right_eye_bottom = avg_simplify_points(pts, 10)
    pts_right_eye_bottom = pts_right_eye_bottom[1:-1]
    # for pt in pts_right_eye_bottom:
    #     cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), cv2.FILLED)

    # pts_106.append(pts_right_eye_bottom[[0, 2]])
    pts_240.append(pts_right_eye_bottom)

    re_order = [928]   #左眉脚左
    re_order.extend(list(range(999, 963, -1)))  #左下眉毛
    pts = pts_1k[re_order]
    # for pt in pts:
    #     cv2.circle(image, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), cv2.FILLED)
    pts_left_brown_bottom = avg_simplify_points(pts, 10)
    pts_left_brown_bottom = pts_left_brown_bottom[1:]
    # for pt in pts_left_brown_bottom:
    #     cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), cv2.FILLED)

    pts_240.append(pts_left_brown_bottom)

    re_order = [856]  #右眉脚左
    re_order.extend(list(range(927, 891, -1))) #右下眉毛
    pts = pts_1k[re_order]
    # for pt in pts:
    #     cv2.circle(image, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), cv2.FILLED)
    pts_right_brown_bottom = avg_simplify_points(pts, 10)
    pts_right_brown_bottom = pts_right_brown_bottom[:-1]
    # for pt in pts_right_brown_bottom:
    #     cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), cv2.FILLED)

    pts_240.append(pts_right_brown_bottom)

    # 72 73 74 75 76 77 78 79 81 81 82 83
    # pts_240.append(pts_left_eye_up[[2]])
    # pts_240.append(pts_left_eye_bottom[[1]])
    # pts_240.append(pts_1k[[654]])
    # pts_240.append(pts_right_eye_up[[2]])
    # pts_240.append(pts_right_eye_bottom[[1]])
    # pts_240.append(pts_1k[[755]])
    pts_240.append(pts_1k[[548]])
    pts_240.append(pts_1k[[554]])
    pts_240.append(pts_1k[[560]])
    pts_240.append(pts_1k[[566]])
    pts_240.append(pts_1k[[571]])
    # pts_240.append(pts_1k[[571]])
    #
    pts_240.append(pts_1k[[616]])
    pts_240.append(pts_1k[[610]])
    pts_240.append(pts_1k[[604]])
    pts_240.append(pts_1k[[598]])
    pts_240.append(pts_1k[[593]])
    #
    pts_240.append(pts_1k[[579]])
    pts_240.append(pts_1k[[585]])

    re_order = [312]
    re_order.extend(list(range(467, 437, -1)))  #左上半嘴唇
    pts = pts_1k[re_order]
    m1 = avg_simplify_points(pts, 6)
    pts_240.append(m1)

    #
    re_order = [432]
    m1 = pts_1k[re_order]
    pts_240.append(m1)

    re_order = list(range(426, 395, -1))
    pts = pts_1k[re_order]
    m1 = avg_simplify_points(pts, 6)
    pts_240.append(m1)
    #
    re_order = list(range(396, 312, -1))
    pts = pts_1k[re_order]
    m1 = avg_simplify_points(pts, 15)
    m1 = m1[1:-1]
    pts_240.append(m1)
    #
    re_order = [468]
    re_order.extend(list(range(546, 507, -1)))
    pts = pts_1k[re_order]
    m1 = avg_simplify_points(pts, 15)
    pts_240.append(m1)
    #
    re_order = list(range(507, 467, -1))
    pts = pts_1k[re_order]
    m1 = avg_simplify_points(pts, 15)
    m1 = m1[1:-1]
    pts_240.append(m1)
    #
    #
    pts_240.append(pts_1k[[654]])
    pts_240.append(pts_1k[[755]])

    pts_240 = np.concatenate(pts_240, axis=0)

    return pts_240



def draw_points(pts,img):

    h, w, _ = img.shape
    ori_img_show_p4 = cv2.resize(img, (w * 4, h * 4))
    idx = 0
    for point in pts:

        cv2.circle(ori_img_show_p4, (int(point[0] * 4), int(point[1] * 4)), 1, (0, 0, 255), 1)
        cv2.putText(ori_img_show_p4, str(idx), (int(point[0] * 4), int(point[1] * 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)
        idx = idx + 1

    return ori_img_show_p4

if __name__ == '__main__':

    img_dir_path = r"/home/colomi/Desktop/new2T/faceData/latest/euler"
    # save_draw_land240_path = r"/home/colomi/Desktop/Data/faceLandmark/temp"
    #
    # if not os.path.exists(save_draw_land240_path):
    #     os.makedirs(save_draw_land240_path)
    idx = 0
    for dirpath,dirnames,filenames in os.walk(img_dir_path):
        for file_name in filenames:
            if not file_name.endswith('.jpg'):continue
            t1 = time.time()
            img_path = osp.join(dirpath,file_name)
            img_src = cv2.imread(img_path)
            pt_path = img_path.replace('jpg','pt240')
            if not osp.exists(pt_path):
                print('240 pts not exists')
            land240 = np.loadtxt(pt_path)
            # landmark240_new = pts_1k_to_240(land1kpts)
            landmark137 = land240_to_137(land240)

            # print(idx , info)
            # cv2.imshow('fff',img_src)
            # cv2.waitKey()
            # landmark96 = land240_to_96(landmark240_new)
            # img_cp= img_src.copy()
            # img_137 = draw_points(landmark137, img_src)
            # img_96 = draw_points(landmark96, img_cp)

            # landmark_240 = np.savetxt(osp.join(dirpath,file_name.replace('.jpg','.pt240')), landmark240_new)
            # print('costs:',time.time() - t1)
            # img_240 = draw_points(landmark240_new, img_src)
            # cv2.imwrite(os.path.join(save_draw_land240_path, file_name.replace('.jpg', '_137.png')),img_137)
            # cv2.imwrite(os.path.join(save_draw_land240_path, file_name.replace('.jpg', '_96.png')),img_96)

            # resized_img = cv2.resize(img_240,(384,384))
