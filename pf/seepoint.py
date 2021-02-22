import numpy as np
import cv2
import time
import os
import os.path as osp
import scipy.io

def draw_points(pts,img):

    h, w, _ = img.shape
    ori_img_show_p4 = cv2.resize(img, (w * 4, h * 4))
    idx = 0
    for point in pts:

        cv2.circle(ori_img_show_p4, (int(point[0] * 4), int(point[1] * 4)), 30, (0, 0, 255), 1)
        cv2.putText(ori_img_show_p4, str(idx), (int(point[0] * 4), int(point[1] * 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)
        idx = idx + 1

    return ori_img_show_p4

def gj_land240_to_68(land240pts):
    index_240_to_96 = [54,56,58,60,62,65,68,71,0,            #左边脸 9
                       3,6,9,12,14,16,18,20,                 #右边脸 8
                       75,77,79,81,83,                       #左眉毛 5
                       84,86,88,90,92,                       #右眉毛 5
                       94,95,96,97,                          #鼻子4
                       98,99,100,101,102,                    #鼻子5
                       107,110,114,117,120,124,              #左眼睛 6
                       127,130,134,137,140,144,              #右眼睛 6
                       179,181,184,186,188,191,193,195,198,200,202,205, #嘴12
                       208,212,216,219,223,227,230,232                  #嘴8
                      ]
    landmarks_96 = land240pts[index_240_to_96, :]
    return landmarks_96

def land273_to_68(land273pts):
    index_240_to_96 = [54, 56, 58, 60, 62, 65, 68, 71, 0 ,              #左边脸 9
                       3, 6, 9, 12, 14, 16, 18, 20,              #右边脸 8
                       74,76,77,78,80,       #左上眉毛 5
                       87,89,90,91,93,       #右上眉毛 5
                       100,101,102,103,      #鼻子
                       104,105,106,107,108,  #鼻子
                       125,129,132,136,140,143,     #左眼
                       147,151,154,158,162,165,     #右眼
                       169,172,174,177,180,182,185,  #上外唇
                       188,190,193,195,197,          #下外唇
                       201,205,209,213,217,          #上内唇
                       221,225,229                   #上外唇
                      ]
    landmarks_96 = land273pts[index_240_to_96, :]
    return landmarks_96
def land240_to_68(land273pts):
    index_240_to_96 = [54, 56, 58, 60, 62, 65, 68, 71, 0 ,              #左边脸 9
                       3, 6, 9, 12, 14, 16, 18, 20,              #右边脸 8
                       74,76,77,78,80,       #左上眉毛 5
                       87,89,90,91,93,       #右上眉毛 5
                       100,101,102,103,      #鼻子
                       104,105,106,107,108,  #鼻子
                       125,129,132,136,140,143,     #左眼
                       147,151,154,158,162,165,     #右眼
                       169,172,174,177,180,182,185,  #上外唇
                       188,190,193,195,197,          #下外唇
                       201,205,209,213,217,          #上内唇
                       221,225,229                   #上外唇
                      ]
    landmarks_96 = land273pts[index_240_to_96, :]
    return landmarks_96
def see273():
    img_dir_path = r"./data/video_2125_90.jpg"
    print(img_dir_path)

if __name__ == '__main__':


    img_dir_path = r"./data"
    #img_dir_path = r"C:\Users\1\Documents\15503"
    # save_draw_land240_path = r"/home/colomi/Desktop/Data/faceLandmark/temp"
    #
    # if not os.path.exists(save_draw_land240_path):
    #     os.makedirs(save_draw_land240_path)
    idx = 0
    for dirpath,dirnames,filenames in os.walk(img_dir_path):
        for file_name in filenames:
            print(file_name)
            if not file_name.endswith('.jpg'):continue
            t1 = time.time()
            img_path = osp.join(dirpath,file_name)
            img_src = cv2.imread(img_path)
            pt_path = img_path.replace('jpg','pt273')
            if not osp.exists(pt_path):
                print('240 pts not exists')
            pt_raw = np.loadtxt(pt_path).astype(np.float32)

            #print(pt_raw)
            # landmark240_new = pts_1k_to_240(land1kpts)
            pt68_path = img_path.replace('jpg', 'pt273')
            pt68_path = img_path.replace('.jpg', 'p68.mat')
            pt273_path = img_path.replace('.jpg', 'p273.mat')

            landmark137 = land273_to_68(pt_raw)
            #scipy.io.savemat(pt68_path,{'pts':landmark137})
            #scipy.io.savemat(pt273_path, {'p273ts': pt_raw})

            # print(idx , info)
            # cv2.imshow('fff',img_src)
            # cv2.waitKey()
            # landmark96 = land240_to_96(landmark240_new)

            #这里
            img_cp= img_src.copy()
            img_137 = draw_points(landmark137, img_src)
            img_273 = draw_points(pt_raw, img_cp)
            cv2.namedWindow("enhanced", 0);
            cv2.resizeWindow("enhanced", 540, 960);
            cv2.imshow("enhanced", img_273)
            cv2.waitKey(0)
            cv2.imwrite("./jj.jpg",img_273)

            # landmark_240 = np.savetxt(osp.join(dirpath,file_name.replace('.jpg','.pt240')), landmark240_new)
            # print('costs:',time.time() - t1)
            # img_240 = draw_points(landmark240_new, img_src)
            # cv2.imwrite(os.path.join(save_draw_land240_path, file_name.replace('.jpg', '_137.png')),img_137)
            # cv2.imwrite(os.path.join(save_draw_land240_path, file_name.replace('.jpg', '_96.png')),img_96)

            # resized_img = cv2.resize(img_240,(384,384))