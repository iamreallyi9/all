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

if __name__ == '__main__':


    img_dir_path = r"./data"
    new_img_dir_path = r'./new_data'
    idx = 0
    for dirpath,dirnames,filenames in os.walk(img_dir_path):
        print(dirpath,dirnames,filenames)

        for file_name in filenames:
            print(file_name)
            new_path = dirpath
            if not file_name.endswith('.jpg'):continue
            t1 = time.time()
            img_path = osp.join(dirpath,file_name)

            new_file_name = file_name[:-4]+"_fp_num.jpg"
            new_img_path = osp.join(new_img_dir_path,new_file_name)
            new_img_src = cv2.imread(new_img_path)
            new_pt_path = new_img_path.replace('jpg', 'pt273')
            new_pt_raw = np.loadtxt(new_pt_path).astype(np.float32)
            new_mg_cp = new_img_src.copy()
            new_img_273 = draw_points(new_pt_raw, new_mg_cp)

            img_src = cv2.imread(img_path)
            pt_path = img_path.replace('jpg','pt273')
            if not osp.exists(pt_path):
                print('240 pts not exists')

            pt_raw = np.loadtxt(pt_path).astype(np.float32)
            img_cp= img_src.copy()

            img_273 = draw_points(pt_raw, img_cp)
            cv2.namedWindow("enhanced", 0);
            cv2.resizeWindow("enhanced", 960, 960);
            imghstack = np.hstack((img_273, new_img_273))
            cv2.imshow("enhanced", imghstack)
            cv2.waitKey(0)
            cv2.imwrite("./jj.jpg",imghstack)
