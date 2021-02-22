import numpy as np
import pandas as pd
import os
import os.path as osp
import matplotlib.pyplot as plt
import shutil

def database_init():
    #path = "./euler/0.7400284__2ED7028E-57EA-4383-9483-AF351B474DE5.euler"
    #pt = np.loadtxt(path).astype(np.float32).reshape(1,3)
    database = pd.DataFrame()
    return database

def read_euler(path):

    pt_raw = np.loadtxt(path).astype(np.float32)
    #pt_raw = list(pt_raw).append(path)
    pt_raw = pd.Series(pt_raw,index=['yaw','pitch','roll'])


    return pt_raw

def pd_join(pt,database):
    #print(pt,database)
    database = database.append(pt,ignore_index=True)
    return database
def make_data(img_dir_path,file_name):
    database = database_init()

    print(database)
    #img_dir_path = "./all-data"
    for dirpath, dirnames, filenames in os.walk(img_dir_path):


        for file_name in filenames:
            if file_name.startswith('.'):
                continue
            img_path = osp.join(dirpath, file_name)
            pt = read_euler(img_path)

            database = pd_join(pt, database)
    print(database.describe())
    database.to_csv("./eu_big.csv")


def make_csv(img_dir_path,file_name):
    database = database_init()

    print(database)
    # img_dir_path = "./all-data"
    for dirpath, dirnames, filenames in os.walk(img_dir_path):

        for file_name in filenames:
            if file_name.startswith("."):
                continue
            if file_name.endswith('.euler'):
                img_path = osp.join(dirpath, file_name)
                pt = read_euler(img_path)
                ph = pd.Series(img_path,index=['path'])
                long_pt = pd.concat([ph,pt], axis=0)
                database = pd_join(long_pt, database)


    database.to_csv("./2018_26_with_name.csv")
    print(database)
    hist = database.hist(column=['pitch'], bins=100)

    plt.title("pitch")
    plt.show()

def make_4_list():
    file_name = "./with_name.csv"
    csv_df = pd.read_csv(file_name, low_memory=False, index_col=0)  # 防止弹出警告
    df = csv_df[csv_df.yaw >= 0].path
    df.to_csv('yaw>=0.txt', sep='\t', index=False)
    print(df)

    af = csv_df[csv_df.yaw < 0].path
    af.to_csv('yaw<0.txt', sep='\t', index=False)
    print(af)

    bf = csv_df[csv_df.pitch >= 0].path
    bf.to_csv('pitch>=0.txt', sep='\t', index=False)
    print(bf)

    cf = csv_df[csv_df.pitch < 0].path
    cf.to_csv('pitch<0.txt', sep='\t', index=False)
    print(cf)

def make_2018_csv():
    #ddfout是2018文件夹生成的euler文件夹
    path = "./train_all"
    database = database_init()
    for dirpath, dirnames, filenames in os.walk(path):

        for file_name in filenames:
            if file_name.startswith("."):
                continue
            img_path = osp.join(dirpath, file_name)
            pt = read_euler(img_path)
            ph = pd.Series(img_path, index=['path'])
            #long_pt = pd.concat([ph, pt], axis=0)
            database = pd_join(pt, database)
    print(database)
    database.to_csv("./eu.csv")

def csv2hist():
    #file_name = "./2018.csv"
    file_name = "./eu_big.csv"
    file_name = './eu.csv'
    csv_df = pd.read_csv(file_name, low_memory=False, index_col=0)  # 防止弹出警告

    #csv_df['0'] = csv_df.apply(lambda x: -x[0] , axis=1)

    print(csv_df.describe())

    sster = '1'
    hist = csv_df.hist(column=[sster], bins=100)
    plt.title('pitch')
    plt.show()

def concet():
    df1 = "./eu_withname.csv"
    df2 = "./2018with_name.csv"
    df3 = './2018_9_with_name.csv'
    df4 = './2018_15_with_name.csv'
    df5 = './2018_26_with_name.csv'
    csv_df1 = pd.read_csv(df1, low_memory=False, index_col=0)  # 防止弹出警告
    csv_df2 = pd.read_csv(df2, low_memory=False, index_col=0)  # 防止弹出警告
    csv_df3 = pd.read_csv(df3, low_memory=False, index_col=0)  # 防止弹出警告
    csv_df4 = pd.read_csv(df4, low_memory=False, index_col=0)  # 防止弹出警告
    csv_df5 = pd.read_csv(df5, low_memory=False, index_col=0)  # 防止弹出警告
    csv_df2 = csv_df2[csv_df2.pitch < 0]
    csv_df3 = csv_df3[csv_df3.pitch < 0]
    csv_df4 = csv_df4[csv_df4.pitch < 0]
    csv_df5 = csv_df5[csv_df5.pitch < 0]


    result = csv_df1.append([csv_df2,csv_df3,csv_df4,csv_df5])
    print(result.describe())
    result.info()
    sster = 'roll'
    hist = result.hist(column=[sster], bins=100)
    plt.title(sster)
    plt.show()

def make_pitchok_list():

    file_name = "./2018_26_with_name.csv"
    csv_df = pd.read_csv(file_name, low_memory=False, index_col=0)  # 防止弹出警告
    df = csv_df[csv_df.pitch <= 0].path
    df.to_csv('ddfout26_pitch.txt', sep='\t', index=False)
    print(df)

def txt2copyimg():
    txt = './ddfout_pitch.txt'
    mid_dir =['bainian','baoquan','five','heart','one','yeah','zan']
    with open(txt, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            img_path =line.replace('.euler','.jpg')
            filename =  img_path.split('/')[2]
            for mid in mid_dir:
                old = '/data/2018-11-01/'+mid+'/'+filename
                new = '/data/ddfout_img/'+mid+'/'+filename
                try:
                    shutil.copy(old, new)
                    print('ok')
                except:
                    print(old,new)
                    pass
def cut_csv():
    pass

def pro_eu():
    file_name = "./eu.csv"
    csv_df = pd.read_csv(file_name, low_memory=False, index_col=0)  # 防止弹出警告
    csv_df.pitch =csv_df.pitch*180/np.pi
    csv_df.yaw = csv_df.yaw * 180 / np.pi
    csv_df.roll = csv_df.roll * 180 / np.pi
    csv_df['yaw'] = csv_df.apply(lambda  x:-x['yaw'],axis=1)
    # 将‘校区’修改为‘所属校区’，将‘All’修改为‘总计’
    csv_df = csv_df.rename(columns={'pitch': 'yaw', 'yaw': 'pitch'})
    csv_df.to_csv("./eu_withname.csv")


if __name__ == '__main__':

    path = "./ddfout26"
    #path ="./testfind/train_data_all_big"
    #path ="./train_all"
    #csv2hist()
    #
    file_name = "./eu_big.csv"
    #make_2018_csv()
    #make_csv(path,file_name)
    #make_pitchok_list()
    #txt2copyimg()
    #make_data(path,file_name)
    concet()
    #pro_eu()

    #csv_df = csv_df/np.pi*180
    #csv_df.drop(['2'],axis=1,inplace=True)
    #print(csv_df.describe())
    #hist = csv_df.hist(column=['2'], bins=100)