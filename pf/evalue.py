import numpy as np
import pandas as pd
import os
import os.path as osp
import matplotlib.pyplot as plt


def gdatabase_init():
    path = "./euler/0.7400284__2ED7028E-57EA-4383-9483-AF351B474DE5.euler"
    pt = np.loadtxt(path).astype(np.float32).reshape(1,3)
    database = pd.DataFrame(pt)
    return database
def database_init():
    database = pd.DataFrame()
    return database


def read_euler(path):
    pt_raw = np.loadtxt(path).astype(np.float32)
    pt_raw = pd.Series(pt_raw)

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
            img_path = osp.join(dirpath, file_name)
            pt = read_euler(img_path)

            database = pd_join(pt, database)
    print(database.describe())
    database.to_csv("./eu_big.csv")
    hist = database.hist(column=[0], bins=100)

    plt.title("yaw")
    plt.show()

if __name__ == '__main__':

    #path = "./euler/0/8cc5e354-7f20-410d-9748-d9c9fc17cae0.euler"

    #path = r"F:\4w_euler"
    file_name = "./eu_big.csv"
    #make_data(path,file_name)

    csv_df = pd.read_csv(file_name, low_memory=False,index_col=0)  # 防止弹出警告
    csv_df = csv_df/np.pi*180
    csv_df['1'] = csv_df.apply(lambda x:-x[1],axis=1)
    #csv_df.drop(['2'],axis=1,inplace=True)
    print(csv_df.describe())
    hist = csv_df.hist(column=['2'], bins=100)

    plt.title("roll")
    plt.show()





