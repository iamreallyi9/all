import os
import numpy as np
import scipy.io
def read_obj(objFilePath):

    with open(objFilePath) as file:
        vertices = []
        triangles =[]
        colors = []

        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                vertices.append((float(strs[1]), float(strs[2]), float(strs[3])))
                colors.append((float(strs[4]), float(strs[5]), float(strs[6])))

            if strs[0] == "vt":
                break
            if strs[0] == "f":
                print(strs)
                triangles.append((int(strs[1].split('//')[-1]), int(strs[2].split('//')[-1]), int(strs[3].split('//')[-1])))

    # points原本为列表，需要转变为矩阵，方便处理
    vertices = np.array(vertices)
    triangles = np.array(triangles)-1
    return vertices,triangles,colors


if __name__ == '__main__':
    objFilePath = './emma.obj'
    #objFilePath = './video_92_390.obj'
    vertices,triangles,colors = read_obj(objFilePath)
    print(vertices,triangles,colors)

    # 假设你要保存的变量为 bData and aData
    # 将bData 和 aData保存到result.mat中
    scipy.io.savemat('result.mat', mdict={'vertices': vertices, 'triangles': triangles,'colors': colors})
    # 此时result.mat包含一个cell，内容是名为'bData'和'aData'的struct
