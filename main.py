from ast import dump
import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import axes3d
import cv2


def func1(tensor,savepath,index):
    '''
    visualize one of the dimension
    '''
    dict = {0:[0,0],1:[0,1],2:[0,2],3:[0,3],4:[1,0],5:[1,1],6:[1,2],7:[1,3],8:[2,0],9:[2,1],10:[2,2],11:[2,3],12:[3,0],13:[3,1],14:[3,2],15:[3,3]}

    tensor = (tensor-torch.min(tensor))/(torch.max(tensor)-torch.min(tensor))*255

    array = tensor[:,dict[index][0],dict[index][1]].numpy()
    canvas = np.ones((800,800,3))*255
    colorbar = np.arange(256)
    colorbar = colorbar.astype(np.uint8)
    colorbar = cv2.applyColorMap(colorbar, cv2.COLORMAP_JET) 

    array2 = np.zeros(256,)
    for i in range(256):
        array2[i] = (array[i*8:i*8+8]).sum()/8

    array2 = array2.astype(np.uint8)
    column_bar = np.zeros((256,1,3))
    for i in range(256):
        column_bar[i] = colorbar[array2[i]]
    column_bar = np.broadcast_to(column_bar,(256,20,3))

    def trans(IMG):
        zero_array = np.zeros((20,20,3))
        img = IMG.copy()
        img = np.concatenate((zero_array,img),axis=0)
        src = np.float32([[0,0], [0,1], [10,0]])
        dst = np.float32([[0,0], [0,1], [10,-10]])
        M = cv2.getAffineTransform(src, dst)
        img = cv2.warpAffine(img, M, (20,276))
        return img

    side = trans(column_bar)
    new_column_bar = np.zeros((276,40,3))
    new_column_bar[0:276,20:40] = side
    new_column_bar[20:276,0:20] = column_bar
    
    b = np.array([[[0,20], [20,0],[40,0], [20,20]]], dtype = np.int32)

    cv2.fillPoly(new_column_bar, b,[int(colorbar[int(array2.mean())][0][0]),int(colorbar[int(array2.mean())][0][1]),int(colorbar[int(array2.mean())][0][2])])
    new_column_bar = new_column_bar[10:276,0:30]
    for x_i in range(266):
        for y_j in range(30):
            if new_column_bar[x_i][y_j][0] == 0 and new_column_bar[x_i][y_j][1] == 0 and new_column_bar[x_i][y_j][2] == 0:
                new_column_bar[x_i][y_j] = [255,255,255]
    cv2.line(new_column_bar,(0,10),(20,10),(0,0,0),2)
    cv2.line(new_column_bar,(0,10),(10,0),(0,0,0),2)
    cv2.line(new_column_bar,(10,1),(30,1),(0,0,0),2)
    cv2.line(new_column_bar,(30,0),(20,10),(0,0,0),2)
    cv2.line(new_column_bar,(1,10),(1,266),(0,0,0),2)
    cv2.line(new_column_bar,(20,10),(20,266),(0,0,0),2)
    cv2.line(new_column_bar,(0,264),(20,264),(0,0,0),2)
    cv2.line(new_column_bar,(28,0),(28,256),(0,0,0),2)
    cv2.line(new_column_bar,(20,264),(28,256),(0,0,0),2)

    canvas[267:533,385:415] = new_column_bar
    dict2 = {0:[394,268],1:[374,268],2:[354,268],3:[334,268],4:[404,258],5:[384,258],6:[364,258],7:[344,258],8:[414,248],9:[394,248],10:[374,248],11:[354,248],12:[424,237],13:[404,237],14:[384,237],15:[364,237]}
    x0,y0 = dict2[index][0],dict2[index][1]
    Length,Width,Height = 40,80,256
    box_color = (0,0,255)
    cuboid = [[x0,y0],[x0,y0+Height],[x0-Length,y0+Length],[x0-Length,y0+Length+Height],[x0+Width,y0],[x0+Width,y0+Height],[x0+Width-Length,y0+Length],[x0+Width-Length,y0+Length+Height]]
    cv2.line(canvas,(cuboid[0][0],cuboid[0][1]),(cuboid[1][0],cuboid[1][1]),box_color,1)
    cv2.line(canvas,(cuboid[0][0],cuboid[0][1]),(cuboid[2][0],cuboid[2][1]),box_color,1)
    cv2.line(canvas,(cuboid[0][0],cuboid[0][1]),(cuboid[4][0],cuboid[4][1]),box_color,1)
    cv2.line(canvas,(cuboid[1][0],cuboid[1][1]),(cuboid[5][0],cuboid[5][1]),box_color,1)
    cv2.line(canvas,(cuboid[1][0],cuboid[1][1]),(cuboid[3][0],cuboid[3][1]),box_color,1)
    cv2.line(canvas,(cuboid[2][0],cuboid[2][1]),(cuboid[3][0],cuboid[3][1]),box_color,1)
    cv2.line(canvas,(cuboid[3][0],cuboid[3][1]),(cuboid[7][0],cuboid[7][1]),box_color,1)
    cv2.line(canvas,(cuboid[2][0],cuboid[2][1]),(cuboid[6][0],cuboid[6][1]),box_color,1)
    cv2.line(canvas,(cuboid[6][0],cuboid[6][1]),(cuboid[7][0],cuboid[7][1]),box_color,1)
    cv2.line(canvas,(cuboid[4][0],cuboid[4][1]),(cuboid[6][0],cuboid[6][1]),box_color,1)
    cv2.line(canvas,(cuboid[5][0],cuboid[5][1]),(cuboid[7][0],cuboid[7][1]),box_color,1)
    cv2.line(canvas,(cuboid[4][0],cuboid[4][1]),(cuboid[5][0],cuboid[5][1]),box_color,1)
    cv2.imwrite(savepath,canvas)

def func2():
    '''
    visualize all of the point 
    '''
    colormap = plt.cm.get_cmap('RdYlBu_r')
    C = 12
    H = 4
    W = 3
    point = torch.rand(C,H,W)
    
    x_axis = [int(x/(H*W)) for x in range(C*H*W)]
    y_axis = [int((x-int(x/(H*W))*(H*W))/W) for x in range(C*H*W)]
    z_axis = [x%W for x in range(C*H*W)]
    d = torch.reshape(point,(C*H*W,)).numpy()

    plt.figure('Visualize 3D Feature Map')
    ax = plt.gca(projection='3d')
    ax.set_xlabel('x', fontsize=6)
    ax.set_ylabel('y', fontsize=6)
    ax.set_zlabel('z', fontsize=6)
    plt.tick_params(labelsize=10)
    ax.scatter(x_axis, y_axis, z_axis, s=120, c=d, cmap=colormap,alpha=0.9,marker=',')
    plt.show()


if __name__ == '__main__':
    tensor = torch.rand(2048,4,4)
    for i in range(16):
        savepath = str(i)+'.jpg'
        func1(tensor,savepath,index=i)
