import numpy as np

# 用于排序时存储原来像素点位置的数据结构
class Node(object):
	def __init__(self,x,y,value):
		self.x = x
		self.y = y
		self.value = value
	def printInfo(self):
		print(self.x,self.y,self.value)


def getAtomsphericLight(darkChannel, img):
    img = np.float32(img)
    height = darkChannel.shape[0]
    width = darkChannel.shape[1]
    nodes = []
    # 用一个链表结构(list)存储数据
    for i in range(0, height):
        for j in range(0, width):
            oneNode = Node(i, j, darkChannel[i, j])
            nodes.append(oneNode)
    # 排序
    nodes = sorted(nodes, key=lambda node: node.value, reverse=False)
    # print('nodes[0]',nodes[0].value)
    # print('img[nodes[0].x, nodes[0].y, 0]+img[nodes[0].x, nodes[0].y, 1])',img[nodes[0].x, nodes[0].y, 0],img[nodes[0].x, nodes[0].y, 1])
    atomsphericLight  = np.mean([img[nodes[0].x, nodes[0].y, 0],img[nodes[0].x, nodes[0].y, 1]])
    atomsphericLightGB = img[nodes[0].x, nodes[0].y, 0:2]
    atomsphericLightRGB = img[nodes[0].x, nodes[0].y, :]
    # atomsphericLight  =  [img[nodes[0].x, nodes[0].y, 0],img[nodes[0].x, nodes[0].y, 1]]
    return atomsphericLight,atomsphericLightGB,atomsphericLightRGB

def getAtomsphericLight2(darkChannel, img, percent):
    size = darkChannel.shape[0] * darkChannel.shape[1]
    height = darkChannel.shape[0]
    width = darkChannel.shape[1]
    nodes = []
    img = np.float16(img)
    for i in range(0, height):
        for j in range(0, width):
            oneNode = Node(i, j, darkChannel[i, j])
            nodes.append(oneNode)
    nodes = sorted(nodes, key=lambda node: node.value, reverse=True)
    atomsphericLight = 0

    # 获取暗通道前0.1%(percent)的位置的像素点在原图像中的最高亮度值

    for i in range(0, int(percent*size)):
        SumImg = sum(img[nodes[i].x, nodes[i].y,:])
        if SumImg > atomsphericLight:
            print('img[nodes[i].x, nodes[i].y, :]',img[nodes[i].x, nodes[i].y, :])
            atomsphericLight = SumImg
            AtomsphericLight = img[nodes[i].x, nodes[i].y, :]
            # print('nodes[i].x, nodes[i].y,SumImg',SumImg,nodes[i].x, nodes[i].y,img[nodes[i].x, nodes[i].y, :])
    return AtomsphericLight


def getAtomsphericLight3(darkChannel, img):
    # img = np.float16(img)
    height = darkChannel.shape[0]
    width = darkChannel.shape[1]
    nodes = []
    # 用一个链表结构(list)存储数据
    for i in range(0, height):
        for j in range(0, width):
            oneNode = Node(i, j, darkChannel[i, j])
            nodes.append(oneNode)
    # 排序
    nodes = sorted(nodes, key=lambda node: node.value, reverse=False)
    atomsphericLight  = img[nodes[0].x, nodes[0].y, :]
    # atomsphericLight  =  [img[nodes[0].x, nodes[0].y, 0],img[nodes[0].x, nodes[0].y, 1]]
    return atomsphericLight