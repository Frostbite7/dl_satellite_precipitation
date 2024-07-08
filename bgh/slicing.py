# --coding:utf-8--#
## 此部分用于裁剪出训练的数据块28×28

import numpy as np
import os
import datetime
import time
import shutil
import gzip
import sys
import math
import re

## 输入时间和空间范围
monthstart, daystart, monthend, dayend = [1, 1, 1, 1]
latitudeN, latitudeS, longtitudeW, longtitudeE = [40, 30, 100, 90]
# monthstart,daystart,monthend,dayend=sys.argv[1:5]
# latitudeN,latitudeS,longtitudeW,longtitudeE=sys.argv[5:9]
monthstart = int(monthstart)
daystart = int(daystart)
monthend = int(monthend)
dayend = int(dayend)
latitudeN = int(latitudeN)
latitudeS = int(latitudeS)
longtitudeW = int(longtitudeW)
longtitudeE = int(longtitudeE)

dstpath = "upscaled/"  # 目标目录
rootpath = "old_upscaled/"  # 原始目录
radarfile_path = 'radar/'
bghfile_path = 'goes/'
radar_pre = 'q2hrus'
bgh_pre = 'bghrus'


## 获取目标区域行列号
def GeoInfo2index(latitudeN, latitudeS, longtitudeW, longtitudeE):  ## 修改了网格数目
    row_start = int(math.floor(-500 / 40 * (latitudeN - 50.0)))  # the index to start,the index itself included
    row_end = int(math.floor(-500 / 40 * (latitudeS - 50.0)))  # the index to end,the index itself excluded
    colomn_start = int(math.floor(-875. / 70 * (longtitudeW - 135.)))  # index to start, included
    colomn_end = int(math.floor(-875. / 70 * (longtitudeE - 135.)))  # index to end,excluded
    return row_start, row_end, colomn_start, colomn_end


## 检查input和output是否配对
def CompletionCheck(FileNameStr):  # To check if the certain time file is complete(has both bgh and radar)
    ExistFlag = 0
    R_CompressFlag = 0
    B_CompressFlag = 0
    if os.path.isfile(rootpath + radarfile_path + radar_pre + FileNameStr + '.bin') == True:
        ExistFlag += 1
        if os.path.isfile(rootpath + radarfile_path + radar_pre + FileNameStr + '.bin.gz') == True:
            R_CompressFlag += 1  # the radar file is compressed or not
    if os.path.isfile(rootpath + bghfile_path + bgh_pre + FileNameStr + '.bin') == True:
        ExistFlag += 1
        if os.path.isfile(rootpath + bghfile_path + bgh_pre + FileNameStr + '.bin.gz') == True:
            B_CompressFlag += 1  # bgh file is compressed or not
    if ExistFlag == 2:
        return (R_CompressFlag, B_CompressFlag)
    else:
        return -1


## 生成时间列表
def timewalk(monthstart, daystart, monthend, dayend):  ## 修改步长
    datelist = []
    year = 2012
    Start_date = datetime.datetime(year, monthstart, daystart, 00, 00)
    Dest_Date = datetime.datetime(year, monthend, dayend, 23, 00)
    start_timestamp = time.mktime(Start_date.timetuple())
    end_timestamp = time.mktime(Dest_Date.timetuple())
    current_stamp = start_timestamp
    while current_stamp <= end_timestamp:
        ltime = time.localtime(current_stamp)
        timeStr = time.strftime("%y%m%d%H%M", ltime)
        ErrorFlag = 0
        if CompletionCheck(timeStr[0:8]) == -1:
            # That means at least one of the radar/bgh files are missing, the CompletionCheck function returned -1
            ErrorFlag = 1
            current_stamp = current_stamp + 60 * 60  # adding the iterator current_stamp
            continue
        else:
            r_compress, b_compress = CompletionCheck(timeStr[0:8])
            datelist.append((timeStr[0:8], r_compress, b_compress))
            current_stamp += 60 * 60
    return datelist


class Clouddata(object):
    def __init__(self, monthstart, daystart, monthend, dayend, N, S, W, E):
        self.monthstart = monthstart
        self.daystart = daystart
        self.monthend = monthend
        self.dayend = dayend
        self.rowstart, self.rowend, self.colomnstart, self.colomnend = \
            GeoInfo2index(N, S, W, E)
        self.datelist = timewalk(self.monthstart, self.daystart, self.monthend, \
                                 self.dayend)
        self.Timelist = []
        print
        self.datelist
        # for x in self.datelist:
        # self.Timelist.append(x[0])

    def CloudSave(self):  ## 主函数
        if os.path.isdir(dstpath) is False:
            os.mkdir(dstpath)

        for date in self.datelist:
            if date[1] == 1 and date[2] == 1:
                continue
            timename = date[0]
            print("Disposing" + timename + "data")
            Monthdate = timename[2:8]
            RadarM = np.fromfile(rootpath + radarfile_path + radar_pre + timename + '.bin', dtype=float)
            BghM = np.fromfile(rootpath + bghfile_path + bgh_pre + timename + '.bin', dtype=float)
            RadarM = RadarM.reshape(500, 875)
            BghM = BghM.reshape(500, 875)

            markers = []
            key = 0

            # ignore the cloud and get all 28*28 area
            for i in range(self.rowstart, self.rowend):
                for j in range(self.colomnstart, self.colomnend):
                    flag = 0
                    for previous_points in markers:
                        if abs(i - previous_points[0]) >= 7 or abs(j - previous_points[1]) >= 7:  ## 修改滑动步长
                            continue
                        else:
                            flag = 1
                            break
                    if flag == 0:

                        New_M_Radar = RadarM[i - 14:i + 14, j - 14:j + 14]  ## 裁剪28×28的块
                        New_M_bgh = BghM[i - 14:i + 14, j - 14:j + 14]

                        ## 判断数据质量
                        # 对于雷达数据，中间的值不能小于0
                        if New_M_Radar[14][14] < 0:
                            continue
                        # 对于亮温数据，不能有nan
                        if sum(sum(np.isnan(New_M_bgh))) > 0:
                            continue

                        key += 1
                        markers.append((i, j))
                        New_M_Radar = New_M_Radar.astype('float')
                        New_M_bgh = New_M_bgh.astype('float')
                        New_M_bgh[New_M_bgh < 200] = 200
                        New_M_bgh[New_M_bgh > 300] = 300
                        rain_mean = np.zeros(2)
                        if New_M_Radar[14, 14] > 0:
                            rain_mean[0] = 1
                        else:
                            rain_mean[1] = 1
                        rain_mean.tofile(dstpath + radarfile_path + 'R' + Monthdate + str(key) + '.bin')
                        New_M_bgh.tofile(dstpath + bghfile_path + 'B' + Monthdate + str(key) + '.bin')

            print("keymax= " + str(key))


A = Clouddata(monthstart, daystart, monthend, dayend, latitudeN, latitudeS, longtitudeW, longtitudeE)
A.CloudSave()
