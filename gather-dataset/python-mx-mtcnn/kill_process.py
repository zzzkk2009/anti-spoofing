# -*- coding:utf-8- *-
import os
import time


def kill_process():

    print(os.popen('tasklist /FI "IMAGENAME eq python.exe"').read().decode('cp936'))

    os.system('TASKKILL /F /IM python.exe')



if __name__=="__main__":
    
    kill_process()