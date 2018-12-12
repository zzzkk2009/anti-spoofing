# -*- coding:utf-8 -*-
import os
import cv2

def mkdir_recursively(path):
    '''
    Create the path recursively, same as os.makedirs().
    
    Return True if success, or return False.
    
    e.g.
    mkdir_recursively('d:\\a\\b\\c') will create the d:\\a, d:\\a\\b, and d:\\a\\b\\c if these paths does not exist.
    '''
    
    #First transform '\\' to '/'
    local_path = path.replace('\\', '/')
    
    path_list = local_path.split('/')
    print(path_list)
    
    if path_list is None: return False 
    
    # For windows, we should add the '\\' at the end of disk name. e.g. C: -> C:\\
    disk_name = path_list[0]
    if disk_name[-1] == ':': path_list[0] = path_list[0] + '\\'
    
    dir = ''
    for path_item in path_list:
        dir = os.path.join(dir, path_item)
        print("dir:", dir )
        if os.path.exists(dir):
            if os.path.isdir(dir):
                print("mkdir skipped: %s, already exist." % (dir,))
            else: # Maybe a regular file, symlink, etc.
                print("Invalid directory already exist:", dir )
                return False
        else:
            try:
                os.mkdir(dir)
            except Exception:
                print("mkdir error: ", dir)
                # print(e) 
                return False 
            print("mkdir ok:", dir)
    return True

def makedir_if_not_exist(dirPath):
    if not os.path.exists(dirPath):
        os.mkdir(dirPath)

def saveXML(name, objs, cls, w, h, isNir=False):
    name = name.replace('\\', '/')
    with open(name, "w") as xml:
        xml.write("<annotation><size><width>%d</width><height>%d</height></size>" % (w, h))
        for item in objs:
            fmt = """
            <object>
                <name>%s</name>
                <bndbox>
                    <xmin>%d</xmin>
                    <ymin>%d</ymin>
                    <xmax>%d</xmax>
                    <ymax>%d</ymax>
                </bndbox>
            </object>
            """
            
            xmin = int(item[0])
            ymin = int(item[1])
            xmax = int(item[2])
            ymax = int(item[3])

            if isNir:
                xmin += 20
                ymin -= 10
                xmax += 20
                ymax -= 10

            xml.write(fmt % ('object', xmin, ymin, xmax, ymax))

        xml.write("</annotation>")

    with open(name + ".txt", "w") as txt:
        cls = 0
        txt.write("%d,%d\n" % (len(objs), cls))
        for item in objs:
            xmin = int(item[0])
            ymin = int(item[1])
            xmax = int(item[2])
            ymax = int(item[3])

            if isNir:
                xmin += 20
                ymin -= 10
                xmax += 20
                ymax -= 10

            txt.write("%d,%d,%d,%d,%d,%s\n" % (xmin, ymin, xmax, ymax, cls, 'object'))
