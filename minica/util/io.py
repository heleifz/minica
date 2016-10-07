# -*- coding: utf-8 -*-

import urllib2
import os

def download_at(url, directory='.'):
    """
    下载数据
    """
    response = urllib2.urlopen(url)
    file_name = os.path.basename(url)
    full_path = os.path.join(directory, file_name)
    print "Downloading: %s ..." % url
    if os.path.exists(full_path):
        print "File: %s already exists." % full_path
        return False
    out = open(full_path, 'w')
    current_bytes = 0
    while True:
        data = response.read(100000)
        l = len(data)
        current_bytes += l
        if l > 0:
            out.write(data)
        else:
            break
        print "Downloaded %d KB.." % (current_bytes / 1024)

    return True
