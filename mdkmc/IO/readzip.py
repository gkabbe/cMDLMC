#!/usr/bin/env python3

import zipfile, sys

if zipfile.is_zipfile(sys.argv[1]):
	zf=zipfile.ZipFile(sys.argv[1])
	datei=zf.open(zf.filelist[0])
	for i in range(int(sys.argv[2])):
		print(datei.readline()[:-1])
	
