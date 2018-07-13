# -*- coding:utf-8 -*-
# /bin/python

import json
import csv

outfile = open("review.tsv", 'w')
sfile = csv.writer(outfile, delimiter ="\t", quoting=csv.QUOTE_MINIMAL)
sfile.writerow(['stars','text'])
with open('E:\\project\\yelp_dataset\\dataset\\review.json','rb') as f:
    for line in f:
        row = json.loads(line)
        # some special char must be encoded in 'utf-8'
        sfile.writerow([row['stars'], (row['text']).encode('utf-8')])
outfile.close()
