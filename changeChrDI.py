#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@Author       : windz
@Date         : 2020-07-02 15:38:05
LastEditTime: 2020-08-20 16:01:27
@Description  : 
'''

import sys

chrID = {
    '1': 'chr1',
    '2': 'chr2',
    '3': 'chr3',
    '4': 'chr4',
    '5': 'chr5',
    'Pt': 'chrC',
    'Mt': 'chrM'
}

with open(sys.argv[1], 'r') as f:
    for i in f:
        if i[0] == '#':
            print(i, end='')
            continue
        i = i.rstrip().split('\t')
        #print(i)
        i[0] = chrID[i[0]]
        print('\t'.join(i))
