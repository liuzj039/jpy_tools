#!/usr/bin/env python
from argparse import ArgumentParser
parser=ArgumentParser()
parser.add_argument('script_file',metavar='FILE',help='input code')
options=parser.parse_args()
with open(options.script_file,'r') as fh:
    contents=fh.readlines()
    output_cont=[line for line in contents if not (line == '\n')]
output_cont=''.join(output_cont)
with open(options.script_file,'w') as fh:
    fh.write(output_cont)
