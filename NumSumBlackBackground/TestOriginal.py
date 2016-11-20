#!/usr/bin/python
import sys

res = {}
n = 0
with open('res.txt') as file:	
    data = file.read()
    lines = data.split('\n')
    for id, line in enumerate(lines):
        if(id>0):
            cols = line.split('\t')
            if(cols[0] == ''):
                continue
            cols[1] = cols[1].replace('\r', '')
            res[cols[0]] = cols
            n += 1

correct = 0
student = []
with open("out.txt") as file:
    data = file.read()
    lines = data.split('\n')
    for id, line in enumerate(lines):
        cols = line.split('\t')
        if(cols[0] == ''):
            continue
        if(id==0):
            student = cols  
        elif(id>1):
            cols[1] = cols[1].replace('\r', '')
            if (res[cols[0]] == cols):
                correct += 1

print student
print 'Tacnih:\t'+str(correct)
print 'Ukupno:\t'+str(n)
print 'Uspeh:\t'+str(100*correct/n)+'%'