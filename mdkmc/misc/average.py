#!/usr/bin/python
import fileinput
from math import sqrt

summe=0
counter=0

vektor=[]

for line in fileinput.input():
	try:
		for x in map(float,line.split()):
			vektor.append(x)
	except ValueError:
		pass
for zahl in vektor:
	counter+=1
	summe+=zahl
summe/=counter
sqsum=0
for zahl in vektor:
	sqsum+=(zahl-summe)*(zahl-summe)

sqsum=sqrt(sqsum/(counter-1))

print((str(summe)+" "+str(sqsum)))
