#!/usr/bin/python

from math import sqrt, cos, acos, degrees
import types
#~ import numpy as np
import pdb
class Atom:
	def __init__(self,name,pos,index):
		self.name=name
		self.pos=pos
		self.index=index	
	def __str__(self):
		return self.name+" "+str(self.pos[0])+" "+str(self.pos[1])+" "+str(self.pos[2])
		
	def __eq__(self,atom2):
		return self.name==atom2.name and self.pos==atom2.pos and self.index==atom2.index
		
	def __hash__(self):
		p=re.compile("\.|-")
		return int(p.sub("",str(ord(self.name))+str(self.pos[0])+str(self.pos[1])+str(self.pos[2])+str(self.index)))	
			
	def sqdist(self, atom2,periodic=None):
		r=self.posdiff(atom2,periodic)
		return r[0]*r[0]+r[1]*r[1]+r[2]*r[2]
		
	def dist(self,atom2,periodic=None):
		return sqrt(self.sqdist(atom2,periodic))
		
	def posdiff(self, atom2,periodic=None):
		diff=[0,0,0]
		for i in xrange(3):
			diff[i]=atom2.pos[i]-self.pos[i]
		if periodic is not None:
			for i in xrange(3):
				while diff[i]>periodic[i]/2:
					diff[i]-=periodic[i]
				while diff[i]<-periodic[i]/2:
					diff[i]+=periodic[i]
			return diff
		else:
			return diff
			
	def next_neighbor(self, atoms, pbc):
		nearestAtom=None
		nearestsqdist=1e10
		for atom in atoms:
			sqdist= self.sqdist(atom, pbc)
			if sqdist < nearestsqdist:
				if atom is not self:
					nearestAtom=atom
					nearestsqdist=sqdist
		return (nearestAtom, sqrt(nearestsqdist))

	def dist_sort(self, atoms, selection=None, pbc=None):
		if type(selection) == str:
			selection = [selection]
			def filter_function(a):
				return a.name in selection
		elif type(selection) == types.NoneType:
			def filter_function(a):
				return True

		newlist = sorted(atoms, key=lambda a: self.sqdist(a, pbc))
		return filter(filter_function, newlist)
			

def angle(v1,v2):
	return degrees(acos((v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])/(sqrt(v1[0]*v1[0]+v1[1]*v1[1]+v1[2]*v1[2])*sqrt(v2[0]*v2[0]+v2[1]*v2[1]+v2[2]*v2[2]))))

def findX(name,atom, atomlist, dist, number, pbc=None): 
	sqdist=dist*dist
	nearest_atoms=[]
	for atom2 in atomlist:
		if atom.sqdist(atom2, pbc)<sqdist and atom2.name==name and atom2.index!=atom.index:
			nearest_atoms.append(atom2)
	nearest_atoms.sort(cmp=lambda x,y: cmp(atom.sqdist(x,pbc),atom.sqdist(y,pbc)))
	return nearest_atoms[:number]	
	
def hbond(O1,O2,H, pbc=None):
	if O1.sqdist(O2)<=9:
		if 1.96<=O1.sqdist(H, pbc)<=3 or 1.96<=O2.sqdist(H, pbc)<=3:
			#~ pdb.set_trace()
			if 0<=angle(O1.posdiff(H, pbc),H.posdiff(O2, pbc))<60:
					return True
	else:
		return False
			
#if __name__=="__main__":
	
