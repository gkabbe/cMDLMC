#!/usr/bin/env python3

PERIODIC=[29.122, 25.354, 12.363]

from atoms import atomclass as ac
import numpy as np

from .xyzparser import XYZFile

class HexaXYZFile(XYZFile):

	def __init__(self, filename):
		super(HexaXYZFile, self).__init__(filename)
		self.pbc = PERIODIC
		self.atomdict["acidic_H"] = self.determine_acidic_protons()



	def parse_frame_OHs(self):
		Os=[]
		Hs=[]
		self.datei.readline()
		self.datei.readline()
		for index in range(self.atomnr):
			line = self.datei.readline()
			if index in self.atomdict["O"]:
				Os.append(ac.atom(line.split()[0],list(map(float,line.split()[1:4])),index))
			elif index in self.atomdict["acidic_H"]:
				Hs.append(ac.atom(line.split()[0],list(map(float,line.split()[1:4])),index))
		return (Os,Hs)

	def parse_frame_Os_numpy(self, n):
		arr = np.zeros((n,3))
		self.datei.readline()	
		i = 0
		for index in range(self.atomnr):
			line = self.datei.readline()
			if index in self.atomdict["O"]:
				arr[i] = list(map(float, line.split()[1:4]))
				i += 1
		return arr

	def parse_frames_Os_numpy_inplace(self, Oarr):
		"""Expects numpy array in the shape (frames, oxygennumber, 3)"""
		frames = Oarr.shape[0]
		for frame in range(frames):
			line = self.datei.readline()
			if line == "":
				return frame
			self.datei.readline()
			Oindex = 0
			for atom in range(self.atomnr):
				line = self.datei.readline()
				if atom in self.atomdict["O"]:
					Oarr[frame, Oindex, :] = list(map(float, line.split()[1:]))
					Oindex += 1
		return frames

	def parse_frame_Os_numpy_inplace(self, Oarr):
		line = self.datei.readline()
		if line == "":
			return frame
		self.datei.readline()
		Oindex = 0
		for atom in range(self.atomnr):
			line = self.datei.readline()
			if atom in self.atomdict["O"]:
				Oarr[Oindex, :] = list(map(float, line.split()[1:]))
				Oindex += 1
		return frames

	def determine_acidic_protons(self):
		acidic_Hs = set()
		self.datei.seek(0)
		atoms = self.parse_frame_full()
		self.datei.seek(0)

		Hs = [atom for atom in atoms if atom.index in self.atomdict["H"]]

		for H in Hs:
			next_neighbor, dist = H.next_neighbor(atoms, self.pbc)
			if next_neighbor.name == "O":
				acidic_Hs.add(H.index)
		return acidic_Hs