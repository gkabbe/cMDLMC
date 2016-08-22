import numpy as np

from atoms import atomclass
from atoms import numpyatom as npa

ATOMFORMAT = "{:<6}{:5}  {:4}{:1}{:3} {:1}{:4}{:1}   {:04.3f}{:04.3f}{:04.3f}{:01.2f}{:01.2f}      {:4}{:2}{:>2}"
ACCEPTED_FIRST_COLUMN = ["ATOM", "HETATM"]

def parse_frame_xyz_numpy(f):
	atoms = []
	for line in f:
		spl = f.readline().split()
		if spl[0] in ACCEPTED_FIRST_COLUMN:
			atoms.append((spl[-1], list(map(float, spl[5:8]))))

	return np.array(atoms, dtype = npa.xyzatom)

def parse_frame_full_numpy(f):
	"""specification according to http://deposit.rcsb.org/adit/docs/pdb_atom_format.html#ATOM"""
	atoms = []

	for line in f:
		if line.split()[0] in ACCEPTED_FIRST_COLUMN:
			try:
				atoms.append((line[0:6], line[6:11], line[12:16], line[16], line[17:20], line[21], int(line[22:26]), line[26], list(map(float,[line[30:38], line[38:46], line[46:54]])),\
			 float(line[54:60]), float(line[60:66]), line[72:76], line[76:78], line[78:80]))
			except:
				print("Error while parsing following line:")
				print(line)

	return np.array(atoms, dtype = npa.pdbatom)

def write_frame_numpy(f, atoms):
	for atom in atoms:
		print(ATOMFORMAT.format(atom["rec"], atom["serialnr"], atom["name"], atom["locind"], atom["resname"], atom["chainid"], atom["rsn"], atom["rescode"],\
			atom["pos"][0], atom["pos"][1], atom["pos"][2], atom["occ"], atom["tempfac"], atom["segment"], atom["element"], atom["charge"]), file=f)

def print_frame_numpy(atoms):
	"""specification according to http://deposit.rcsb.org/adit/docs/pdb_atom_format.html#ATOM (not quite yet -> indentation!)"""
	for atom in atoms:
		print(ATOMFORMAT.format(atom["rec"], atom["serialnr"], atom["name"], atom["locind"], atom["resname"], atom["chainid"], atom["rsn"], atom["rescode"],\
			atom["pos"][0], atom["pos"][1], atom["pos"][2], atom["occ"], atom["tempfac"], atom["segment"], atom["element"], atom["charge"]))
