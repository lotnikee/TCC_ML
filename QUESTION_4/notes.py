import ase.db
from dscribe.descriptors import SOAP
import numpy as np 
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist


db = ase.db.connect("/Users/lotburgstra/Desktop/TCC_ML/QUESTION_4/ase.db")

structures = []
for row in db.select():
    structures.append(row.toatoms())

all_species = set()
for atoms in structures: 
    all_species.update(atoms.get_chemical_symbols())
all_species = sorted(all_species)

print("Species in my dataset:", all_species)

dists = pdist(s.get_positions())
print("Largest pairwise distance:", np.max(dists))