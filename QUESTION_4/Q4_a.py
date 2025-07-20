import ase.db
from dscribe.descriptors import SOAP
import numpy as np 
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

db = ase.db.connect("/Users/lotburgstra/Desktop/TCC_ML/QUESTION_4/ase.db")

structures = []
for row in db.select():
    structures.append(row.toatoms())

all_species = set()
for atoms in structures: 
    all_species.update(atoms.get_chemical_symbols())
all_species = sorted(all_species)


soap = SOAP(
    species = ["Al", "Ca", "Mg"],
    r_cut = 5.0,
    n_max = 8, 
    l_max = 6,
    sparse = False
)

X = np.array([soap.create(s).mean(axis = 0) for s in structures[:110000]])

kpca = KernelPCA(n_components = 2, kernel = "linear")
X_kpca = kpca.fit_transform(X)

plt.figure(figsize = (6,5))
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], s= 10)
plt.title("2D Structure Map via SOAP and Kernel PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()