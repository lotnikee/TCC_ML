### Import useful packages 
import ase.db
from dscribe.descriptors import SOAP
import numpy as np 
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import random

### Include a random seed for reproducibility 
random.seed(13)

### Access the training data in the ase format 
db = ase.db.connect("/Users/lotburgstra/Desktop/TCC_ML/QUESTION_4/ase.db")

### Loop over all entries in the ASE database and convert each one to an ASE Atoms object, storing them in the 'structures' list
structures = []
for row in db.select():
    structures.append(row.toatoms())

### Determine which chemical symbols are present in the structures
all_species = set()
for atoms in structures: 
    all_species.update(atoms.get_chemical_symbols())
all_species = sorted(all_species)

#############################################################
### Set up the SOAP descriptor object from DScribe
### - species: list all elements in the dataset
### - r_cut: radius cut-off for local environment (in Angstroms)
### - n_max: number of radial basis functions 
### - l_max: maximum degree of spherical harmonics
### - sparse: output as dense arrays
#############################################################

soap = SOAP(
    species = all_species,               
    r_cut = 5.0,
    n_max = 8, 
    l_max = 6,
    sparse = False
)

### Take a random sample of 15000 structures and assign to sample_structures variable
N = 15000 
idx = random.sample(range(len(structures)), N)
sample_structures = [structures[i] for i in idx]

### For each structure in 'sample_structures', compute the SOAP descriptor and store the resulting vectors in a NumPy array 'X'
X = np.array([soap.create(s).mean(axis = 0) for s in sample_structures])

### Perform kernel PCA to reduce the high dimensional SOAP feature vectors to 2D for visualisation
kpca = KernelPCA(n_components = 2, kernel = "linear")           # n_components = 2 to keep only the two most important principal components
X_kpca = kpca.fit_transform(X)

### Visualise the 2D map
plt.figure(figsize = (6,5))
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], s= 10)
plt.title("2D Structure Map via SOAP and Kernel PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()