import numpy as np
import matplotlib.pyplot as plt

# Demonstration
latRange = 6  # Defines the number of points shown on the lattice
e = 0   # Roudning error
n = 2  # Dimension of the lattice
r = 0.5 # Perturbation vector
I = (n, n) # Identity matrix
gamma = np.random.randint(n, n+1000) # Random gamma value
P = np.random.choice([-1, 0, 1], size=(n, n))
B = gamma * I + P
A = B^(-1)
i = 0 # Row index
j = 0 # Column index
k = 0
h = 0


# Generate lattice points based on B (Parameter 1)
def generateLatticePoints(basis, multiplierRange=range(-latRange, latRange)):
    u = np.array([i, j])  # Integer vector u ∈ Z^n (2D in this case)
    point = np.dot(B, u)  # Multiply matrix B by vector u to get the lattice point
    points.append(point)
    return np.array(points)

# Genereates the Orthogonality Defect of B (Parameter 2)
def orthogonalityDefect(B):
    vectorLength = np.linalg.norm(B, axis=1) # Determines the length of each vector in basis B
    detB = np.linalg.det(B) # Determines the determinant of the matrix
    oDefect = np.prod(vectorLength) / abs(detB) # Determines the orthogonality defect

# (Parameter 8 and 4)
if (i == j):
    A[i,j] <= 2/gamma
else:
    A[i,j] < 2/gamma^2

# (Parameter 7)
2k + 2h < gamma

# Message input
message = input("Enter the secret message: ")  # Input message
asciiMessage = [ord(char) for char in message]  # Convert message to ASCII values

# Generate Bases
goodBasis = np.array([[2, 1], [1, 2]])  # Private basis (W)
badBasis = np.array([[4, 1], [1, 3]])  # Public basis (U)

# Lattice Generation
privateLatticePoints = generateLatticePoints(goodBasis)
publicLatticePoints = generateLatticePoints(badBasis)

# Map ASCII values to lattice points
def encodeMessage(asciiValues, latticePoints):
    maxValue = max(asciiValues)
    return np.array([latticePoints[min(len(latticePoints)-1, int((ascii / maxValue) * (len(latticePoints) - 1)))] for ascii in asciiValues])    # (c)

# Function to add Perturbation
def addPerturbation(points):
    perturbation = np.random.uniform(-r, r, points.shape)
    # Perturbation = np.zeros(n)    # This assigns all values of the perturbation vector to 0
    # nonZeroPurturbation = np.random.choice(n, size=k, replace=False)
    return points + perturbation

# Encode message to lattice points
messageLatticePoints = encodeMessage(asciiMessage, privateLatticePoints)
perturbedPoints = addPerturbation(messageLatticePoints)

# Plotting
plt.figure(figsize=(12, 10))

# Plot private basis lattice
plt.subplot(2, 2, 1)
plt.scatter(privateLatticePoints[:, 0], privateLatticePoints[:, 1], color='blue', label='Private Lattice Points')
plt.scatter(messageLatticePoints[:, 0], messageLatticePoints[:, 1], color='green', label='Encoded Message Points', marker='x')
plt.title("Private Basis Lattice with Message Points")
plt.grid(True)
plt.legend()

# Plot public basis lattice
plt.subplot(2, 2, 2)
plt.scatter(publicLatticePoints[:, 0], publicLatticePoints[:, 1], color='red', label='Public Lattice Points')
plt.scatter(messageLatticePoints[:, 0], messageLatticePoints[:, 1], color='green', label='Encoded Message Points', marker='x')
plt.title("Public Basis Lattice with Message Points")
plt.grid(True)
plt.legend()

# Plot perturbed points on private basis
plt.subplot(2, 2, 3)
plt.scatter(privateLatticePoints[:, 0], privateLatticePoints[:, 1], color='blue', label='Private Lattice Points')
plt.scatter(perturbedPoints[:, 0], perturbedPoints[:, 1], color='orange', label='Perturbed Message Points', marker='x')
plt.title("Private Basis Lattice with Perturbed Points")
plt.grid(True)
plt.legend()

# Plot perturbed points on public basis
plt.subplot(2, 2, 4)
plt.scatter(publicLatticePoints[:, 0], publicLatticePoints[:, 1], color='red', label='Public Lattice Points')
plt.scatter(perturbedPoints[:, 0], perturbedPoints[:, 1], color='orange', label='Perturbed Message Points', marker='x')
plt.title("Public Basis Lattice with Perturbed Points")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
