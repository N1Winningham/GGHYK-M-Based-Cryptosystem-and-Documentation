import numpy as np
import matplotlib.pyplot as plt

# Demonstration
latRange = 6  # Defines the number of points shown on the lattice
e = 0.5   # Defines the perturbation value
n = 2  # Dimension of the lattice
I = (n, n) # Identity matrix
# gamma = np.random.uniform(1, 1000)


# Message input
message = input("Enter the secret message: ")  # Input message
asciiMessage = [ord(char) for char in message]  # Convert message to ASCII values

# Generate Bases
goodBasis = np.array([[2, 1], [1, 2]])  # Private basis (W)
badBasis = np.array([[4, 1], [1, 3]])  # Public basis (U)

# Generate lattice points based on a basis
def generateLatticePoints(basis, multiplierRange=range(-latRange, latRange)):
    points = [i * basis[:, 0] + j * basis[:, 1] for i in multiplierRange for j in multiplierRange]
    return np.array(points)

# Lattice Generation
privateLatticePoints = generateLatticePoints(goodBasis)
publicLatticePoints = generateLatticePoints(badBasis)

# Map ASCII values to lattice points
def encodeMessage(asciiValues, latticePoints):
    maxValue = max(asciiValues)
    return np.array([latticePoints[min(len(latticePoints)-1, int((ascii / maxValue) * (len(latticePoints) - 1)))] for ascii in asciiValues])    # (c)

# Function to add Perturbation
def addPerturbation(points):
    perturbation = np.random.uniform(-e, e, points.shape)
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
