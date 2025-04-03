from braket.circuits import Circuit
from braket.devices import LocalSimulator
import random
import numpy as np

# Choose a matrix that is invertible in modulo 2
matrixKey = np.array([[1, 1], [1, 0]], dtype=int)

# Computes the inverse in modulo 2 arithmetic
def modularInverse(matrix, mod=2):
    # For 2x2 matrix in GF(2), the inverse has a simple form
    a, b = matrix[0, 0] % mod, matrix[0, 1] % mod
    c, d = matrix[1, 0] % mod, matrix[1, 1] % mod
    
    # Computes determinant in modulo 2
    det = (a * d - b * c) % mod
    if det == 0:
        raise ValueError("Matrix is not invertible in modulo 2")
    
    arr = np.array([[d, b], [c, a]], dtype=int)
    
    return arr % mod

# Calculates the inverse matrix
matrixKeyInv = modularInverse(matrixKey)

# Debugging
# Verifies that the key and its inverse work correctly
# def verifyInverse():
#     # Creats identity matrix
#     identity = np.eye(2, dtype=int) % 2
    
#     # Check if KEY * keyInv = I (mod 2)
#     test1 = (np.dot(matrixKey, matrixKeyInv)) % 2
#     # Check if keyInv * KEY = I (mod 2)
#     test2 = (np.dot(matrixKeyInv, matrixKey)) % 2

# Converts the message to binary
def encodeMessage(message):
    binaryMessage = ''.join(format(ord(char), '08b') for char in message)  # Converts to binary
    binaryList = [int(bit) for bit in binaryMessage]  # Converts to list of integers
    
    # Ensures binary length is even for matrix multiplication
    if len(binaryList) % 2 != 0:
        binaryList.append(0)  # Pads with zero
    
    return binaryList

# Applies matrix transformation to encode binary message
def applyMatrixEncryption(binaryMessage):
    binaryArray = np.array(binaryMessage).reshape(-1, 2)  # Reshapes into 2D array
    transformed = (np.dot(matrixKey, binaryArray.T)) % 2  # Matrix multiplication mod 2
    return transformed.T.flatten().tolist()

# Applies inverse matrix transformation to decode binary message
def applyMatrixDecryption(transformedBits):
    binaryArray = np.array(transformedBits).reshape(-1, 2)  # Reshapes into 2D array
    original = (np.dot(matrixKeyInv, binaryArray.T)) % 2  # Applies inverse matrix mod 2
    return original.T.flatten().astype(int).tolist()

# Generates a random scrambling key
def generateScrambleKey(numQubits):
    random.seed(62037805048071288749818984769982605928716074890523023955042929301231792830917332140519031258480388642442328589315262123953527728909194380136645275766169948586963776124902216249160179944561755656568904803278853093821693889242880979621954870183687257649142848210880417801945192869728526831968106424267949319349668366715859758006341575943013998853271178456781207519208015469387336068772757592642364315978304274723315282470645046783050670049574181488276977739289311864918116970534717084072218782043280952587796019506309642979172164367579215083357769431076074264163866099888611124720428117144760683653664124246909372638677489456238703709820236347414256099045705027627997568969892685753947388221903153054401146526845545340946951540328957146107046869210377835349565102446460542845593328523127882783787995773429925220802164733704362104420466083929609722090175939256270228877463183832151654690095705813600740167999639420296105860726933664792046858977485270485630029131923178133146948606996211032708654429202303782637577392784956700652858361377408617689243268537730719932379042936178201131439720817301906943998757619382042908093369047594303328351594239638312315829549287294746873446198786509903621491670720449141295461847303975601592749121006)  # Set seed for reproducibility
    key = []
    
    for i in range(numQubits):
        gate = random.choice(["X", "CNOT"])  # Randomly selects a gate
        if gate == "X":
            qubit = random.randint(0, numQubits - 1)   # Randomly selects a qubit
            key.append(("X", qubit))    # Appends X operation
        else:  # "CNOT"
            control = random.randint(0, numQubits - 2)  # Randomly selects a control qubit
            target = control + 1  # Ensures a valid target
            key.append(("CNOT", control, target))   # Appends CNOT operation
    
    return key

# Scrambles the message
def quantumScramble(binaryMessage, scrambleKey):
    circuit = Circuit()
    numQubits = len(binaryMessage)

    # Applies X gates to encode binary message
    for i in range(numQubits):
        if binaryMessage[i] == 1:
            circuit.x(i)

    # Applies random scrambling based on the key
    for operation in scrambleKey:
        if operation[0] == "X":
            circuit.x(operation[1])
        elif operation[0] == "CNOT":
            circuit.cnot(operation[1], operation[2])

    # Adds measurement
    for i in range(numQubits):
        circuit.measure(i)

    return circuit

# Classical descrambling after measurement
def classicalDescramble(measuredBits, scrambleKey):
    bits = measuredBits[:]  # Copies the measured bits

    # Applies inverse operations in reverse order
    for operation in reversed(scrambleKey):
        if operation[0] == "X":
            bits[operation[1]] ^= 1  # Flips bit
        elif operation[0] == "CNOT":
            # Reverse CNOT: if control bit is 1, flip the target bit
            if bits[operation[1]] == 1:
                bits[operation[2]] ^= 1  

    return bits

# Converts bits back to text
def bitsToText(bits):
    # Debug: prints the bit sequence before converting to text
    print("Bits before conversion to text:", bits)
    
    # Checks if we have all eight bits
    if len(bits) % 8 != 0:
        print(f"Warning: Bit length ({len(bits)}) is not a multiple of 8, padding with zeros")
        # Pads with zeros to make complete bytes
        bits = bits + [0] * (8 - (len(bits) % 8))
    
    # Groups bits into bytes and convert to characters
    chars = []
    for i in range(0, len(bits), 8):
        byteBits = bits[i:i+8]
        byteStr = ''.join(map(str, byteBits))
        try:
            byteVal = int(byteStr, 2)
            char = chr(byteVal)
            print(f"Byte {i//8}: {byteStr} -> {byteVal} -> '{char}'")
            chars.append(char)
        except ValueError as e:
            print(f"Error converting bits to character at position {i}: {e}")
            print(f"Problematic bits: {byteBits}")
    
    return ''.join(chars)

# Debugging
# Tests matrix operations
# def testMatrixOperations():
#     print("Testing matrix operations:")
#     testBits = [1, 0, 1, 1, 0, 1]
#     if len(testBits) % 2 != 0:
#         testBits.append(0)
#     
#     print("Original bits:", testBits)
#     encrypted = applyMatrixEncryption(testBits)
#     print("After encryption:", encrypted)
#     decrypted = applyMatrixDecryption(encrypted)
#     print("After decryption:", decrypted)
#     print("Encryption/decryption test:", "PASSED" if testBits == decrypted else "FAILED")
#    print()
    


#Main Function
# Sample message
message = "Hi!"
print("Original message:", message)

# Test matrix operations
#testMatrixOperations()

binaryMessage = encodeMessage(message)
print("Binary encoded message (before matrix):", binaryMessage)

encryptedMessage = applyMatrixEncryption(binaryMessage)
print("After matrix encryption:", encryptedMessage)

numQubits = len(encryptedMessage)

# Use AWS Braket simulator
device = LocalSimulator()

# Generates a random scrambling key
scrambleKey = generateScrambleKey(numQubits)
print("Generated scramble key:", scrambleKey)

# Step 1: Scramble message
scrambleCircuit = quantumScramble(encryptedMessage, scrambleKey)

# Step 2: Run the circuit on the quantum device
result = device.run(scrambleCircuit, shots=1).result()

# Extracts measurement results
scrambledOutput = result.measurements.flatten().tolist()
print("Scrambled (measured) output:", scrambledOutput)

# Step 3: Descramble classically after measurement
descrambledBits = classicalDescramble(scrambledOutput, scrambleKey)
print("After classical descrambling:", descrambledBits)

# Applies matrix decryption
finalBits = applyMatrixDecryption(descrambledBits)
print("After matrix decryption:", finalBits)

# Compares with original binary message
originalBinary = encodeMessage(message)
print("Original binary (for comparison):", originalBinary)
print("Final bits match original binary:", originalBinary == finalBits)

# Converts back to text
originalMessage = bitsToText(finalBits)

# Output
print("\nFinal Results:")
print("Original Message:", message)
print("Recovered Message:", originalMessage)
