##This cryptosystem takes inspiration from GGHYK-M while changing properties and adding a scramble to ensure exact matrices are required to give the proper output.

import numpy as np
import os

unicodeMax = 1114111  # Maximum Unicode value


# Function to read text in the input file
def readFile(filePathname, chunkSize):
    with open(filePathname, 'r') as file:
        while True:
            chunk = file.read(chunkSize)  # Reads a chunk of the file
            if not chunk:
                break
            asciiValues = [ord(char) for char in chunk]  # Converts characters to ASCII values
            if len(asciiValues) < chunkSize:
                asciiValues.extend([0] * (chunkSize - len(asciiValues)))  # Pad with null if needed
            yield np.array(asciiValues)  # Yield ASCII values as a NumPy array


# Encryption function
def encrypt(basis, message):
    scrambledValues, remainderIndices = scramble(basis, message)  # Scrambles the message
    
    basisSize = basis.shape[0]  # Gets the size of the basis matrix

    # Adds padding if needed
    if len(scrambledValues) % basisSize != 0:
        paddingNeeded = basisSize - (len(scrambledValues) % basisSize)
        scrambledValues.extend([0] * paddingNeeded)

    # Reshape into chunks of basis size - directly using numpy arrays
    scrambledValues = np.array(scrambledValues).reshape(-1, basisSize)
    
    result = []
    for pair in scrambledValues:
        # Perform matrix multiplication directly with floats
        res = np.dot(pair, basis.T)
        #print(res, "\n")   # Debugging
        result.extend(res.flatten())

    return [float(x) for x in result], remainderIndices


# Decryption function
def decrypt(basis, message, remainderIndices):
    basisInv = np.linalg.inv(basis)  # Inverts the basis matrix
    message = np.array(message, dtype=np.float64)  # Ensures message is a float64 NumPy array
    
    # Checks if the message size is a multiple of the basis size
    basisSize = basis.shape[0]
    if message.size % basisSize != 0:
        raise ValueError("Message size is not a multiple of basis size")
    
    # Reshape message into chunks of basis size
    message = message.reshape(-1, basisSize)
    #print(f"Message chunks to process: {len(message)}")    # Debugging
    
    result = []  # Stores final decrypted characters

    # Decryption loop for each chunk
    for i, chunk in enumerate(message):
        #print(f"\nProcessing chunk {i+1}:")    # Debugging
        #print(f"Input chunk: {chunk}") # Debugging
        
        # Matrix multiplication with inverse basis
        res = np.dot(basisInv, chunk)
       #print(f"After matrix multiplication: {res}")    # Debugging
        
        # Gets remainder indices for this chunk
        chunkStart = i * basisSize
        chunkEnd = chunkStart + basisSize
        chunkRemainders = {}
        for k, v in remainderIndices.items():
            if chunkStart < len(v):
                chunkRemainders[k] = v[chunkStart:chunkEnd] # Extracts the remainder indices for this chunk
            else:
                chunkRemainders[k] = [0] * basisSize    # If the remainder index is out of range, pad with zeros
        
        # Applies unscrambling
        decryptedValues = unscramble(basis, res, chunkRemainders)
        #print(f"After unscrambling: {decryptedValues}")    # Debugging
        
        # Converts numerical values to characters (without filtering)
        for val in decryptedValues:
            if isinstance(val, (int, float)):
                charVal = int(round(val))  # Round and convert to integer
                #print(charVal) # Debugging
                char = chr(charVal)  # Convert directly to character
                if char != '\0':  # Exclude null characters
                    result.append(char)
    
    decryptedText = ''.join(result)
    #print(f"\nDecrypted text: {decryptedText}")
    return decryptedText
    

# Function to process encrypted numbers and generate adjustments
def processNumbersWithAdjustments(encryptedChunks):
    # Unpack the tuple if needed
    if isinstance(encryptedChunks, tuple):
        encryptedChunks = encryptedChunks[0]

    # Flatten nested lists/arrays
    def flatten(lst):   # 'lst' because 'list' is a built-in function
        flatList = []
        for item in lst:
            if isinstance(item, (list, tuple, np.ndarray)):
                flatList.extend(flatten(item))
            elif isinstance(item, dict):
                continue
            else:
                flatList.append(item)
        return flatList

    encryptedChunks = flatten(encryptedChunks)
    encryptedChunks = [x for x in encryptedChunks if isinstance(x, (int, float))]   # Filter out non-numeric values
    encryptedChunks = np.array(encryptedChunks, dtype=np.float64)  # Convert to NumPy array

    processedNumbers = []
    adjustments = []

    # Apply adjustments to each value
    for value in encryptedChunks:
        adjustment = 0
        
        intValue = int(np.round(value))
        isNegative = intValue < 0

        # Handles negative numbers
        if isNegative:
            intValue = abs(intValue)
            adjustment += 1  # Track negative values with +1

        # Handles overflows
        overflows = intValue // unicodeMax  # Applies floor division
        if overflows > 0:
            adjustment += overflows * 2  # +2 for each time it exceeds 1114111
            intValue = intValue % unicodeMax  # Keep it within the Unicode range

        # Handle special character ranges
        if 0 <= intValue <= 31:
            intValue += 31
            adjustment += 0.5  # Adjust low control characters
        
        # Handles surrogate pairs
        if 55296 <= intValue <= 57343:
            intValue += 2047
            adjustment += 0.333  # Adjust surrogate pairs

        processedNumbers.append(intValue)
        adjustments.append(adjustment)

        #print(f"Processed: {intValue}, Adjustment: {adjustment}")  # Debugging

    return processedNumbers, adjustments


# Function to apply adjustments during decryption
def applyAdjustments(processedNumbers, adjustments):
    adjustedNumbers = []

    # Undoes the adjustments
    for i, value in enumerate(processedNumbers):
        adjustment = adjustments[i]

        # Reverse surrogate pair adjustment
        if round(adjustment % 1, 3) == 0.333:
            value -= 2047
            adjustment -= 0.333  # Ensure tracking remains accurate

        # Reverse control character adjustment
        if adjustment % 1 == 0.5:
            value -= 31
            adjustment -= 0.5

        # Reverse overflow adjustment
        if int(adjustment) >= 2:
            overFlow = int(adjustment) // 2 # Finds number of times the overflow occurred
            value += unicodeMax * overFlow
            adjustment -= 2 * overFlow

        # Reverse negative number adjustment
        if int(adjustment) == 1:
            value = -value
            adjustment -= 1

        adjustedNumbers.append(value)

    return adjustedNumbers


# Function to validate the key is an M-matrix
def isMMatrix(matrix):
    matrix = np.array(matrix, dtype=np.float64)  # Ensure it's a float64 NumPy array
    
    # Check if off-diagonal elements are nonpositive
    offDiagonal = matrix - np.diag(np.diagonal(matrix))
    if np.any(offDiagonal > 0):  # Checks to see if any off-diagonal elements are positive
        print("Error: Off-diagonal elements must be nonpositive.")
        return False

    # Checks that diagonal elements are greater than the sum of the absolute values of the off-diagonal elements
    diagonal = np.diagonal(matrix)
    rowSums = np.sum(np.abs(matrix), axis=1) - np.abs(diagonal)  # Calculate the row sums
    if np.any(diagonal <= rowSums):  # Checks to see if any diagonal elements are less than or equal to the row sums
        print("Error: Matrix is not strictly diagonally dominant.")
        return False
    
    # Checks if the determinant is non-positive (using log determinant method to avoid overflow)
    sign, lodget = np.linalg.slogdet(matrix)
    if sign <= 0:  # Checks if the determinant is non-positive
        print(f"Error: Determinant is non-positive.")
        return False
    
    # Check if all eigenvalues have non-negative real parts
    eigenvalues = np.linalg.eigvals(matrix)  # Get the eigenvalues of the matrix
    nonNegativeRealParts = np.all(np.real(eigenvalues) >= 0)  # Check if all eigenvalues have non-negative real parts
    if nonNegativeRealParts == False:
        print("Error: Matrix has eigenvalues with negative real parts.")
        return False

    return True


# Function to write processed numbers and adjustments to files
def writeProcessedAndAdjustments(processedNumbers, adjustments):
    desktopPath = os.path.expanduser("~/Desktop")  # Get the Desktop path

    processedFile = os.path.join(desktopPath, "ciphertext.txt")
    with open(processedFile, 'w') as file:
        for value in processedNumbers:
            if 0 <= value < 0x110000:
                file.write(chr(int(value)))
            else:
                file.write(f"?\n")  # Placeholder for out-of-range values

    adjustmentsFile = os.path.join(desktopPath, "adjustments.txt")
    with open(adjustmentsFile, 'w') as file:
        for adjustment in adjustments:
            file.write(f"{adjustment}\n")
    
    print(f"Processed numbers saved to {processedFile}")
    print(f"Adjustments saved to {adjustmentsFile}")


# Function to auto-generate a random M-matrix of size n x n
def generateAndSaveMMatrix(n):
    while True:
        # Generate the diagonal matrix with diagonal elements
        matrix = np.eye(n) * 2 + np.random.randint(0, 1, (n, n))  # Generates the base matrix
        np.fill_diagonal(matrix, np.random.randint(1, 9999999, n))  # Fills diagonal elements with value from 1 to a large number
        
        # Get the smallest diagonal value
        smallestDiagonal = np.min(np.diagonal(matrix))
        
        # Replace the off-diagonal values to be between 0 and the negative of the smallest diagonal value divided by the size of the matrix
        for i in range(n):
            for j in range(n):
                if i != j:  # Skip the diagonal elements
                    matrix[i, j] = np.random.randint(-(smallestDiagonal/n), 1)  # Fills off-diagonal with values between -smallestDiagonal/n and 0
        
        # Check if the matrix is an M-matrix
        if isMMatrix(matrix):
            desktopPath = os.path.expanduser("~/Desktop")
            matrixFile = os.path.join(desktopPath, "m-matrix.txt")
            np.savetxt(matrixFile, matrix, fmt='%d')  # Save the matrix to file
            print(f"Generated M-matrix saved to {matrixFile}")
            return matrix


# Function to scramble the values
def scramble(basis, message):
    # Convert basis to numpy array and validate
    basis = np.array(basis, dtype=np.float64)
    if len(basis.shape) != 2 or basis.shape[0] != basis.shape[1]:
        raise ValueError("Basis must be a square matrix")
    
    # Flatten basis matrix
    basis = basis.flatten()
    
    # Convert message to numpy array of float64
    if isinstance(message, np.ndarray):
        message = message.tolist()
    values = np.array([ord(str(c)) if isinstance(c, str) else c for c in message], dtype=np.float64)
    # print("Initial values:", values)  # Debugging
    
    # Dictionary to store remainder indices
    remainderIndices = {}
    
    # Process each basis value
    for i, val in enumerate(basis):
        operationType = i % 4
        
        if operationType == 0:  # Multiply every 4th value (1,5,9,etc.)
            if val != 0:
                values *= val
                #print(f"After multiplication by {val}: ", values)  # Debugging
            # elif val == 0 do nothing (multiply by one)
            
        elif operationType == 1:  # Divide every 4th value (2,6,10,etc.)
            if val != 0:  # Protect against division by zero
                remainders = values % abs(val)  # Use absolute divisor for remainder
                quotient = (values - remainders) / val  # Compute correct integer division

                remainderIndices[i] = remainders.tolist()
                values = quotient
            # elif val == 0 do nothing (divide by one)
            
            #print(f"After division by {val}: ", values)    # Debugging
            #if i in remainderIndices:
            #    print(f"Remainders stored at index {i}: ", remainderIndices[i])    # Debugging
            
        elif operationType == 2:  # Add every 4th value (3,7,11,etc.)
            values += val
            #print(f"After addition of {val}: ", values)    # Debugging
            
        elif operationType == 3:  # Subtract every 4th value (4,8,12,etc.)
            values -= val
            #print(f"After subtraction of {val}: ", values) # Debugging
    
    return values.tolist(), remainderIndices


# Function to unscramble the values
def unscramble(basis, encryptedValues, remainderIndices):
    #print(f"Starting unscramble with values: {encryptedValues}")   # Debugging
    basis = np.array(basis).flatten()
    values = np.array(encryptedValues)
    
    # Unscramble the values using reverse order of scrambling
    for i in range(len(basis) - 1, -1, -1):
        val = basis[i]
        operationType = i % 4
        
        if val == 0:
            continue
            
        #print(f"Reversing operation {i} (type {operationType}) with value {val}")  # Debugging
        #print(f"Before: {values}") # Debugging
        
        if operationType == 3:  # Reverse subtraction
            values = values + val
        elif operationType == 2:  # Reverse addition
            values = values - val
        elif operationType == 1:  # Reverse division
            if val != 0:
                if i in remainderIndices:
                    values = values * val + np.array(remainderIndices[i])
                else:
                    values = values * val
            # elif val == 0 do nothing (multiply by one)
        elif operationType == 0:  # Reverse multiplication
            if val != 0:
                values = values / val
            # elif val == 0 do nothing (divide by one)
                
        #print(f"After: {values}")  # Debugging
    
    return values


# Main Program
if __name__ == "__main__":
    print("Select an option:")
    print("1 - Encrypt")
    print("2 - Decrypt")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        # Main encryption loop
        filePathname = input("Enter the path to the text file to encrypt (must be a .txt file): ").strip()
        if not os.path.isfile(filePathname):
            print(f"Error: File '{filePathname}' not found.")
            exit(1)

        m_matrixFile = input("Enter the path to the m-matrix file or leave blank to auto-generate: ").strip()

        # Checks if the M-matrix file exists and then either loads it or auto-generates one if prompted
        if m_matrixFile:
            if not os.path.isfile(m_matrixFile):
                print(f"Error: M-matrix file '{m_matrixFile}' not found.")
                exit(1)

            m_matrix = np.loadtxt(m_matrixFile, dtype=int)
            n = m_matrix.shape[0]
        else:
            n = int(input("Enter the size of the M-matrix to auto-generate: ").strip())
            m_matrix = generateAndSaveMMatrix(n)

        encryptedChunks = []
        allRemainderIndices = {}

        # Encrypt the file in chunks
        for asciiChunk in readFile(filePathname, n):
            encryptedChunk, chunkRemainderIndices = encrypt(m_matrix, asciiChunk)
            encryptedChunks.append(encryptedChunk)
            # Merge remainder indices for all chunks
            for key, value in chunkRemainderIndices.items():
                if key not in allRemainderIndices:
                    allRemainderIndices[key] = []
                allRemainderIndices[key].extend(value)

        # Save remainder indices to a file
        desktopPath = os.path.expanduser("~/Desktop")
        remainderFile = os.path.join(desktopPath, "remainder_indices.txt")
        with open(remainderFile, 'w') as f:
            for key, value in allRemainderIndices.items():
                f.write(f"{key}:{','.join(map(str, value))}\n")

        # Process encrypted numbers and generate adjustments file
        processedNumbers, adjustments = processNumbersWithAdjustments(encryptedChunks)
        #print(processedNumbers)    # Debugging
        
        # Write processed numbers and adjustments to files
        writeProcessedAndAdjustments(processedNumbers, adjustments)

    elif choice == "2":
        # Main decryption loop
        processedFile = input("Enter the path to the ciphertext file: ").strip()
        adjustmentsFile = input("Enter the path to the adjustments file: ").strip()
        remainderFile = input("Enter the path to the remainder indices file: ").strip()

        # Check if files exist
        if not os.path.isfile(processedFile) or not os.path.isfile(adjustmentsFile) or not os.path.isfile(remainderFile):
            print(f"Error: One or more required files not found.")
            exit(1)

        # Read the m-matrix file
        m_matrixFile = input("Enter the path to the m-matrix file: ").strip()
        if not os.path.isfile(m_matrixFile):
            print(f"Error: M-matrix file '{m_matrixFile}' not found.")
            exit(1)

        # Load the matrix
        m_matrix = np.loadtxt(m_matrixFile, dtype=int)
        n = m_matrix.shape[0]

        # Load remainder indices from file
        remainderIndices = {}
        with open(remainderFile, 'r') as f:
            for line in f:
                key, values = line.strip().split(':')   # Split key and values
                remainderIndices[int(key)] = [float(x) for x in values.split(',')]  # Convert to list of floats

        # Read processed numbers and adjustments
        with open(processedFile, 'r', encoding='utf-8') as file:
            processedNumbers = [ord(char) for char in file.read()]

        # Read adjustments
        with open(adjustmentsFile, 'r') as file:
            adjustments = [float(line.strip()) for line in file]

        # Apply adjustments
        adjustedNumbers = applyAdjustments(processedNumbers, adjustments)

        decryptedChunks = []

        # Decrypt the numbers in chunks
        decryptedText = ""
        for i in range(0, len(adjustedNumbers), n):
            chunk = adjustedNumbers[i:i + n]    # Extract the chunk
            
            if len(chunk) == n:
                chunkRemainderIndices = {
                    k: v[i:i+n] for k, v in remainderIndices.items()    # Extract the remainder indices for this chunk
                }
                
                decryptedChunk = decrypt(m_matrix, np.array(chunk), chunkRemainderIndices)
                if decryptedChunk:  # Only add non-empty chunks
                    decryptedText += decryptedChunk
        
        # Write the complete decrypted text
        with open(os.path.join(os.path.expanduser("~/Desktop"), "decrypted_text.txt"), 'w') as f:
            f.write(decryptedText)
            #print(f"Wrote {len(decryptedText)} characters to file")    # Debugging

    else:
        print("Invalid option, please enter 1 or 2.")
        exit(1)

