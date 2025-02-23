import numpy as np
import os

unicodeMax = 1114111  # Maximum Unicode value

# Function to read text in the input file
def readFile(filePathname, chunkSize):
    with open(filePathname, 'r') as file:
        while True:
            chunk = file.read(chunkSize)  # Read a chunk of the file
            if not chunk:
                break
            asciiValues = [ord(char) for char in chunk]  # Convert characters to ASCII values
            if len(asciiValues) < chunkSize:
                asciiValues.extend([0] * (chunkSize - len(asciiValues)))  # Pad with null if needed
            yield np.array(asciiValues)  # Yield ASCII values as a NumPy array

# Encryption function
def encrypt(basis, message):
    
    basis_size = basis.shape[0]

    if message.size % basis_size != 0:
        padding_needed = basis_size - (message.size % basis_size)
        message = np.append(message, np.zeros(padding_needed))

    message = message.reshape(-1, basis_size)

    result = []
    for pair in message:
        res = np.dot(pair, basis.T)
        result.extend(res.flatten())
    
    # Convert to list of floats
    return [float(x) for x in result]

# Decryption function
def decrypt(basis, message):
    basis_inverse = np.linalg.inv(basis)
    message_array = np.array(message)
    
    # Reshape message into appropriate dimensions
    basis_size = basis.shape[0]
    if message_array.size % basis_size != 0:
        raise ValueError("Message size is not a multiple of basis size")
    
    message_array = message_array.reshape(-1, basis_size)
    
    # Perform inverse transformation
    result = np.dot(message_array, basis_inverse)
    
    # Round to nearest integer and convert to int type
    result = np.round(result).astype(int)
    result = result[~np.all(result == 0, axis=1)]  # Remove zero rows (padding)
    
    # Convert to characters, ensuring integers
    return ''.join(chr(int(round(x))) for x in result.flatten() if 0 <= x < unicodeMax)


# Function to write the result to a file on the Desktop
def writeFile(outputPathname, dataChunks):
    desktopPathname = os.path.expanduser("~/Desktop")  # Get the Desktop path
    outputFile = os.path.join(desktopPathname, outputPathname)  # Construct the output file path
    
    print(f"Saving decrypted data to: {outputFile}")  # Debugging output file path
    
    with open(outputFile, 'w', encoding='utf-8') as file:
        # Directly write each chunk as a string without modifying it
        for chunk in dataChunks:
            #print(f"Writing chunk: {chunk}")  # Verify the content of each chunk
            file.write(chunk)  # Write the chunk directly
    
    print(f"Decrypted data saved to {outputFile}")


# Function to process encrypted numbers and generate adjustments
def processNumbersWithAdjustments(encryptedChunks):
    processedNumbers = []
    adjustments = []

    # Flatten encryptedChunks if it contains nested lists/tuples
    encryptedChunks = np.array(encryptedChunks, dtype=np.float64).flatten()

    for value in encryptedChunks:
        adjustment = 0
        # Convert to integer for comparison while preserving the sign
        int_value = int(np.round(value))
        isNegative = int_value < 0

        if isNegative:
            int_value = abs(int_value)
            adjustment += 1  # Track adjustment (negative)

        # Handle large values
        while int_value > unicodeMax:
            int_value -= unicodeMax
            adjustment += 2

        # Handle control characters
        if 0 <= int_value <= 31:
            int_value += 31
            adjustment += 0.5

        # Handle surrogate pairs
        if 55296 <= int_value <= 57343:
            int_value += 2047
            adjustment += 0.333

        processedNumbers.append(int_value)
        ##QUICK FIX##
        adjustment = adjustment
        adjustments.append(adjustment)

    return processedNumbers, adjustments

# Function to apply adjustments during decryption
def applyAdjustments(processedNumbers, adjustments):
    adjustedNumbers = []

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

        # Reverse large value wrapping
        while int(adjustment) >= 2:
            value += unicodeMax
            adjustment -= 2

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
        
        if isMMatrix(matrix):
            desktopPath = os.path.expanduser("~/Desktop")
            matrixFile = os.path.join(desktopPath, "m-matrix.txt")
            np.savetxt(matrixFile, matrix, fmt='%d')  # Save the matrix to file
            print(f"Generated M-matrix saved to {matrixFile}")
            return matrix


# Main Program
if __name__ == "__main__":
    print("Select an option:")
    print("1 - Encrypt")
    print("2 - Decrypt")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        # Encryption
        filePathname = input("Enter the path to the text file to encrypt (must be a .txt file): ").strip()
        if not os.path.isfile(filePathname):
            print(f"Error: File '{filePathname}' not found.")
            exit(1)

        m_matrixFile = input("Enter the path to the m-matrix file or leave blank to auto-generate: ").strip()

        if m_matrixFile:
            if not os.path.isfile(m_matrixFile):
                print(f"Error: M-matrix file '{m_matrixFile}' not found.")
                exit(1)

            goodBasis = np.loadtxt(m_matrixFile, dtype=int)
            n = goodBasis.shape[0]

        else:
            n = int(input("Enter the size of the M-matrix to auto-generate: ").strip())
            goodBasis = generateAndSaveMMatrix(n)

        encryptedChunks = []
        for asciiChunk in readFile(filePathname, n):
            encryptedChunk = encrypt(goodBasis, asciiChunk)
            encryptedChunks.append(encryptedChunk)
            #print(encryptedChunk)  #Debugging

        # Process encrypted numbers and generate adjustments
        processedNumbers, adjustments = processNumbersWithAdjustments(encryptedChunks)
        # for i in processedNumbers:
        #     print(i)  # Debugging

        # Write processed numbers and adjustments to files
        writeProcessedAndAdjustments(processedNumbers, adjustments)

    elif choice == "2":
        # Decryption process
        processedFile = input("Enter the path to the ciphertext file: ").strip()
        adjustmentsFile = input("Enter the path to the adjustments file: ").strip()

        # Check if files exist
        if not os.path.isfile(processedFile) or not os.path.isfile(adjustmentsFile):
            print(f"Error: One or both required files not found.")
            exit(1)

        # Read the m-matrix file
        m_matrixFile = input("Enter the path to the m-matrix file: ").strip()
        if not os.path.isfile(m_matrixFile):
            print(f"Error: M-matrix file '{m_matrixFile}' not found.")
            exit(1)

        # Load the basis matrix
        goodBasis = np.loadtxt(m_matrixFile, dtype=int)
        n = goodBasis.shape[0]

        # Read processed numbers as Unicode characters and convert to numeric values
        with open(processedFile, 'r', encoding='utf-8') as file:
            processedNumbers = [ord(char) for char in file.read()]  # This treats each character as a number

        # Read adjustments from the adjustments file
        with open(adjustmentsFile, 'r') as file:
            adjustments = [float(line.strip()) for line in file]

        # Apply adjustments
        adjustedNumbers = applyAdjustments(processedNumbers, adjustments)

        # Decrypt the numbers in chunks
        decryptedChunks = []
        for i in range(0, len(adjustedNumbers), n):
            chunk = adjustedNumbers[i:i + n]
            if len(chunk) == n:
                decryptedChunk = decrypt(goodBasis, np.array(chunk))
                decryptedChunks.append(decryptedChunk)
            #print(decryptedChunks) # Debugging

        # Write the decrypted text to a file
        writeFile("decrypted_text.txt", decryptedChunks)

    else:
        ("Invalid option, please enter 1 or 2.")
        exit(1)
