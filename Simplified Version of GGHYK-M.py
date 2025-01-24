import numpy as np
import os

# Constants
r = 0.5  # Perturbation vector
e = 0  # Rounding error vector
unicodeMax = 1114111  # Maximum Unicode value
newline_replacement_char = chr(0x2028)  # Unicode character for Line Separator

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
def encrypt(basis, message, e):
    cipherText = np.dot(basis, message) + e  # Matrix-vector multiplication with addition of rounding error
    return cipherText

# Decryption function
def decrypt(basis, message, e):
    basisInv = np.linalg.inv(basis)  # Compute the inverse of the basis matrix
    adjustedMessage = np.dot(basisInv, message - e)  # Apply the inverse and subtract the rounding vector
    roundedMessage = np.rint(adjustedMessage).astype(int)  # Round to the nearest integers
    return roundedMessage

# Function to write the result to a file on the Desktop
def writeFile(outputPathname, dataChunks):
    desktopPathname = os.path.expanduser("~/Desktop")  # Get the Desktop path
    outputFile = os.path.join(desktopPathname, outputPathname)  # Construct the output file path

    with open(outputFile, 'w') as file:
        for chunk in dataChunks:
            file.write(''.join(chr(int(value)) for value in chunk if 0 <= value < 0x110000))  # Safely handle Unicode range
    print(f"Data saved to {outputFile}")

# Function to process encrypted numbers and generate adjustments
def processNumbersWithAdjustments(encryptedChunks):
    processedNumbers = []
    adjustments = []

    for chunk in encryptedChunks:
        for value in chunk:
            adjustment = 0
            isNegative = value < 0  # Check if the value is negative

            if isNegative:
                value = abs(value)  # Convert to positive (or apply absolute value)
                adjustment += 1  # Track adjustment (negative)

            while value > unicodeMax:
                value -= unicodeMax
                adjustment += 2  # Track adjustment (overflow)

            if value == ord('\n'):
                value = ord(newline_replacement_char)  # Replace newline character
                adjustment += 0.75  # Track newline replacement adjustment

            if 0 <= value <= 31:
                value += 31  # Avoid the control characters
                adjustment += 0.5  # Track adjustment (control)

            if 55296 <= value <= 57343:
                value -= 2047   # Avoid the surrogate pairs
                adjustment += 0.333 # Track adjustment (surrogate)

            processedNumbers.append(value)
            adjustments.append(adjustment)

    return processedNumbers, adjustments

# Function to apply adjustments during decryption
def applyAdjustments(processedNumbers, adjustments):
    adjustedNumbers = []
    for i, value in enumerate(processedNumbers):
        adjustment = adjustments[i]
        if adjustment % 1 != 0:
            if adjustment % 1 == 0.75:
                value = ord('\n')  # Reverse newline replacement
            else:
                adjustment -= 0.5
                value -= 31  # Reverse control adjustment
        if int(adjustment) % 2 == 1:
            value = -value  # Reverse negative adjustment
        value += (int(adjustment) // 2) * unicodeMax  # Reverse overflow wrapping
        if round(adjustment % 1, 3) == 0.333:
            value += 2047  # Reverse surrogate adjustment
        adjustedNumbers.append(value)
    return adjustedNumbers

# Function to validate the key is an M-matrix
def isMMatrix(matrix):
    # Check if off-diagonal elements are less than or equal to 1
    off_diagonal = matrix - np.diag(np.diagonal(matrix))
    if np.any(np.abs(off_diagonal) > 1):
        print("Error: Off-diagonal elements must have absolute values less than or equal to 1.")
        return False
    # Check if diagonal elements are greater than 1
    if np.any(np.diagonal(matrix) <= 1):
        print("Error: Diagonal elements must be greater than 1.")
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
        matrix = np.eye(n) * 2 + np.random.randint(-1, 2, (n, n))  # Fills diagonal elements > 1
        np.fill_diagonal(matrix, np.random.randint(2, 9999999, n))  # Diagonal elements from 2 to a large number
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

        m_matrix_file = input("Enter the path to the m-matrix file or leave blank to auto-generate: ").strip()

        if m_matrix_file:
            if not os.path.isfile(m_matrix_file):
                print(f"Error: M-matrix file '{m_matrix_file}' not found.")
                exit(1)

            goodBasis = np.loadtxt(m_matrix_file, dtype=int)
            n = goodBasis.shape[0]

        else:
            n = int(input("Enter the size of the M-matrix to auto-generate: ").strip())
            goodBasis = generateAndSaveMMatrix(n)

        e = np.array([(-1)**i for i in range(n)])  # Rounding error vector of size n

        encryptedChunks = []
        for asciiChunk in readFile(filePathname, n):
            encryptedChunk = encrypt(goodBasis, asciiChunk, e)
            encryptedChunks.append(encryptedChunk)

        # Process encrypted numbers and generate adjustments
        processedNumbers, adjustments = processNumbersWithAdjustments(encryptedChunks)

        # Write processed numbers and adjustments to files
        writeProcessedAndAdjustments(processedNumbers, adjustments)

    elif choice == "2":
        # Decryption
        processedFile = input("Enter the path to the ciphertext file: ").strip()
        adjustmentsFile = input("Enter the path to the adjustments file: ").strip()

        if not os.path.isfile(processedFile) or not os.path.isfile(adjustmentsFile):
            print(f"Error: One or both required files not found.")
            exit(1)

        m_matrix_file = input("Enter the path to the m-matrix file: ").strip()
        if not os.path.isfile(m_matrix_file):
            print(f"Error: M-matrix file '{m_matrix_file}' not found.")
            exit(1)

        goodBasis = np.loadtxt(m_matrix_file, dtype=int)
        n = goodBasis.shape[0]
        e = np.array([(-1)**i for i in range(n)])

        # Read processed numbers as Unicode characters and convert to numeric values
        with open(processedFile, 'r') as file:
            processedNumbers = [ord(char) for char in file.read()]

        # Read adjustments from the adjustments file
        with open(adjustmentsFile, 'r') as file:
            adjustments = [float(line.strip()) for line in file]

        # Apply adjustments to the numeric values
        adjustedNumbers = applyAdjustments(processedNumbers, adjustments)

        # Decryption of the final values
        decryptedChunks = []
        for i in range(0, len(adjustedNumbers), n):
            chunk = adjustedNumbers[i:i + n]
            if len(chunk) == n:
                decryptedChunk = decrypt(goodBasis, np.array(chunk), e)
                decryptedChunks.append(decryptedChunk)

        # Write the decrypted text to a file
        writeFile(f"decrypted_text.txt", decryptedChunks)

    else:
        print("Invalid option, please enter 1 or 2.")
        exit(1)
