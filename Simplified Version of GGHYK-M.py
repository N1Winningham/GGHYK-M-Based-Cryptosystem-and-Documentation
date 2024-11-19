import numpy as np
import os

r = 0.5  # Perturbation vector (should be between about 10-50 for 400 dimensions)
n = 2  # Dimension of the lattice
chunkSize = n  # Number of characters to read at a time
e = np.array([1, -1])   # Rounding error
unicodeMax = 1114111  # Maximum Unicode value

# Function to read text in the input file
def readFile(filePathname):
    with open(filePathname, 'r') as file:
        while True:
            chunk = file.read(chunkSize)    # Breaks the text into chunks in the number of dimensions so that operations can be performed
            if not chunk:
                break
            asciiValues = [ord(char) for char in chunk]  # Convert characters to ASCII values
            if len(asciiValues) < chunkSize:
                asciiValues.append(0)  # Pad with null if the last chunk would not be filled completely
            yield np.array(asciiValues) # Returns the item in the list instead of the entire list

# Encryption function
def encrypt(basis, message, e):
    cipherText = np.dot(basis, message) + e  # Encryption formula
    return cipherText

# Decryption function
def decrypt(basis, message, e):
    basisInv = np.linalg.inv(basis)  # Inverse of the basis matrix
    decryptedText = np.dot(basisInv, message) - e  # Decryption formula
    decryptedText[0] = np.ceil(decryptedText[0])    # Round the first value up
    decryptedText[1] = np.floor(decryptedText[1])   # Round the second value down
    return decryptedText

# Function to write the result to a file on the Desktop
def writeFile(outputPathname, dataChunks):
    desktopPathname = os.path.expanduser("~/Desktop")  # Path to the Desktop
    outputFile = os.path.join(desktopPathname, outputPathname)  # Combines the Desktop path with the rest of the directory path
    
    with open(outputFile, 'w') as file:
        # Write characters (without newlines or spaces) to the file
        p = 0
        for chunk in dataChunks:
            print(p, [int(value) for value in chunk])
            p += 1
            file.write(''.join(chr(int(value)) for value in chunk if value != 0))  # Exclude padding
    print(f"Data saved to {outputFile}")

# Function to process encrypted numbers and generate adjustments
def processNumbersWithAdjustments(encryptedChunks):
    processedNumbers = []
    adjustments = []

    for chunk in encryptedChunks:
        for value in chunk:
            adjustment = 0
            isNegative = value < 0

            if isNegative:
                value = abs(value)
                adjustment += 1  # Mark as odd for negativity

            while value > unicodeMax:
                value -= unicodeMax
                adjustment += 2  # Increment adjustment for each overflow

            if 0 <= value <= 31:
                value += 31  # Shift the value out of the restricted range
                adjustment += 900000000  # Mark as large for restricted range

            processedNumbers.append(value)
            adjustments.append(adjustment)

    return processedNumbers, adjustments


# Function to apply adjustments during decryption
def applyAdjustments(processedNumbers, adjustments):
    adjustedNumbers = []
    for i, value in enumerate(processedNumbers):    # Runs through the list and access the index and content in that index
        adjustment = adjustments[i]
        if adjustment >= 900000000: # Large adjustment indicates the value was in the restricted range
            adjustment -= 900000000  
            value -= 31  # Reverse the addition of 31 made during encryption
        if adjustment % 2 == 1:  # Odd adjustment indicates the value was negative
            value = -value
        value += (adjustment // 2) * unicodeMax  # Add back overflows
        adjustedNumbers.append(value)
    return adjustedNumbers

# Function to write the decrypted vales of the processed numbers and adjustments to files
def writeProcessedAndAdjustments(processedNumbers, adjustments):
    desktopPath = os.path.expanduser("~/Desktop")

    # ciphertext file
    processedFile = os.path.join(desktopPath, "ciphertext.txt")
    with open(processedFile, 'w') as file:
        # Write processed numbers as Unicode characters
        file.write(''.join(chr(int(value)) for value in processedNumbers if value != 0))
    
    # Adjustments file
    adjustmentsFile = os.path.join(desktopPath, "adjustments.txt")
    with open(adjustmentsFile, 'w') as file:
        for adjustment in adjustments:
            file.write(f"{adjustment}\n")
    
    print(f"Processed numbers saved to {processedFile}")
    print(f"Adjustments saved to {adjustmentsFile}")

# Main Program
if __name__ == "__main__":
    print("Select an option:")
    print("1 - Encrypt")
    print("2 - Decrypt")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        # Encryption
        filePathname = input("Enter the path to the text file to encrypt: ").strip()
        if not os.path.isfile(filePathname):
            print(f"Error: File '{filePathname}' not found.")
            exit(1)

    elif choice == "2":
        # Decryption
        processedFile = input("Enter the path to the ciphertext file: ").strip()
        adjustmentsFile = input("Enter the path to the adjustments file: ").strip()

        if not os.path.isfile(processedFile) or not os.path.isfile(adjustmentsFile):
            print(f"Error: One or both required files not found.")
            exit(1)

    else:
        print("Invalid option, please enter 1 or 2.")
        exit(1)

    # Basis matrix input
    rows = int(input("Enter the number of rows for the basis matrix: "))
    columns = int(input("Enter the number of columns for the basis matrix: "))
    goodBasis = []
    for i in range(rows):
        row = list(map(int, input(f"Enter row {i+1} (space-separated values): ").split()))
        if len(row) != columns:
            print(f"Error: Row {i+1} must have {columns} elements.")
        goodBasis.append(row)
    goodBasis = np.array(goodBasis)

    # Assign values to the operations
    if choice == "1":
        encryptedChunks = []
        for asciiChunk in readFile(filePathname):
            encryptedChunk = encrypt(goodBasis, asciiChunk, e)
            encryptedChunks.append(encryptedChunk)

        # Process encrypted numbers and generate adjustments
        processedNumbers, adjustments = processNumbersWithAdjustments(encryptedChunks)

        # Write processed numbers and adjustments to files
        writeProcessedAndAdjustments(processedNumbers, adjustments)

    elif choice == "2":
        # Read processed numbers as Unicode characters and convert to numeric values
        with open(processedFile, 'r') as file:
            processedNumbers = [ord(char) for char in file.read()]

        # Read adjustments from the adjustments file
        with open(adjustmentsFile, 'r') as file:
            adjustments = [int(line.strip()) for line in file]

        # Apply adjustments to the numeric values
        adjustedNumbers = applyAdjustments(processedNumbers, adjustments)

        # Decrypt the adjusted numbers
        decryptedChunks = []
        for i in range(0, len(adjustedNumbers), 2):
            chunk = adjustedNumbers[i:i + 2]
            if len(chunk) == 2:
                decryptedChunk = decrypt(goodBasis, chunk, e)
                decryptedChunks.append(decryptedChunk)

        # Write decrypted text to a file
        writeFile(f"decrypted_text.txt", decryptedChunks)
