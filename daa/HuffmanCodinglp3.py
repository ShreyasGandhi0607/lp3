import heapq
from collections import defaultdict, Counter

# Node class for Huffman Tree
class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    # Defining comparators for priority queue
    def __lt__(self, other):
        return self.freq < other.freq

# Function to generate Huffman codes
def generate_codes(root, current_code, codes):
    if root is None:
        return

    if root.char is not None:  # Leaf node
        codes[root.char] = current_code

    generate_codes(root.left, current_code + "0", codes)
    generate_codes(root.right, current_code + "1", codes)

# Huffman Coding Function
def huffman_encoding(text):
    # Calculate frequency of each character
    frequency = Counter(text)
    
    # Priority queue for the Huffman Tree nodes
    heap = [Node(char, freq) for char, freq in frequency.items()]
    heapq.heapify(heap)

    # Build the Huffman Tree
    while len(heap) > 1:
        # Pop two nodes with the lowest frequency
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        # Create a new internal node with combined frequency
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right

        # Push the new node back into the heap
        heapq.heappush(heap, merged)

    # Root of Huffman Tree
    root = heap[0]

    # Generate Huffman codes from the tree
    codes = {}
    generate_codes(root, "", codes)

    # Encode the input text
    encoded_text = ''.join(codes[char] for char in text)
    
    return encoded_text, codes

# Huffman Decoding Function
def huffman_decoding(encoded_text, codes):
    # Reverse the codes dictionary to decode
    reverse_codes = {v: k for k, v in codes.items()}
    
    decoded_text = ""
    current_code = ""
    for bit in encoded_text:
        current_code += bit
        if current_code in reverse_codes:
            decoded_text += reverse_codes[current_code]
            current_code = ""
    
    return decoded_text

# Example usage
text = "hello huffman coding"
encoded_text, codes = huffman_encoding(text)
decoded_text = huffman_decoding(encoded_text, codes)

print("Original Text:", text)
print("Encoded Text:", encoded_text)
print("Huffman Codes:", codes)
print("Decoded Text:", decoded_text)
