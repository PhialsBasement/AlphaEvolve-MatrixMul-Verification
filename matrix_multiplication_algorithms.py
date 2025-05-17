import os
import numpy as np
import time
import requests  # For quantum random number generation

# =============================================================================
# QUANTUM RANDOM NUMBER GENERATOR FUNCTION
# =============================================================================

def get_quantum_random_numbers(count, use_complex=False):
    """Get quantum random numbers from ANU Quantum Random Number Generator API.
    
    Args:
        count: Number of values needed
        use_complex: Whether to generate complex numbers
        
    Returns:
        Numpy array of random numbers between 0 and 1
    """
    try:
        # If we need complex numbers, we need twice as many random values
        api_count = count * 2 if use_complex else count
        API_KEY = os.getenv("ANU_QRNG_KEY")
        if not API_KEY:
            API_KEY = input("Enter your ANU QRNG API Key")
            if not API_KEY:
                raise RuntimeError("Missing ANU_QRNG_KEY environment variable")
        # Make request to the ANU Quantum RNG API with API key
        url = f"https://api.quantumnumbers.anu.edu.au?length={api_count}&type=uint8"
        headers = {"x-api-key": API_KEY}
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Warning: Quantum RNG API returned status {response.status_code}. Falling back to numpy RNG.")
            return None
        
        # Get the data
        data = response.json()
        

        
        # Properly extract the data based on API format
        if 'data' in data:
            # For uint8 type, data is already integers 0-255
            random_values = np.array(data['data'], dtype=np.float64) / 255.0
        else:
            print("Warning: Unexpected API response format. Falling back to numpy RNG.")
            return None
            
        if use_complex:
            # Split the array into real and imaginary parts
            real_parts = random_values[:count]
            imag_parts = random_values[count:]
            # Create complex numbers
            return real_parts + 1j * imag_parts
        else:
            return random_values
    except Exception as e:
        print(f"Warning: Error fetching quantum random numbers: {e}. Falling back to numpy RNG.")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# STANDARD MATRIX MULTIPLICATION
# =============================================================================

def standard_multiply(A, B):
    """Standard matrix multiplication algorithm.
    
    Args:
        A: First matrix (n×n)
        B: Second matrix (n×n)
    
    Returns:
        C: Result of A×B
    """
    n = A.shape[0]
    C = np.zeros((n, n), dtype=A.dtype)
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
                
    return C

# =============================================================================
# STRASSEN'S ALGORITHM
# =============================================================================

def strassen_multiply(A, B):
    """Strassen's matrix multiplication algorithm.
    
    This implementation works for matrices of size 2^n × 2^n.
    For 4×4 matrices, it uses 49 scalar multiplications.
    
    Args:
        A: First matrix (n×n, where n is a power of 2)
        B: Second matrix (n×n, where n is a power of 2)
    
    Returns:
        C: Result of A×B
    """
    n = A.shape[0]
    
    # Base case: 1×1 matrix
    if n == 1:
        return A * B
    
    # Split matrices into quadrants
    mid = n // 2
    
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]
    
    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]
    
    # Compute 7 products (these are the scalar multiplications)
    P1 = strassen_multiply(A11 + A22, B11 + B22)
    P2 = strassen_multiply(A21 + A22, B11)
    P3 = strassen_multiply(A11, B12 - B22)
    P4 = strassen_multiply(A22, B21 - B11)
    P5 = strassen_multiply(A11 + A12, B22)
    P6 = strassen_multiply(A21 - A11, B11 + B12)
    P7 = strassen_multiply(A12 - A22, B21 + B22)
    
    # Combine the products to form the quadrants of the result
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    
    # Combine the quadrants to form the result
    C = np.zeros((n, n), dtype=A.dtype)
    C[:mid, :mid] = C11
    C[:mid, mid:] = C12
    C[mid:, :mid] = C21
    C[mid:, mid:] = C22
    
    return C

# For 4×4 case specifically, we can optimize by directly using Strassen's algorithm once
def strassen_4x4(A, B):
    """Strassen's algorithm for 4×4 matrices.
    
    This uses exactly 49 scalar multiplications.
    
    Args:
        A: First 4×4 matrix
        B: Second 4×4 matrix
    
    Returns:
        C: Result of A×B
    """
    # Split matrices into 2×2 blocks
    A11 = A[:2, :2]
    A12 = A[:2, 2:]
    A21 = A[2:, :2]
    A22 = A[2:, 2:]
    
    B11 = B[:2, :2]
    B12 = B[:2, 2:]
    B21 = B[2:, :2]
    B22 = B[2:, 2:]
    
    # Compute 7 products using standard multiplication (7 * 7 = 49 scalar multiplications)
    P1 = strassen_multiply(A11 + A22, B11 + B22)
    P2 = strassen_multiply(A21 + A22, B11)
    P3 = strassen_multiply(A11, B12 - B22)
    P4 = strassen_multiply(A22, B21 - B11)
    P5 = strassen_multiply(A11 + A12, B22)
    P6 = strassen_multiply(A21 - A11, B11 + B12)
    P7 = strassen_multiply(A12 - A22, B21 + B22)
    
    # Combine the products
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    
    # Combine the quadrants
    C = np.zeros((4, 4), dtype=A.dtype)
    C[:2, :2] = C11
    C[:2, 2:] = C12
    C[2:, :2] = C21
    C[2:, 2:] = C22
    
    return C

# =============================================================================
# ALPHAEVOLVE'S ALGORITHM 
# =============================================================================

def alphaevolve_4x4(A, B):
    """
    AlphaEvolve's optimized algorithm for 4×4 matrices.
    Uses exactly 48 scalar multiplications.
    """
    # Initialize the result matrix
    C = np.zeros((4, 4), dtype=complex)

    # Linear combinations of elements from A
    a0 = (0.5+0.5j)*A[0,0] + (0.5+0.5j)*A[0,1] + (0.5+-0.5j)*A[1,0] + (0.5+-0.5j)*A[1,1] + (0.5+-0.5j)*A[2,0] + (0.5+-0.5j)*A[2,1] + (0.5+-0.5j)*A[3,0] + (0.5+-0.5j)*A[3,1]
    a1 = (0.5+0.5j)*A[0,0] + (-0.5+0.5j)*A[0,3] + (0.5+0.5j)*A[1,0] + (-0.5+0.5j)*A[1,3] + (-0.5+-0.5j)*A[2,0] + (0.5+-0.5j)*A[2,3] + (0.5+-0.5j)*A[3,0] + (0.5+0.5j)*A[3,3]
    a2 = -0.5*A[0,1] + 0.5*A[0,2] + -0.5j*A[1,1] + 0.5j*A[1,2] + 0.5j*A[2,1] + -0.5j*A[2,2] + -0.5j*A[3,1] + 0.5j*A[3,2]
    a3 = -0.5j*A[0,0] + -0.5*A[0,1] + 0.5*A[0,2] + -0.5*A[0,3] + 0.5j*A[1,0] + -0.5*A[1,1] + 0.5*A[1,2] + 0.5*A[1,3] + -0.5j*A[2,0] + -0.5*A[2,1] + 0.5*A[2,2] + -0.5*A[2,3] + -0.5*A[3,0] + -0.5j*A[3,1] + 0.5j*A[3,2] + 0.5j*A[3,3]
    a4 = (0.5+0.5j)*A[0,0] + (-0.5+-0.5j)*A[0,1] + (-0.5+0.5j)*A[1,0] + (0.5+-0.5j)*A[1,1] + (-0.5+0.5j)*A[2,0] + (0.5+-0.5j)*A[2,1] + (0.5+-0.5j)*A[3,0] + (-0.5+0.5j)*A[3,1]
    a5 = (0.5+-0.5j)*A[0,2] + (-0.5+-0.5j)*A[0,3] + (0.5+-0.5j)*A[1,2] + (-0.5+-0.5j)*A[1,3] + (-0.5+0.5j)*A[2,2] + (0.5+0.5j)*A[2,3] + (-0.5+-0.5j)*A[3,2] + (-0.5+0.5j)*A[3,3]
    a6 = 0.5j*A[0,0] + 0.5*A[0,3] + -0.5*A[1,0] + 0.5j*A[1,3] + 0.5*A[2,0] + -0.5j*A[2,3] + -0.5*A[3,0] + 0.5j*A[3,3]
    a7 = (0.5+0.5j)*A[0,0] + (-0.5+-0.5j)*A[0,1] + (-0.5+-0.5j)*A[1,0] + (0.5+0.5j)*A[1,1] + (-0.5+-0.5j)*A[2,0] + (0.5+0.5j)*A[2,1] + (-0.5+0.5j)*A[3,0] + (0.5+-0.5j)*A[3,1]
    a8 = -0.5j*A[0,0] + -0.5j*A[0,1] + -0.5*A[0,2] + -0.5j*A[0,3] + 0.5*A[1,0] + 0.5*A[1,1] + -0.5j*A[1,2] + 0.5*A[1,3] + -0.5*A[2,0] + -0.5*A[2,1] + -0.5j*A[2,2] + 0.5*A[2,3] + 0.5*A[3,0] + 0.5*A[3,1] + 0.5j*A[3,2] + -0.5*A[3,3]
    a9 = (-0.5+0.5j)*A[0,0] + (-0.5+-0.5j)*A[0,3] + (0.5+0.5j)*A[1,0] + (-0.5+0.5j)*A[1,3] + (-0.5+-0.5j)*A[2,0] + (0.5+-0.5j)*A[2,3] + (-0.5+-0.5j)*A[3,0] + (0.5+-0.5j)*A[3,3]
    a10 = (-0.5+0.5j)*A[0,0] + (0.5+-0.5j)*A[0,1] + (-0.5+0.5j)*A[1,0] + (0.5+-0.5j)*A[1,1] + (0.5+-0.5j)*A[2,0] + (-0.5+0.5j)*A[2,1] + (0.5+0.5j)*A[3,0] + (-0.5+-0.5j)*A[3,1]
    a11 = 0.5*A[0,0] + 0.5*A[0,1] + -0.5j*A[0,2] + -0.5*A[0,3] + -0.5*A[1,0] + -0.5*A[1,1] + 0.5j*A[1,2] + 0.5*A[1,3] + 0.5*A[2,0] + 0.5*A[2,1] + 0.5j*A[2,2] + 0.5*A[2,3] + -0.5j*A[3,0] + -0.5j*A[3,1] + 0.5*A[3,2] + -0.5j*A[3,3]
    a12 = (0.5+0.5j)*A[0,1] + (-0.5+-0.5j)*A[0,2] + (-0.5+0.5j)*A[1,1] + (0.5+-0.5j)*A[1,2] + (-0.5+0.5j)*A[2,1] + (0.5+-0.5j)*A[2,2] + (0.5+-0.5j)*A[3,1] + (-0.5+0.5j)*A[3,2]
    a13 = (0.5+-0.5j)*A[0,1] + (-0.5+0.5j)*A[0,2] + (0.5+-0.5j)*A[1,1] + (-0.5+0.5j)*A[1,2] + (0.5+-0.5j)*A[2,1] + (-0.5+0.5j)*A[2,2] + (0.5+0.5j)*A[3,1] + (-0.5+-0.5j)*A[3,2]
    a14 = 0.5j*A[0,0] + -0.5*A[0,1] + 0.5*A[0,2] + -0.5*A[0,3] + 0.5*A[1,0] + -0.5j*A[1,1] + 0.5j*A[1,2] + 0.5j*A[1,3] + 0.5*A[2,0] + 0.5j*A[2,1] + -0.5j*A[2,2] + 0.5j*A[2,3] + 0.5*A[3,0] + -0.5j*A[3,1] + 0.5j*A[3,2] + 0.5j*A[3,3]
    a15 = (-0.5+0.5j)*A[0,2] + (0.5+0.5j)*A[0,3] + (0.5+-0.5j)*A[1,2] + (-0.5+-0.5j)*A[1,3] + (0.5+-0.5j)*A[2,2] + (-0.5+-0.5j)*A[2,3] + (-0.5+-0.5j)*A[3,2] + (-0.5+0.5j)*A[3,3]
    a16 = -0.5*A[0,0] + 0.5j*A[0,1] + 0.5j*A[0,2] + -0.5j*A[0,3] + -0.5*A[1,0] + -0.5j*A[1,1] + -0.5j*A[1,2] + -0.5j*A[1,3] + -0.5*A[2,0] + 0.5j*A[2,1] + 0.5j*A[2,2] + -0.5j*A[2,3] + -0.5j*A[3,0] + 0.5*A[3,1] + 0.5*A[3,2] + 0.5*A[3,3]
    a17 = (0.5+0.5j)*A[0,0] + (0.5+0.5j)*A[0,1] + (0.5+0.5j)*A[1,0] + (0.5+0.5j)*A[1,1] + (0.5+0.5j)*A[2,0] + (0.5+0.5j)*A[2,1] + (-0.5+0.5j)*A[3,0] + (-0.5+0.5j)*A[3,1]
    a18 = 0.5j*A[0,0] + 0.5j*A[0,1] + -0.5*A[0,2] + 0.5j*A[0,3] + 0.5j*A[1,0] + 0.5j*A[1,1] + -0.5*A[1,2] + 0.5j*A[1,3] + 0.5j*A[2,0] + 0.5j*A[2,1] + 0.5*A[2,2] + -0.5j*A[2,3] + -0.5*A[3,0] + -0.5*A[3,1] + 0.5j*A[3,2] + 0.5*A[3,3]
    a19 = (0.5+-0.5j)*A[0,2] + (0.5+0.5j)*A[0,3] + (0.5+-0.5j)*A[1,2] + (0.5+0.5j)*A[1,3] + (0.5+-0.5j)*A[2,2] + (0.5+0.5j)*A[2,3] + (0.5+0.5j)*A[3,2] + (-0.5+0.5j)*A[3,3]
    a20 = (0.5+0.5j)*A[0,1] + (-0.5+-0.5j)*A[0,2] + (0.5+0.5j)*A[1,1] + (-0.5+-0.5j)*A[1,2] + (-0.5+-0.5j)*A[2,1] + (0.5+0.5j)*A[2,2] + (0.5+-0.5j)*A[3,1] + (-0.5+0.5j)*A[3,2]
    a21 = 0.5j*A[0,0] + -0.5j*A[0,1] + -0.5*A[0,2] + -0.5j*A[0,3] + -0.5j*A[1,0] + 0.5j*A[1,1] + 0.5*A[1,2] + 0.5j*A[1,3] + -0.5j*A[2,0] + 0.5j*A[2,1] + -0.5*A[2,2] + -0.5j*A[2,3] + -0.5*A[3,0] + 0.5*A[3,1] + 0.5j*A[3,2] + -0.5*A[3,3]
    a22 = (-0.5+-0.5j)*A[0,0] + (-0.5+0.5j)*A[0,3] + (0.5+-0.5j)*A[1,0] + (-0.5+-0.5j)*A[1,3] + (0.5+-0.5j)*A[2,0] + (-0.5+-0.5j)*A[2,3] + (-0.5+0.5j)*A[3,0] + (0.5+0.5j)*A[3,3]
    a23 = (-0.5+-0.5j)*A[0,2] + (0.5+-0.5j)*A[0,3] + (0.5+-0.5j)*A[1,2] + (0.5+0.5j)*A[1,3] + (0.5+-0.5j)*A[2,2] + (0.5+0.5j)*A[2,3] + (-0.5+0.5j)*A[3,2] + (-0.5+-0.5j)*A[3,3]
    a24 = -0.5*A[0,0] + 0.5*A[0,1] + -0.5j*A[0,2] + -0.5*A[0,3] + -0.5j*A[1,0] + 0.5j*A[1,1] + 0.5*A[1,2] + -0.5j*A[1,3] + -0.5j*A[2,0] + 0.5j*A[2,1] + -0.5*A[2,2] + 0.5j*A[2,3] + 0.5j*A[3,0] + -0.5j*A[3,1] + 0.5*A[3,2] + -0.5j*A[3,3]
    a25 = (0.5+-0.5j)*A[0,2] + (0.5+0.5j)*A[0,3] + (-0.5+-0.5j)*A[1,2] + (0.5+-0.5j)*A[1,3] + (0.5+0.5j)*A[2,2] + (-0.5+0.5j)*A[2,3] + (0.5+0.5j)*A[3,2] + (-0.5+0.5j)*A[3,3]
    a26 = (0.5+0.5j)*A[0,1] + (0.5+0.5j)*A[0,2] + (-0.5+-0.5j)*A[1,1] + (-0.5+-0.5j)*A[1,2] + (0.5+0.5j)*A[2,1] + (0.5+0.5j)*A[2,2] + (0.5+-0.5j)*A[3,1] + (0.5+-0.5j)*A[3,2]
    a27 = -0.5j*A[0,0] + -0.5j*A[0,1] + 0.5*A[0,2] + 0.5j*A[0,3] + -0.5*A[1,0] + -0.5*A[1,1] + -0.5j*A[1,2] + 0.5*A[1,3] + -0.5*A[2,0] + -0.5*A[2,1] + 0.5j*A[2,2] + -0.5*A[2,3] + -0.5*A[3,0] + -0.5*A[3,1] + 0.5j*A[3,2] + -0.5*A[3,3]
    a28 = (-0.5+0.5j)*A[0,0] + (-0.5+0.5j)*A[0,1] + (-0.5+-0.5j)*A[1,0] + (-0.5+-0.5j)*A[1,1] + (0.5+0.5j)*A[2,0] + (0.5+0.5j)*A[2,1] + (-0.5+-0.5j)*A[3,0] + (-0.5+-0.5j)*A[3,1]
    a29 = (0.5+0.5j)*A[0,0] + (0.5+-0.5j)*A[0,3] + (-0.5+-0.5j)*A[1,0] + (-0.5+0.5j)*A[1,3] + (0.5+0.5j)*A[2,0] + (0.5+-0.5j)*A[2,3] + (0.5+-0.5j)*A[3,0] + (-0.5+-0.5j)*A[3,3]
    a30 = (0.5+0.5j)*A[0,1] + (0.5+0.5j)*A[0,2] + (-0.5+-0.5j)*A[1,1] + (-0.5+-0.5j)*A[1,2] + (-0.5+-0.5j)*A[2,1] + (-0.5+-0.5j)*A[2,2] + (-0.5+0.5j)*A[3,1] + (-0.5+0.5j)*A[3,2]
    a31 = 0.5*A[0,0] + -0.5*A[0,1] + -0.5j*A[0,2] + 0.5*A[0,3] + 0.5*A[1,0] + -0.5*A[1,1] + -0.5j*A[1,2] + 0.5*A[1,3] + -0.5*A[2,0] + 0.5*A[2,1] + -0.5j*A[2,2] + 0.5*A[2,3] + -0.5j*A[3,0] + 0.5j*A[3,1] + 0.5*A[3,2] + 0.5j*A[3,3]
    a32 = (0.5+0.5j)*A[0,2] + (0.5+-0.5j)*A[0,3] + (-0.5+0.5j)*A[1,2] + (0.5+0.5j)*A[1,3] + (0.5+-0.5j)*A[2,2] + (-0.5+-0.5j)*A[2,3] + (-0.5+0.5j)*A[3,2] + (0.5+0.5j)*A[3,3]
    a33 = 0.5*A[0,0] + 0.5j*A[0,1] + -0.5j*A[0,2] + -0.5j*A[0,3] + -0.5*A[1,0] + 0.5j*A[1,1] + -0.5j*A[1,2] + 0.5j*A[1,3] + -0.5*A[2,0] + -0.5j*A[2,1] + 0.5j*A[2,2] + 0.5j*A[2,3] + 0.5j*A[3,0] + 0.5*A[3,1] + -0.5*A[3,2] + 0.5*A[3,3]
    a34 = -0.5j*A[0,0] + 0.5j*A[0,1] + -0.5*A[0,2] + 0.5j*A[0,3] + -0.5*A[1,0] + 0.5*A[1,1] + 0.5j*A[1,2] + 0.5*A[1,3] + 0.5*A[2,0] + -0.5*A[2,1] + 0.5j*A[2,2] + 0.5*A[2,3] + 0.5*A[3,0] + -0.5*A[3,1] + 0.5j*A[3,2] + 0.5*A[3,3]
    a35 = (0.5+-0.5j)*A[0,2] + (0.5+0.5j)*A[0,3] + (-0.5+0.5j)*A[1,2] + (-0.5+-0.5j)*A[1,3] + (0.5+-0.5j)*A[2,2] + (0.5+0.5j)*A[2,3] + (-0.5+-0.5j)*A[3,2] + (0.5+-0.5j)*A[3,3]
    a36 = (-0.5+-0.5j)*A[0,1] + (-0.5+-0.5j)*A[0,2] + (-0.5+0.5j)*A[1,1] + (-0.5+0.5j)*A[1,2] + (0.5+-0.5j)*A[2,1] + (0.5+-0.5j)*A[2,2] + (0.5+-0.5j)*A[3,1] + (0.5+-0.5j)*A[3,2]
    a37 = 0.5*A[0,0] + -0.5j*A[0,1] + -0.5j*A[0,2] + -0.5j*A[0,3] + 0.5j*A[1,0] + -0.5*A[1,1] + -0.5*A[1,2] + 0.5*A[1,3] + 0.5j*A[2,0] + 0.5*A[2,1] + 0.5*A[2,2] + 0.5*A[2,3] + -0.5j*A[3,0] + 0.5*A[3,1] + 0.5*A[3,2] + -0.5*A[3,3]
    a38 = (0.5+-0.5j)*A[0,1] + (0.5+-0.5j)*A[0,2] + (-0.5+-0.5j)*A[1,1] + (-0.5+-0.5j)*A[1,2] + (-0.5+-0.5j)*A[2,1] + (-0.5+-0.5j)*A[2,2] + (-0.5+-0.5j)*A[3,1] + (-0.5+-0.5j)*A[3,2]
    a39 = -0.5*A[0,0] + -0.5j*A[0,1] + -0.5j*A[0,2] + -0.5j*A[0,3] + -0.5*A[1,0] + 0.5j*A[1,1] + 0.5j*A[1,2] + -0.5j*A[1,3] + 0.5*A[2,0] + 0.5j*A[2,1] + 0.5j*A[2,2] + 0.5j*A[2,3] + 0.5j*A[3,0] + 0.5*A[3,1] + 0.5*A[3,2] + -0.5*A[3,3]
    a40 = (-0.5+-0.5j)*A[0,0] + (-0.5+-0.5j)*A[0,1] + (0.5+0.5j)*A[1,0] + (0.5+0.5j)*A[1,1] + (-0.5+-0.5j)*A[2,0] + (-0.5+-0.5j)*A[2,1] + (-0.5+0.5j)*A[3,0] + (-0.5+0.5j)*A[3,1]
    a41 = (0.5+-0.5j)*A[0,0] + (-0.5+-0.5j)*A[0,3] + (-0.5+0.5j)*A[1,0] + (0.5+0.5j)*A[1,3] + (-0.5+0.5j)*A[2,0] + (0.5+0.5j)*A[2,3] + (0.5+0.5j)*A[3,0] + (0.5+-0.5j)*A[3,3]
    a42 = (0.5+0.5j)*A[0,0] + (-0.5+0.5j)*A[0,3] + (0.5+-0.5j)*A[1,0] + (0.5+0.5j)*A[1,3] + (0.5+-0.5j)*A[2,0] + (0.5+0.5j)*A[2,3] + (0.5+-0.5j)*A[3,0] + (0.5+0.5j)*A[3,3]
    a43 = 0.5j*A[0,0] + 0.5*A[0,1] + -0.5*A[0,2] + -0.5*A[0,3] + 0.5*A[1,0] + 0.5j*A[1,1] + -0.5j*A[1,2] + 0.5j*A[1,3] + -0.5*A[2,0] + 0.5j*A[2,1] + -0.5j*A[2,2] + -0.5j*A[2,3] + -0.5*A[3,0] + -0.5j*A[3,1] + 0.5j*A[3,2] + -0.5j*A[3,3]
    a44 = (0.5+-0.5j)*A[0,2] + (-0.5+-0.5j)*A[0,3] + (-0.5+-0.5j)*A[1,2] + (-0.5+0.5j)*A[1,3] + (-0.5+-0.5j)*A[2,2] + (-0.5+0.5j)*A[2,3] + (-0.5+-0.5j)*A[3,2] + (-0.5+0.5j)*A[3,3]
    a45 = (-0.5+0.5j)*A[0,0] + (0.5+-0.5j)*A[0,1] + (0.5+0.5j)*A[1,0] + (-0.5+-0.5j)*A[1,1] + (-0.5+-0.5j)*A[2,0] + (0.5+0.5j)*A[2,1] + (-0.5+-0.5j)*A[3,0] + (0.5+0.5j)*A[3,1]
    a46 = (0.5+-0.5j)*A[0,0] + (0.5+0.5j)*A[0,3] + (0.5+-0.5j)*A[1,0] + (0.5+0.5j)*A[1,3] + (0.5+-0.5j)*A[2,0] + (0.5+0.5j)*A[2,3] + (0.5+0.5j)*A[3,0] + (-0.5+0.5j)*A[3,3]
    a47 = 0.5*A[0,0] + 0.5j*A[0,1] + 0.5j*A[0,2] + -0.5j*A[0,3] + 0.5j*A[1,0] + 0.5*A[1,1] + 0.5*A[1,2] + 0.5*A[1,3] + -0.5j*A[2,0] + 0.5*A[2,1] + 0.5*A[2,2] + -0.5*A[2,3] + 0.5j*A[3,0] + 0.5*A[3,1] + 0.5*A[3,2] + 0.5*A[3,3]

    # Linear combinations of elements from B
    b0 = -0.5*B[0,0] + -0.5*B[1,0] + 0.5*B[2,0] + -0.5j*B[3,0]
    b1 = 0.5j*B[0,1] + 0.5j*B[0,3] + 0.5j*B[1,1] + 0.5j*B[1,3] + 0.5j*B[2,1] + 0.5j*B[2,3] + 0.5*B[3,1] + 0.5*B[3,3]
    b2 = (0.5+0.5j)*B[0,1] + (-0.5+-0.5j)*B[1,1] + (0.5+0.5j)*B[2,1] + (0.5+-0.5j)*B[3,1]
    b3 = -0.5j*B[0,0] + 0.5j*B[0,2] + -0.5j*B[1,1] + -0.5j*B[1,2] + 0.5j*B[2,1] + 0.5j*B[2,2] + 0.5*B[3,0] + -0.5*B[3,2]
    b4 = -0.5*B[0,0] + 0.5*B[0,2] + 0.5*B[0,3] + 0.5*B[1,0] + -0.5*B[1,2] + -0.5*B[1,3] + 0.5*B[2,0] + -0.5*B[2,2] + -0.5*B[2,3] + 0.5j*B[3,0] + -0.5j*B[3,2] + -0.5j*B[3,3]
    b5 = 0.5*B[0,1] + 0.5*B[0,3] + 0.5*B[1,1] + 0.5*B[1,3] + 0.5*B[2,1] + 0.5*B[2,3] + 0.5j*B[3,1] + 0.5j*B[3,3]
    b6 = (-0.5+-0.5j)*B[0,1] + (0.5+0.5j)*B[1,1] + (0.5+0.5j)*B[2,1] + (0.5+-0.5j)*B[3,1]
    b7 = -0.5*B[0,0] + 0.5*B[0,3] + 0.5*B[1,0] + -0.5*B[1,3] + -0.5*B[2,0] + 0.5*B[2,3] + 0.5j*B[3,0] + -0.5j*B[3,3]
    b8 = 0.5*B[0,0] + -0.5*B[0,2] + -0.5*B[0,3] + 0.5*B[1,0] + -0.5*B[1,2] + -0.5*B[1,3] + 0.5*B[2,1] + -0.5j*B[3,1]
    b9 = 0.5j*B[0,1] + 0.5j*B[0,2] + 0.5j*B[0,3] + 0.5j*B[1,1] + 0.5j*B[1,2] + 0.5j*B[1,3] + -0.5j*B[2,1] + -0.5j*B[2,2] + -0.5j*B[2,3] + 0.5*B[3,1] + 0.5*B[3,2] + 0.5*B[3,3]
    b10 = 0.5j*B[0,1] + 0.5j*B[0,3] + -0.5j*B[1,1] + -0.5j*B[1,3] + -0.5j*B[2,1] + -0.5j*B[2,3] + -0.5*B[3,1] + -0.5*B[3,3]
    b11 = -0.5j*B[0,0] + 0.5j*B[0,3] + -0.5j*B[1,0] + 0.5j*B[1,3] + 0.5j*B[2,1] + 0.5j*B[2,2] + -0.5*B[3,1] + -0.5*B[3,2]
    b12 = -0.5*B[0,0] + 0.5*B[0,2] + 0.5*B[0,3] + -0.5*B[1,0] + 0.5*B[1,2] + 0.5*B[1,3] + 0.5*B[2,0] + -0.5*B[2,2] + -0.5*B[2,3] + 0.5j*B[3,0] + -0.5j*B[3,2] + -0.5j*B[3,3]
    b13 = 0.5j*B[0,0] + -0.5j*B[0,2] + -0.5j*B[1,0] + 0.5j*B[1,2] + 0.5j*B[2,0] + -0.5j*B[2,2] + -0.5*B[3,0] + 0.5*B[3,2]
    b14 = -0.5*B[0,1] + -0.5*B[1,0] + 0.5*B[2,0] + 0.5j*B[3,1]
    b15 = 0.5j*B[0,0] + -0.5j*B[0,3] + 0.5j*B[1,0] + -0.5j*B[1,3] + -0.5j*B[2,0] + 0.5j*B[2,3] + 0.5*B[3,0] + -0.5*B[3,3]
    b16 = 0.5*B[0,1] + 0.5*B[0,2] + 0.5*B[1,0] + -0.5*B[1,2] + 0.5*B[2,0] + -0.5*B[2,2] + -0.5j*B[3,1] + -0.5j*B[3,2]
    b17 = -0.5j*B[0,0] + 0.5j*B[0,2] + -0.5j*B[1,0] + 0.5j*B[1,2] + -0.5j*B[2,0] + 0.5j*B[2,2] + 0.5*B[3,0] + -0.5*B[3,2]
    b18 = -0.5j*B[0,1] + -0.5j*B[0,3] + -0.5j*B[1,1] + -0.5j*B[1,3] + -0.5j*B[2,0] + 0.5j*B[2,2] + 0.5*B[3,0] + -0.5*B[3,2]
    b19 = -0.5j*B[0,0] + 0.5j*B[0,2] + 0.5j*B[1,0] + -0.5j*B[1,2] + 0.5j*B[2,0] + -0.5j*B[2,2] + 0.5*B[3,0] + -0.5*B[3,2]
    b20 = -0.5j*B[0,1] + -0.5j*B[0,3] + -0.5j*B[1,1] + -0.5j*B[1,3] + 0.5j*B[2,1] + 0.5j*B[2,3] + 0.5*B[3,1] + 0.5*B[3,3]
    b21 = -0.5*B[0,1] + -0.5*B[0,2] + 0.5*B[1,1] + 0.5*B[1,2] + -0.5*B[2,0] + 0.5*B[2,3] + 0.5j*B[3,0] + -0.5j*B[3,3]
    b22 = -0.5j*B[0,0] + 0.5j*B[0,2] + 0.5j*B[0,3] + -0.5j*B[1,0] + 0.5j*B[1,2] + 0.5j*B[1,3] + -0.5j*B[2,0] + 0.5j*B[2,2] + 0.5j*B[2,3] + 0.5*B[3,0] + -0.5*B[3,2] + -0.5*B[3,3]
    b23 = -0.5*B[0,0] + 0.5*B[0,2] + 0.5*B[0,3] + -0.5*B[1,0] + 0.5*B[1,2] + 0.5*B[1,3] + -0.5*B[2,0] + 0.5*B[2,2] + 0.5*B[2,3] + 0.5j*B[3,0] + -0.5j*B[3,2] + -0.5j*B[3,3]
    b24 = 0.5j*B[0,1] + -0.5j*B[1,1] + -0.5j*B[2,0] + 0.5j*B[2,2] + 0.5j*B[2,3] + 0.5*B[3,0] + -0.5*B[3,2] + -0.5*B[3,3]
    b25 = 0.5j*B[0,1] + 0.5j*B[0,2] + 0.5j*B[0,3] + 0.5j*B[1,1] + 0.5j*B[1,2] + 0.5j*B[1,3] + -0.5j*B[2,1] + -0.5j*B[2,2] + -0.5j*B[2,3] + -0.5*B[3,1] + -0.5*B[3,2] + -0.5*B[3,3]
    b26 = 0.5*B[0,1] + 0.5*B[0,2] + -0.5*B[1,1] + -0.5*B[1,2] + -0.5*B[2,1] + -0.5*B[2,2] + -0.5j*B[3,1] + -0.5j*B[3,2]
    b27 = 0.5j*B[0,1] + 0.5j*B[0,2] + 0.5j*B[0,3] + 0.5j*B[1,1] + 0.5j*B[1,2] + 0.5j*B[1,3] + -0.5j*B[2,0] + -0.5*B[3,0]
    b28 = 0.5*B[0,1] + 0.5*B[1,1] + 0.5*B[2,1] + -0.5j*B[3,1]
    b29 = 0.5j*B[0,1] + 0.5j*B[0,2] + -0.5j*B[1,1] + -0.5j*B[1,2] + 0.5j*B[2,1] + 0.5j*B[2,2] + -0.5*B[3,1] + -0.5*B[3,2]
    b30 = -0.5*B[0,0] + 0.5*B[0,3] + -0.5*B[1,0] + 0.5*B[1,3] + -0.5*B[2,0] + 0.5*B[2,3] + 0.5j*B[3,0] + -0.5j*B[3,3]
    b31 = 0.5*B[0,0] + -0.5*B[0,2] + -0.5*B[1,0] + 0.5*B[1,2] + -0.5*B[2,1] + -0.5*B[2,3] + 0.5j*B[3,1] + 0.5j*B[3,3]
    b32 = 0.5j*B[0,1] + -0.5j*B[1,1] + -0.5j*B[2,1] + 0.5*B[3,1]
    b33 = -0.5*B[0,1] + -0.5*B[0,3] + 0.5*B[1,0] + -0.5*B[1,3] + -0.5*B[2,0] + 0.5*B[2,3] + -0.5j*B[3,1] + -0.5j*B[3,3]
    b34 = 0.5j*B[0,0] + -0.5j*B[1,0] + 0.5j*B[2,1] + 0.5j*B[2,2] + 0.5j*B[2,3] + -0.5*B[3,1] + -0.5*B[3,2] + -0.5*B[3,3]
    b35 = -0.5j*B[0,1] + -0.5j*B[0,2] + 0.5j*B[1,1] + 0.5j*B[1,2] + -0.5j*B[2,1] + -0.5j*B[2,2] + -0.5*B[3,1] + -0.5*B[3,2]
    b36 = -0.5*B[0,1] + -0.5*B[0,2] + -0.5*B[0,3] + -0.5*B[1,1] + -0.5*B[1,2] + -0.5*B[1,3] + -0.5*B[2,1] + -0.5*B[2,2] + -0.5*B[2,3] + -0.5j*B[3,1] + -0.5j*B[3,2] + -0.5j*B[3,3]
    b37 = 0.5j*B[0,1] + 0.5j*B[0,2] + 0.5j*B[0,3] + -0.5j*B[1,0] + 0.5j*B[1,2] + 0.5j*B[1,3] + -0.5j*B[2,0] + 0.5j*B[2,2] + 0.5j*B[2,3] + -0.5*B[3,1] + -0.5*B[3,2] + -0.5*B[3,3]
    b38 = 0.5j*B[0,0] + -0.5j*B[1,0] + -0.5j*B[2,0] + -0.5*B[3,0]
    b39 = -0.5j*B[0,0] + 0.5j*B[0,3] + 0.5j*B[1,1] + 0.5j*B[1,3] + 0.5j*B[2,1] + 0.5j*B[2,3] + -0.5*B[3,0] + 0.5*B[3,3]
    b40 = 0.5j*B[0,1] + 0.5j*B[0,2] + 0.5j*B[1,1] + 0.5j*B[1,2] + -0.5j*B[2,1] + -0.5j*B[2,2] + 0.5*B[3,1] + 0.5*B[3,2]
    b41 = 0.5*B[0,0] + -0.5*B[0,3] + 0.5*B[1,0] + -0.5*B[1,3] + -0.5*B[2,0] + 0.5*B[2,3] + 0.5j*B[3,0] + -0.5j*B[3,3]
    b42 = 0.5j*B[0,0] + -0.5j*B[1,0] + 0.5j*B[2,0] + 0.5*B[3,0]
    b43 = 0.5*B[0,0] + -0.5*B[0,2] + -0.5*B[0,3] + -0.5*B[1,1] + -0.5*B[1,2] + -0.5*B[1,3] + 0.5*B[2,1] + 0.5*B[2,2] + 0.5*B[2,3] + -0.5j*B[3,0] + 0.5j*B[3,2] + 0.5j*B[3,3]
    b44 = -0.5j*B[0,0] + 0.5j*B[1,0] + -0.5j*B[2,0] + 0.5*B[3,0]
    b45 = -0.5j*B[0,1] + -0.5j*B[0,2] + -0.5j*B[0,3] + 0.5j*B[1,1] + 0.5j*B[1,2] + 0.5j*B[1,3] + -0.5j*B[2,1] + -0.5j*B[2,2] + -0.5j*B[2,3] + 0.5*B[3,1] + 0.5*B[3,2] + 0.5*B[3,3]
    b46 = -0.5*B[0,0] + 0.5*B[0,2] + 0.5*B[1,0] + -0.5*B[1,2] + 0.5*B[2,0] + -0.5*B[2,2] + 0.5j*B[3,0] + -0.5j*B[3,2]
    b47 = 0.5*B[0,0] + 0.5*B[1,1] + 0.5*B[2,1] + 0.5j*B[3,0]

    # Perform the 48 multiplications
    m0 = a0 * b0
    m1 = a1 * b1
    m2 = a2 * b2
    m3 = a3 * b3
    m4 = a4 * b4
    m5 = a5 * b5
    m6 = a6 * b6
    m7 = a7 * b7
    m8 = a8 * b8
    m9 = a9 * b9
    m10 = a10 * b10
    m11 = a11 * b11
    m12 = a12 * b12
    m13 = a13 * b13
    m14 = a14 * b14
    m15 = a15 * b15
    m16 = a16 * b16
    m17 = a17 * b17
    m18 = a18 * b18
    m19 = a19 * b19
    m20 = a20 * b20
    m21 = a21 * b21
    m22 = a22 * b22
    m23 = a23 * b23
    m24 = a24 * b24
    m25 = a25 * b25
    m26 = a26 * b26
    m27 = a27 * b27
    m28 = a28 * b28
    m29 = a29 * b29
    m30 = a30 * b30
    m31 = a31 * b31
    m32 = a32 * b32
    m33 = a33 * b33
    m34 = a34 * b34
    m35 = a35 * b35
    m36 = a36 * b36
    m37 = a37 * b37
    m38 = a38 * b38
    m39 = a39 * b39
    m40 = a40 * b40
    m41 = a41 * b41
    m42 = a42 * b42
    m43 = a43 * b43
    m44 = a44 * b44
    m45 = a45 * b45
    m46 = a46 * b46
    m47 = a47 * b47

    # Construct the result matrix
    C[0,0] = 0.5j*m0 + -0.5j*m1 + -0.5*m5 + 0.5*m8 + 0.5j*m9 + (-0.5+0.5j)*m11 + 0.5*m14 + -0.5j*m15 + (-0.5+-0.5j)*m16 + 0.5j*m17 + (-0.5+-0.5j)*m18 + -0.5j*m24 + 0.5j*m26 + 0.5j*m27 + 0.5*m28 + 0.5j*m30 + -0.5j*m32 + 0.5*m34 + 0.5*m36 + -0.5j*m37 + -0.5*m38 + (0.5+-0.5j)*m39 + -0.5j*m40 + -0.5*m42 + -0.5*m43 + -0.5*m44 + -0.5j*m46 + 0.5*m47
    C[0,1] = -0.5j*m0 + 0.5*m2 + (-0.5+-0.5j)*m3 + 0.5*m5 + 0.5*m6 + -0.5*m8 + (0.5+-0.5j)*m11 + -0.5*m12 + 0.5j*m13 + 0.5j*m14 + 0.5j*m15 + -0.5j*m17 + (0.5+0.5j)*m18 + 0.5*m20 + -0.5*m22 + 0.5j*m24 + -0.5j*m27 + -0.5*m28 + -0.5j*m29 + 0.5j*m32 + (-0.5+-0.5j)*m33 + -0.5*m34 + -0.5*m37 + 0.5j*m40 + 0.5j*m41 + -0.5j*m43 + 0.5*m44 + -0.5j*m47
    C[0,2] = -0.5*m2 + 0.5*m3 + -0.5*m5 + -0.5j*m8 + 0.5j*m11 + 0.5*m12 + -0.5j*m13 + -0.5j*m14 + -0.5j*m15 + -0.5*m16 + -0.5*m18 + 0.5j*m19 + -0.5*m20 + 0.5j*m21 + -0.5*m23 + -0.5j*m24 + -0.5*m25 + 0.5j*m26 + 0.5*m27 + 0.5j*m30 + -0.5*m31 + -0.5j*m32 + 0.5*m33 + 0.5*m34 + 0.5j*m35 + 0.5*m36 + -0.5j*m37 + -0.5*m38 + -0.5j*m39 + 0.5j*m43 + -0.5*m44 + 0.5*m47
    C[0,3] = 0.5j*m0 + -0.5j*m1 + 0.5j*m3 + -0.5j*m4 + -0.5*m6 + 0.5*m7 + 0.5*m8 + 0.5j*m9 + -0.5*m10 + -0.5*m11 + 0.5*m14 + -0.5j*m16 + 0.5j*m17 + -0.5j*m18 + -0.5*m21 + 0.5*m22 + 0.5*m24 + 0.5j*m27 + 0.5*m28 + 0.5j*m29 + -0.5j*m31 + 0.5j*m33 + 0.5j*m34 + 0.5*m37 + 0.5*m39 + -0.5j*m40 + -0.5j*m41 + -0.5*m42 + -0.5*m43 + -0.5j*m45 + -0.5j*m46 + 0.5j*m47
    C[1,0] = -0.5*m0 + -0.5*m1 + -0.5*m5 + -0.5j*m8 + -0.5j*m9 + (0.5+-0.5j)*m11 + -0.5j*m14 + 0.5j*m15 + (-0.5+0.5j)*m16 + 0.5j*m17 + (-0.5+-0.5j)*m18 + -0.5*m24 + 0.5*m26 + -0.5*m27 + -0.5j*m28 + 0.5*m30 + -0.5*m32 + 0.5j*m34 + 0.5*m36 + -0.5*m37 + -0.5*m38 + (-0.5+-0.5j)*m39 + 0.5j*m40 + 0.5*m42 + 0.5j*m43 + -0.5j*m44 + -0.5*m46 + -0.5j*m47
    C[1,1] = 0.5*m0 + -0.5*m2 + (0.5+-0.5j)*m3 + 0.5*m5 + 0.5*m6 + 0.5j*m8 + (-0.5+0.5j)*m11 + 0.5*m12 + -0.5*m13 + -0.5*m14 + -0.5j*m15 + -0.5j*m17 + (0.5+0.5j)*m18 + 0.5j*m20 + -0.5*m22 + 0.5*m24 + 0.5*m27 + 0.5j*m28 + 0.5*m29 + 0.5*m32 + (0.5+-0.5j)*m33 + -0.5j*m34 + -0.5j*m37 + -0.5j*m40 + -0.5*m41 + 0.5*m43 + 0.5j*m44 + 0.5*m47
    C[1,2] = 0.5*m2 + -0.5*m3 + -0.5*m5 + -0.5*m8 + -0.5j*m11 + -0.5*m12 + 0.5*m13 + 0.5*m14 + 0.5j*m15 + -0.5*m16 + -0.5*m18 + 0.5j*m19 + -0.5j*m20 + -0.5j*m21 + 0.5j*m23 + -0.5*m24 + -0.5j*m25 + 0.5*m26 + 0.5j*m27 + 0.5*m30 + -0.5*m31 + -0.5*m32 + -0.5*m33 + 0.5j*m34 + -0.5j*m35 + 0.5*m36 + -0.5*m37 + -0.5*m38 + -0.5j*m39 + -0.5*m43 + -0.5j*m44 + -0.5j*m47
    C[1,3] = -0.5*m0 + -0.5*m1 + 0.5j*m3 + -0.5*m4 + -0.5*m6 + -0.5*m7 + -0.5j*m8 + -0.5j*m9 + -0.5*m10 + 0.5*m11 + -0.5j*m14 + 0.5j*m16 + 0.5j*m17 + -0.5j*m18 + 0.5*m21 + 0.5*m22 + -0.5j*m24 + -0.5*m27 + -0.5j*m28 + -0.5*m29 + -0.5j*m31 + 0.5j*m33 + -0.5*m34 + 0.5j*m37 + -0.5*m39 + 0.5j*m40 + 0.5*m41 + 0.5*m42 + 0.5j*m43 + 0.5*m45 + -0.5*m46 + -0.5*m47
    C[2,0] = -0.5j*m0 + 0.5j*m1 + 0.5j*m5 + -0.5j*m8 + 0.5*m9 + (0.5+0.5j)*m11 + 0.5j*m14 + -0.5*m15 + (-0.5+-0.5j)*m16 + 0.5*m17 + (-0.5+0.5j)*m18 + -0.5*m24 + 0.5j*m26 + 0.5*m27 + -0.5*m28 + -0.5j*m30 + -0.5j*m32 + -0.5j*m34 + -0.5j*m36 + -0.5*m37 + -0.5j*m38 + (-0.5+0.5j)*m39 + -0.5*m40 + -0.5j*m42 + 0.5j*m43 + -0.5*m44 + -0.5j*m46 + 0.5j*m47
    C[2,1] = 0.5j*m0 + 0.5j*m2 + (-0.5+-0.5j)*m3 + -0.5j*m5 + 0.5j*m6 + 0.5j*m8 + (-0.5+-0.5j)*m11 + 0.5j*m12 + 0.5j*m13 + -0.5*m14 + 0.5*m15 + -0.5*m17 + (0.5+-0.5j)*m18 + -0.5*m20 + 0.5j*m22 + 0.5*m24 + -0.5*m27 + 0.5*m28 + -0.5j*m29 + 0.5j*m32 + (0.5+0.5j)*m33 + 0.5j*m34 + 0.5j*m37 + 0.5*m40 + -0.5j*m41 + -0.5*m43 + 0.5*m44 + 0.5*m47
    C[2,2] = -0.5j*m2 + 0.5*m3 + 0.5j*m5 + 0.5*m8 + 0.5j*m11 + -0.5j*m12 + -0.5j*m13 + 0.5*m14 + -0.5*m15 + -0.5*m16 + -0.5*m18 + -0.5*m19 + 0.5*m20 + -0.5j*m21 + 0.5*m23 + -0.5*m24 + 0.5*m25 + 0.5j*m26 + 0.5j*m27 + -0.5j*m30 + 0.5*m31 + -0.5j*m32 + -0.5*m33 + -0.5j*m34 + -0.5*m35 + -0.5j*m36 + -0.5*m37 + -0.5j*m38 + 0.5j*m39 + 0.5*m43 + -0.5*m44 + 0.5j*m47
    C[2,3] = -0.5j*m0 + 0.5j*m1 + 0.5j*m3 + -0.5j*m4 + -0.5j*m6 + 0.5j*m7 + -0.5j*m8 + 0.5*m9 + -0.5j*m10 + 0.5*m11 + 0.5j*m14 + -0.5j*m16 + 0.5*m17 + 0.5j*m18 + -0.5*m21 + -0.5j*m22 + 0.5j*m24 + 0.5*m27 + -0.5*m28 + 0.5j*m29 + -0.5j*m31 + -0.5j*m33 + -0.5*m34 + -0.5j*m37 + -0.5*m39 + -0.5*m40 + 0.5j*m41 + -0.5j*m42 + 0.5j*m43 + -0.5j*m45 + -0.5j*m46 + -0.5*m47
    C[3,0] = -0.5j*m0 + -0.5j*m1 + 0.5*m5 + 0.5j*m8 + 0.5j*m9 + (-0.5+0.5j)*m11 + -0.5j*m14 + -0.5j*m15 + (0.5+0.5j)*m16 + -0.5j*m17 + (0.5+0.5j)*m18 + 0.5*m24 + -0.5j*m26 + 0.5*m27 + 0.5*m28 + 0.5j*m30 + 0.5j*m32 + -0.5j*m34 + -0.5*m36 + 0.5*m37 + -0.5*m38 + (0.5+-0.5j)*m39 + -0.5j*m40 + 0.5*m42 + -0.5j*m43 + -0.5*m44 + 0.5j*m46 + -0.5j*m47
    C[3,1] = 0.5j*m0 + -0.5*m2 + (-0.5+-0.5j)*m3 + -0.5*m5 + 0.5*m6 + -0.5j*m8 + (0.5+-0.5j)*m11 + -0.5*m12 + 0.5j*m13 + -0.5*m14 + 0.5j*m15 + 0.5j*m17 + (-0.5+-0.5j)*m18 + -0.5*m20 + 0.5*m22 + -0.5*m24 + -0.5*m27 + -0.5*m28 + -0.5j*m29 + -0.5j*m32 + (0.5+0.5j)*m33 + 0.5j*m34 + 0.5j*m37 + 0.5j*m40 + -0.5j*m41 + -0.5*m43 + 0.5*m44 + 0.5*m47
    C[3,2] = 0.5*m2 + 0.5j*m3 + 0.5*m5 + -0.5*m8 + -0.5*m11 + 0.5*m12 + -0.5j*m13 + 0.5*m14 + -0.5j*m15 + 0.5j*m16 + 0.5j*m18 + 0.5j*m19 + 0.5*m20 + 0.5*m21 + -0.5*m23 + 0.5*m24 + 0.5*m25 + -0.5j*m26 + 0.5j*m27 + 0.5j*m30 + -0.5j*m31 + 0.5j*m32 + -0.5j*m33 + -0.5j*m34 + -0.5j*m35 + -0.5*m36 + 0.5*m37 + -0.5*m38 + 0.5*m39 + 0.5*m43 + -0.5*m44 + -0.5j*m47
    C[3,3] = -0.5j*m0 + -0.5j*m1 + 0.5*m3 + 0.5j*m4 + -0.5*m6 + -0.5*m7 + 0.5j*m8 + 0.5j*m9 + -0.5*m10 + 0.5j*m11 + -0.5j*m14 + 0.5*m16 + -0.5j*m17 + 0.5*m18 + -0.5j*m21 + -0.5*m22 + -0.5j*m24 + 0.5*m27 + 0.5*m28 + 0.5j*m29 + -0.5*m31 + -0.5*m33 + -0.5*m34 + -0.5j*m37 + -0.5j*m39 + -0.5j*m40 + 0.5j*m41 + 0.5*m42 + -0.5j*m43 + -0.5j*m45 + 0.5j*m46 + -0.5*m47

    # If input was real, ensure output is real
    if np.isrealobj(A) and np.isrealobj(B):
        C = np.real(C)

    return C

# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_correctness(use_quantum_rng=False):
    """Test the correctness of the algorithms.
    
    Args:
        use_quantum_rng: Whether to use quantum random numbers instead of pseudorandom
    """
    print("Testing correctness...")
    
    if use_quantum_rng:
        print("Using QUANTUM random numbers from ANU Quantum RNG!")
        # Get quantum random numbers for 4×4 matrices (16 numbers each)
        quantum_values = get_quantum_random_numbers(32)
        
        if quantum_values is not None:
            # Reshape into two 4×4 matrices
            A = quantum_values[:16].reshape(4, 4)
            B = quantum_values[16:32].reshape(4, 4)
            print("Successfully generated quantum random matrices!")
        else:
            # Fall back to numpy if quantum RNG fails
            print("Falling back to numpy random numbers.")
            np.random.seed(42)
            A = np.random.rand(4, 4)
            B = np.random.rand(4, 4)
    else:
        # Use numpy random numbers with seed for reproducibility
        np.random.seed(42)
        A = np.random.rand(4, 4)
        B = np.random.rand(4, 4)
    
    print("Matrix A (Real):")
    print(A)
    print("\nMatrix B (Real):")
    print(B)
    
    # Compute using numpy's built-in multiplication
    C_numpy = A @ B
    
    # Compute using standard algorithm
    C_standard = standard_multiply(A, B)
    
    # Compute using Strassen's algorithm
    C_strassen = strassen_4x4(A, B)
    
    # Compute using AlphaEvolve's algorithm
    C_alphaevolve = alphaevolve_4x4(A, B)
    
    print("\nResult using NumPy:")
    print(C_numpy)
    
    # Check if the results are close
    print("\nAccuracy checks:")
    print("Standard vs NumPy:", np.allclose(C_standard, C_numpy))
    print("Strassen vs NumPy:", np.allclose(C_strassen, C_numpy))
    print("AlphaEvolve vs NumPy:", np.allclose(C_alphaevolve, C_numpy))
    
    print("\nMax absolute error:")
    print("Standard: ", np.max(np.abs(C_standard - C_numpy)))
    print("Strassen: ", np.max(np.abs(C_strassen - C_numpy)))
    print("AlphaEvolve:", np.max(np.abs(C_alphaevolve - C_numpy)))
    
    # Test with complex matrices
    print("\n" + "="*50)
    print("Testing with complex matrices:")
    
    if use_quantum_rng:
        # Get quantum random numbers for complex matrices
        quantum_complex_values = get_quantum_random_numbers(32, use_complex=True)
        
        if quantum_complex_values is not None:
            # Reshape into two 4×4 matrices
            A_complex = quantum_complex_values[:16].reshape(4, 4)
            B_complex = quantum_complex_values[16:32].reshape(4, 4)
            print("Successfully generated quantum random complex matrices!")
        else:
            # Fall back to numpy if quantum RNG fails
            print("Falling back to numpy random numbers for complex matrices.")
            np.random.seed(43)
            A_complex = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
            B_complex = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
    else:
        # Use numpy random numbers with seed for reproducibility
        np.random.seed(43)
        A_complex = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
        B_complex = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
    
    print("\nMatrix A (Complex) - showing first row:")
    print(A_complex[0])
    print("\nMatrix B (Complex) - showing first row:")
    print(B_complex[0])
    
    # Compute using numpy's built-in multiplication
    C_numpy_complex = A_complex @ B_complex
    
    # Compute using standard algorithm
    C_standard_complex = standard_multiply(A_complex, B_complex)
    
    # Compute using Strassen's algorithm
    C_strassen_complex = strassen_4x4(A_complex, B_complex)
    
    # Compute using AlphaEvolve's algorithm
    C_alphaevolve_complex = alphaevolve_4x4(A_complex, B_complex)
    
    # Check if the results are close
    print("\nAccuracy checks for complex matrices:")
    print("Standard vs NumPy:", np.allclose(C_standard_complex, C_numpy_complex))
    print("Strassen vs NumPy:", np.allclose(C_strassen_complex, C_numpy_complex))
    print("AlphaEvolve vs NumPy:", np.allclose(C_alphaevolve_complex, C_numpy_complex))
    
    print("\nMax absolute error for complex matrices:")
    print("Standard: ", np.max(np.abs(C_standard_complex - C_numpy_complex)))
    print("Strassen: ", np.max(np.abs(C_strassen_complex - C_numpy_complex)))
    print("AlphaEvolve:", np.max(np.abs(C_alphaevolve_complex - C_numpy_complex)))
    
    print()

def test_performance(use_quantum_rng=False):
    """Test the performance of the algorithms.
    
    Args:
        use_quantum_rng: Whether to use quantum random numbers instead of pseudorandom
    """
    print("Testing performance...")
    
    # Number of iterations for more accurate timing
    n_iter = 1000
    
    if use_quantum_rng:
        print("Using QUANTUM random numbers from ANU Quantum RNG for performance testing!")
        # Get quantum random numbers for 4×4 matrices (16 numbers each)
        quantum_values = get_quantum_random_numbers(32)
        
        if quantum_values is not None:
            # Reshape into two 4×4 matrices
            A = quantum_values[:16].reshape(4, 4)
            B = quantum_values[16:32].reshape(4, 4)
            print("Successfully generated quantum random matrices for performance testing!")
        else:
            # Fall back to numpy if quantum RNG fails
            print("Falling back to numpy random numbers for performance testing.")
            np.random.seed(42)
            A = np.random.rand(4, 4)
            B = np.random.rand(4, 4)
    else:
        # Use numpy random numbers with seed for reproducibility
        np.random.seed(42)
        A = np.random.rand(4, 4)
        B = np.random.rand(4, 4)
    
    # Warm up
    _ = standard_multiply(A, B)
    _ = strassen_4x4(A, B)
    _ = alphaevolve_4x4(A, B)
    
    # Time standard multiplication
    start = time.time()
    for _ in range(n_iter):
        _ = standard_multiply(A, B)
    standard_time = time.time() - start
    
    # Time Strassen's algorithm
    start = time.time()
    for _ in range(n_iter):
        _ = strassen_4x4(A, B)
    strassen_time = time.time() - start
    
    # Time AlphaEvolve's algorithm
    start = time.time()
    for _ in range(n_iter):
        _ = alphaevolve_4x4(A, B)
    alphaevolve_time = time.time() - start
    
    print(f"Standard time: {standard_time:.6f}s for {n_iter} iterations")
    print(f"Strassen time: {strassen_time:.6f}s for {n_iter} iterations")
    print(f"AlphaEvolve time: {alphaevolve_time:.6f}s for {n_iter} iterations")
    
    if strassen_time > alphaevolve_time:
        print(f"AlphaEvolve is {strassen_time / alphaevolve_time:.3f}x faster than Strassen for real matrices")
    else:
        print(f"Strassen is {alphaevolve_time / strassen_time:.3f}x faster than AlphaEvolve for real matrices")
    
    print()
    
    # Test for complex matrices
    if use_quantum_rng:
        # Get quantum random numbers for complex matrices
        quantum_complex_values = get_quantum_random_numbers(32, use_complex=True)
        
        if quantum_complex_values is not None:
            # Reshape into two 4×4 matrices
            A_complex = quantum_complex_values[:16].reshape(4, 4)
            B_complex = quantum_complex_values[16:32].reshape(4, 4)
            print("Successfully generated quantum random complex matrices for performance testing!")
        else:
            # Fall back to numpy if quantum RNG fails
            print("Falling back to numpy random numbers for complex matrices in performance testing.")
            np.random.seed(43)
            A_complex = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
            B_complex = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
    else:
        # Use numpy random numbers with seed for reproducibility
        np.random.seed(43)
        A_complex = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
        B_complex = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
    
    # Warm up
    _ = standard_multiply(A_complex, B_complex)
    _ = strassen_4x4(A_complex, B_complex)
    _ = alphaevolve_4x4(A_complex, B_complex)
    
    # Time standard multiplication
    start = time.time()
    for _ in range(n_iter):
        _ = standard_multiply(A_complex, B_complex)
    standard_time_complex = time.time() - start
    
    # Time Strassen's algorithm
    start = time.time()
    for _ in range(n_iter):
        _ = strassen_4x4(A_complex, B_complex)
    strassen_time_complex = time.time() - start
    
    # Time AlphaEvolve's algorithm
    start = time.time()
    for _ in range(n_iter):
        _ = alphaevolve_4x4(A_complex, B_complex)
    alphaevolve_time_complex = time.time() - start
    
    print("Complex matrices:")
    print(f"Standard time: {standard_time_complex:.6f}s for {n_iter} iterations")
    print(f"Strassen time: {strassen_time_complex:.6f}s for {n_iter} iterations")
    print(f"AlphaEvolve time: {alphaevolve_time_complex:.6f}s for {n_iter} iterations")
    
    if strassen_time_complex > alphaevolve_time_complex:
        print(f"AlphaEvolve is {strassen_time_complex / alphaevolve_time_complex:.3f}x faster than Strassen for complex matrices")
    else:
        print(f"Strassen is {alphaevolve_time_complex / strassen_time_complex:.3f}x faster than AlphaEvolve for complex matrices")

# =============================================================================
# DEMONSTRATION
# =============================================================================

def main():
    """Run the demonstration."""
    
    print("=" * 80)
    print("Matrix Multiplication Algorithms Comparison")
    print("=" * 80)
    print("Comparing different matrix multiplication algorithms for 4×4 matrices:")
    print("1. Standard algorithm: Uses 64 scalar multiplications")
    print("2. Strassen's algorithm: Uses 49 scalar multiplications")
    print("3. AlphaEvolve's algorithm: Uses 48 scalar multiplications")
    print("=" * 80)
    
    # Ask user if they want to use quantum RNG
    try:
        use_quantum = input("Do you want to use quantum random numbers from ANU Quantum RNG? (y/n): ").lower().startswith('y')
    except:
        use_quantum = False
        
    test_correctness(use_quantum_rng=use_quantum)
    test_performance(use_quantum_rng=use_quantum)
    
    print("=" * 80)
    print("Conclusion:")
    print("AlphaEvolve's algorithm requires 48 scalar multiplications")
    print("compared to Strassen's 49, which is a mathematical breakthrough")
    print("after 56 years! The implementation demonstrates that the algorithm")
    print("works correctly for both real and complex matrices.")
    print("=" * 80)
    
    if use_quantum:
        print("\nYou used quantum randomness from the ANU Quantum RNG!\n")
        print("This means your test data was generated using quantum fluctuations")
        print("of the vacuum as measured by the Australian National University,")
        print("instead of a deterministic pseudorandom number generator.")
        print("\nAPI provided by: Australian National University Quantum Random Numbers Server")
        print("https://quantumnumbers.anu.edu.au/")
        print("=" * 80)

if __name__ == "__main__":
    main()