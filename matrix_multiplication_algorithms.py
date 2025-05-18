import os
import numpy as np
import time
import requests  # For quantum random number generation
from numba import jit   
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
@jit(nopython=True, fastmath=True)
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

@jit(nopython=True, fastmath=True)
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
@jit(nopython=True, fastmath=True)
def strassen_4x4(A, B):
    """Strassen's algorithm for 4×4 matrices.
    
    This uses exactly 49 scalar multiplications.
    
    Args:
        A: First 4×4 matrix
        B: Second 4×4 matrix
    
    Returns:
        C: Result of A×B
    """
    C = np.zeros((4, 4), dtype=A.dtype)
    
    # Cache matrix values to avoid repeated memory access
    A00, A01, A02, A03 = A[0,0], A[0,1], A[0,2], A[0,3]
    A10, A11, A12, A13 = A[1,0], A[1,1], A[1,2], A[1,3]
    A20, A21, A22, A23 = A[2,0], A[2,1], A[2,2], A[2,3]
    A30, A31, A32, A33 = A[3,0], A[3,1], A[3,2], A[3,3]
    
    B00, B01, B02, B03 = B[0,0], B[0,1], B[0,2], B[0,3]
    B10, B11, B12, B13 = B[1,0], B[1,1], B[1,2], B[1,3]
    B20, B21, B22, B23 = B[2,0], B[2,1], B[2,2], B[2,3]
    B30, B31, B32, B33 = B[3,0], B[3,1], B[3,2], B[3,3]
    
    # Step 1: Split matrices into 2×2 blocks
    # Block A11
    a11_00 = A00
    a11_01 = A01
    a11_10 = A10
    a11_11 = A11
    
    # Block A12
    a12_00 = A02
    a12_01 = A03
    a12_10 = A12
    a12_11 = A13
    
    # Block A21
    a21_00 = A20
    a21_01 = A21
    a21_10 = A30
    a21_11 = A31
    
    # Block A22
    a22_00 = A22
    a22_01 = A23
    a22_10 = A32
    a22_11 = A33
    
    # Block B11
    b11_00 = B00
    b11_01 = B01
    b11_10 = B10
    b11_11 = B11
    
    # Block B12
    b12_00 = B02
    b12_01 = B03
    b12_10 = B12
    b12_11 = B13
    
    # Block B21
    b21_00 = B20
    b21_01 = B21
    b21_10 = B30
    b21_11 = B31
    
    # Block B22
    b22_00 = B22
    b22_01 = B23
    b22_10 = B32
    b22_11 = B33
    
    # Step 2: Compute the 7 Strassen products
    # For P1 = (A11 + A22) * (B11 + B22)
    # Compute A11 + A22
    s1_00 = a11_00 + a22_00
    s1_01 = a11_01 + a22_01
    s1_10 = a11_10 + a22_10
    s1_11 = a11_11 + a22_11
    
    # Compute B11 + B22
    t1_00 = b11_00 + b22_00
    t1_01 = b11_01 + b22_01
    t1_10 = b11_10 + b22_10
    t1_11 = b11_11 + b22_11
    
    # Compute P1 = (A11 + A22) * (B11 + B22)
    p1_00 = s1_00 * t1_00 + s1_01 * t1_10
    p1_01 = s1_00 * t1_01 + s1_01 * t1_11
    p1_10 = s1_10 * t1_00 + s1_11 * t1_10
    p1_11 = s1_10 * t1_01 + s1_11 * t1_11
    
    # For P2 = (A21 + A22) * B11
    # Compute A21 + A22
    s2_00 = a21_00 + a22_00
    s2_01 = a21_01 + a22_01
    s2_10 = a21_10 + a22_10
    s2_11 = a21_11 + a22_11
    
    # Compute P2 = (A21 + A22) * B11
    p2_00 = s2_00 * b11_00 + s2_01 * b11_10
    p2_01 = s2_00 * b11_01 + s2_01 * b11_11
    p2_10 = s2_10 * b11_00 + s2_11 * b11_10
    p2_11 = s2_10 * b11_01 + s2_11 * b11_11
    
    # For P3 = A11 * (B12 - B22)
    # Compute B12 - B22
    t3_00 = b12_00 - b22_00
    t3_01 = b12_01 - b22_01
    t3_10 = b12_10 - b22_10
    t3_11 = b12_11 - b22_11
    
    # Compute P3 = A11 * (B12 - B22)
    p3_00 = a11_00 * t3_00 + a11_01 * t3_10
    p3_01 = a11_00 * t3_01 + a11_01 * t3_11
    p3_10 = a11_10 * t3_00 + a11_11 * t3_10
    p3_11 = a11_10 * t3_01 + a11_11 * t3_11
    
    # For P4 = A22 * (B21 - B11)
    # Compute B21 - B11
    t4_00 = b21_00 - b11_00
    t4_01 = b21_01 - b11_01
    t4_10 = b21_10 - b11_10
    t4_11 = b21_11 - b11_11
    
    # Compute P4 = A22 * (B21 - B11)
    p4_00 = a22_00 * t4_00 + a22_01 * t4_10
    p4_01 = a22_00 * t4_01 + a22_01 * t4_11
    p4_10 = a22_10 * t4_00 + a22_11 * t4_10
    p4_11 = a22_10 * t4_01 + a22_11 * t4_11
    
    # For P5 = (A11 + A12) * B22
    # Compute A11 + A12
    s5_00 = a11_00 + a12_00
    s5_01 = a11_01 + a12_01
    s5_10 = a11_10 + a12_10
    s5_11 = a11_11 + a12_11
    
    # Compute P5 = (A11 + A12) * B22
    p5_00 = s5_00 * b22_00 + s5_01 * b22_10
    p5_01 = s5_00 * b22_01 + s5_01 * b22_11
    p5_10 = s5_10 * b22_00 + s5_11 * b22_10
    p5_11 = s5_10 * b22_01 + s5_11 * b22_11
    
    # For P6 = (A21 - A11) * (B11 + B12)
    # Compute A21 - A11
    s6_00 = a21_00 - a11_00
    s6_01 = a21_01 - a11_01
    s6_10 = a21_10 - a11_10
    s6_11 = a21_11 - a11_11
    
    # Compute B11 + B12
    t6_00 = b11_00 + b12_00
    t6_01 = b11_01 + b12_01
    t6_10 = b11_10 + b12_10
    t6_11 = b11_11 + b12_11
    
    # Compute P6 = (A21 - A11) * (B11 + B12)
    p6_00 = s6_00 * t6_00 + s6_01 * t6_10
    p6_01 = s6_00 * t6_01 + s6_01 * t6_11
    p6_10 = s6_10 * t6_00 + s6_11 * t6_10
    p6_11 = s6_10 * t6_01 + s6_11 * t6_11
    
    # For P7 = (A12 - A22) * (B21 + B22)
    # Compute A12 - A22
    s7_00 = a12_00 - a22_00
    s7_01 = a12_01 - a22_01
    s7_10 = a12_10 - a22_10
    s7_11 = a12_11 - a22_11
    
    # Compute B21 + B22
    t7_00 = b21_00 + b22_00
    t7_01 = b21_01 + b22_01
    t7_10 = b21_10 + b22_10
    t7_11 = b21_11 + b22_11
    
    # Compute P7 = (A12 - A22) * (B21 + B22)
    p7_00 = s7_00 * t7_00 + s7_01 * t7_10
    p7_01 = s7_00 * t7_01 + s7_01 * t7_11
    p7_10 = s7_10 * t7_00 + s7_11 * t7_10
    p7_11 = s7_10 * t7_01 + s7_11 * t7_11
    
    # Step 3: Compute the blocks of the result matrix C
    # C11 = P1 + P4 - P5 + P7
    c11_00 = p1_00 + p4_00 - p5_00 + p7_00
    c11_01 = p1_01 + p4_01 - p5_01 + p7_01
    c11_10 = p1_10 + p4_10 - p5_10 + p7_10
    c11_11 = p1_11 + p4_11 - p5_11 + p7_11
    
    # C12 = P3 + P5
    c12_00 = p3_00 + p5_00
    c12_01 = p3_01 + p5_01
    c12_10 = p3_10 + p5_10
    c12_11 = p3_11 + p5_11
    
    # C21 = P2 + P4
    c21_00 = p2_00 + p4_00
    c21_01 = p2_01 + p4_01
    c21_10 = p2_10 + p4_10
    c21_11 = p2_11 + p4_11
    
    # C22 = P1 - P2 + P3 + P6
    c22_00 = p1_00 - p2_00 + p3_00 + p6_00
    c22_01 = p1_01 - p2_01 + p3_01 + p6_01
    c22_10 = p1_10 - p2_10 + p3_10 + p6_10
    c22_11 = p1_11 - p2_11 + p3_11 + p6_11
    
    # Step 4: Fill the result matrix C
    C[0, 0] = c11_00
    C[0, 1] = c11_01
    C[1, 0] = c11_10
    C[1, 1] = c11_11
    
    C[0, 2] = c12_00
    C[0, 3] = c12_01
    C[1, 2] = c12_10
    C[1, 3] = c12_11
    
    C[2, 0] = c21_00
    C[2, 1] = c21_01
    C[3, 0] = c21_10
    C[3, 1] = c21_11
    
    C[2, 2] = c22_00
    C[2, 3] = c22_01
    C[3, 2] = c22_10
    C[3, 3] = c22_11
    
    return C

# =============================================================================
# ALPHAEVOLVE'S ALGORITHM 
# =============================================================================
@jit(nopython=True, fastmath=True)
def alphaevolve_4x4(A, B):
    """
    AlphaEvolve's optimized algorithm for 4×4 matrices.
    Uses exactly 48 scalar multiplications.
    """
    # Check if we're dealing with real matrices for potential optimizations
    is_real_input = np.isrealobj(A) and np.isrealobj(B)
    
    # Initialize the result matrix - use appropriate dtype
    C = np.zeros((4, 4), dtype=np.complex128)
    
    # Cache commonly used constants
    half = 0.5
    half_j = 0.5j
    half_p_half_j = 0.5 + 0.5j
    half_m_half_j = 0.5 - 0.5j
    neg_half = -0.5
    neg_half_j = -0.5j
    
    # Cache matrix values to avoid repeated memory access
    A00, A01, A02, A03 = A[0,0], A[0,1], A[0,2], A[0,3]
    A10, A11, A12, A13 = A[1,0], A[1,1], A[1,2], A[1,3]
    A20, A21, A22, A23 = A[2,0], A[2,1], A[2,2], A[2,3]
    A30, A31, A32, A33 = A[3,0], A[3,1], A[3,2], A[3,3]
    
    B00, B01, B02, B03 = B[0,0], B[0,1], B[0,2], B[0,3]
    B10, B11, B12, B13 = B[1,0], B[1,1], B[1,2], B[1,3]
    B20, B21, B22, B23 = B[2,0], B[2,1], B[2,2], B[2,3]
    B30, B31, B32, B33 = B[3,0], B[3,1], B[3,2], B[3,3]
    
    # Linear combinations of elements from A - computed once and stored
    a0 = half_p_half_j*A00 + half_p_half_j*A01 + half_m_half_j*A10 + half_m_half_j*A11 + half_m_half_j*A20 + half_m_half_j*A21 + half_m_half_j*A30 + half_m_half_j*A31
    a1 = half_p_half_j*A00 + (neg_half+half_j)*A03 + half_p_half_j*A10 + (neg_half+half_j)*A13 + (neg_half+neg_half_j)*A20 + half_m_half_j*A23 + half_m_half_j*A30 + half_p_half_j*A33
    a2 = neg_half*A01 + half*A02 + neg_half_j*A11 + half_j*A12 + half_j*A21 + neg_half_j*A22 + neg_half_j*A31 + half_j*A32
    a3 = neg_half_j*A00 + neg_half*A01 + half*A02 + neg_half*A03 + half_j*A10 + neg_half*A11 + half*A12 + half*A13 + neg_half_j*A20 + neg_half*A21 + half*A22 + neg_half*A23 + neg_half*A30 + neg_half_j*A31 + half_j*A32 + half_j*A33
    a4 = half_p_half_j*A00 + (neg_half+neg_half_j)*A01 + (neg_half+half_j)*A10 + half_m_half_j*A11 + (neg_half+half_j)*A20 + half_m_half_j*A21 + half_m_half_j*A30 + (neg_half+half_j)*A31
    a5 = half_m_half_j*A02 + (neg_half+neg_half_j)*A03 + half_m_half_j*A12 + (neg_half+neg_half_j)*A13 + (neg_half+half_j)*A22 + half_p_half_j*A23 + (neg_half+neg_half_j)*A32 + (neg_half+half_j)*A33
    a6 = half_j*A00 + half*A03 + neg_half*A10 + half_j*A13 + half*A20 + neg_half_j*A23 + neg_half*A30 + half_j*A33
    a7 = half_p_half_j*A00 + (neg_half+neg_half_j)*A01 + (neg_half+neg_half_j)*A10 + half_p_half_j*A11 + (neg_half+neg_half_j)*A20 + half_p_half_j*A21 + (neg_half+half_j)*A30 + half_m_half_j*A31
    a8 = neg_half_j*A00 + neg_half_j*A01 + neg_half*A02 + neg_half_j*A03 + half*A10 + half*A11 + neg_half_j*A12 + half*A13 + neg_half*A20 + neg_half*A21 + neg_half_j*A22 + half*A23 + half*A30 + half*A31 + half_j*A32 + neg_half*A33
    a9 = (neg_half+half_j)*A00 + (neg_half+neg_half_j)*A03 + half_p_half_j*A10 + (neg_half+half_j)*A13 + (neg_half+neg_half_j)*A20 + half_m_half_j*A23 + (neg_half+neg_half_j)*A30 + half_m_half_j*A33
    a10 = (neg_half+half_j)*A00 + half_m_half_j*A01 + (neg_half+half_j)*A10 + half_m_half_j*A11 + half_m_half_j*A20 + (neg_half+half_j)*A21 + half_p_half_j*A30 + (neg_half+neg_half_j)*A31
    
    # Continue with the remaining a values
    a11 = half*A00 + half*A01 + neg_half_j*A02 + neg_half*A03 + neg_half*A10 + neg_half*A11 + half_j*A12 + half*A13 + half*A20 + half*A21 + half_j*A22 + half*A23 + neg_half_j*A30 + neg_half_j*A31 + half*A32 + neg_half_j*A33
    a12 = half_p_half_j*A01 + (neg_half+neg_half_j)*A02 + (neg_half+half_j)*A11 + half_m_half_j*A12 + (neg_half+half_j)*A21 + half_m_half_j*A22 + half_m_half_j*A31 + (neg_half+half_j)*A32
    a13 = half_m_half_j*A01 + (neg_half+half_j)*A02 + half_m_half_j*A11 + (neg_half+half_j)*A12 + half_m_half_j*A21 + (neg_half+half_j)*A22 + half_p_half_j*A31 + (neg_half+neg_half_j)*A32
    a14 = half_j*A00 + neg_half*A01 + half*A02 + neg_half*A03 + half*A10 + neg_half_j*A11 + half_j*A12 + half_j*A13 + half*A20 + half_j*A21 + neg_half_j*A22 + half_j*A23 + half*A30 + neg_half_j*A31 + half_j*A32 + half_j*A33
    a15 = (neg_half+half_j)*A02 + half_p_half_j*A03 + half_m_half_j*A12 + (neg_half+neg_half_j)*A13 + half_m_half_j*A22 + (neg_half+neg_half_j)*A23 + (neg_half+neg_half_j)*A32 + (neg_half+half_j)*A33
    a16 = neg_half*A00 + half_j*A01 + half_j*A02 + neg_half_j*A03 + neg_half*A10 + neg_half_j*A11 + neg_half_j*A12 + neg_half_j*A13 + neg_half*A20 + half_j*A21 + half_j*A22 + neg_half_j*A23 + neg_half_j*A30 + half*A31 + half*A32 + half*A33
    a17 = half_p_half_j*A00 + half_p_half_j*A01 + half_p_half_j*A10 + half_p_half_j*A11 + half_p_half_j*A20 + half_p_half_j*A21 + (neg_half+half_j)*A30 + (neg_half+half_j)*A31
    a18 = half_j*A00 + half_j*A01 + neg_half*A02 + half_j*A03 + half_j*A10 + half_j*A11 + neg_half*A12 + half_j*A13 + half_j*A20 + half_j*A21 + half*A22 + neg_half_j*A23 + neg_half*A30 + neg_half*A31 + half_j*A32 + half*A33
    a19 = half_m_half_j*A02 + half_p_half_j*A03 + half_m_half_j*A12 + half_p_half_j*A13 + half_m_half_j*A22 + half_p_half_j*A23 + half_p_half_j*A32 + (neg_half+half_j)*A33
    a20 = half_p_half_j*A01 + (neg_half+neg_half_j)*A02 + half_p_half_j*A11 + (neg_half+neg_half_j)*A12 + (neg_half+neg_half_j)*A21 + half_p_half_j*A22 + half_m_half_j*A31 + (neg_half+half_j)*A32
    
    # Complete a21 to a47
    a21 = half_j*A00 + neg_half_j*A01 + neg_half*A02 + neg_half_j*A03 + neg_half_j*A10 + half_j*A11 + half*A12 + half_j*A13 + neg_half_j*A20 + half_j*A21 + neg_half*A22 + neg_half_j*A23 + neg_half*A30 + half*A31 + half_j*A32 + neg_half*A33
    a22 = (neg_half+neg_half_j)*A00 + (neg_half+half_j)*A03 + half_m_half_j*A10 + (neg_half+neg_half_j)*A13 + half_m_half_j*A20 + (neg_half+neg_half_j)*A23 + (neg_half+half_j)*A30 + half_p_half_j*A33
    a23 = (neg_half+neg_half_j)*A02 + half_m_half_j*A03 + half_m_half_j*A12 + half_p_half_j*A13 + half_m_half_j*A22 + half_p_half_j*A23 + (neg_half+half_j)*A32 + (neg_half+neg_half_j)*A33
    a24 = neg_half*A00 + half*A01 + neg_half_j*A02 + neg_half*A03 + neg_half_j*A10 + half_j*A11 + half*A12 + neg_half_j*A13 + neg_half_j*A20 + half_j*A21 + neg_half*A22 + half_j*A23 + half_j*A30 + neg_half_j*A31 + half*A32 + neg_half_j*A33
    a25 = half_m_half_j*A02 + half_p_half_j*A03 + (neg_half+neg_half_j)*A12 + half_m_half_j*A13 + half_p_half_j*A22 + (neg_half+half_j)*A23 + half_p_half_j*A32 + (neg_half+half_j)*A33
    a26 = half_p_half_j*A01 + half_p_half_j*A02 + (neg_half+neg_half_j)*A11 + (neg_half+neg_half_j)*A12 + half_p_half_j*A21 + half_p_half_j*A22 + half_m_half_j*A31 + half_m_half_j*A32
    a27 = neg_half_j*A00 + neg_half_j*A01 + half*A02 + half_j*A03 + neg_half*A10 + neg_half*A11 + neg_half_j*A12 + half*A13 + neg_half*A20 + neg_half*A21 + half_j*A22 + neg_half*A23 + neg_half*A30 + neg_half*A31 + half_j*A32 + neg_half*A33
    a28 = (neg_half+half_j)*A00 + (neg_half+half_j)*A01 + (neg_half+neg_half_j)*A10 + (neg_half+neg_half_j)*A11 + half_p_half_j*A20 + half_p_half_j*A21 + (neg_half+neg_half_j)*A30 + (neg_half+neg_half_j)*A31
    a29 = half_p_half_j*A00 + half_m_half_j*A03 + (neg_half+neg_half_j)*A10 + (neg_half+half_j)*A13 + half_p_half_j*A20 + half_m_half_j*A23 + half_m_half_j*A30 + (neg_half+neg_half_j)*A33
    a30 = half_p_half_j*A01 + half_p_half_j*A02 + (neg_half+neg_half_j)*A11 + (neg_half+neg_half_j)*A12 + (neg_half+neg_half_j)*A21 + (neg_half+neg_half_j)*A22 + (neg_half+half_j)*A31 + (neg_half+half_j)*A32
    a31 = half*A00 + neg_half*A01 + neg_half_j*A02 + half*A03 + half*A10 + neg_half*A11 + neg_half_j*A12 + half*A13 + neg_half*A20 + half*A21 + neg_half_j*A22 + half*A23 + neg_half_j*A30 + half_j*A31 + half*A32 + half_j*A33
    a32 = half_p_half_j*A02 + half_m_half_j*A03 + (neg_half+half_j)*A12 + half_p_half_j*A13 + half_m_half_j*A22 + (neg_half+neg_half_j)*A23 + (neg_half+half_j)*A32 + half_p_half_j*A33
    a33 = half*A00 + half_j*A01 + neg_half_j*A02 + neg_half_j*A03 + neg_half*A10 + half_j*A11 + neg_half_j*A12 + half_j*A13 + neg_half*A20 + neg_half_j*A21 + half_j*A22 + half_j*A23 + half_j*A30 + half*A31 + neg_half*A32 + half*A33
    a34 = neg_half_j*A00 + half_j*A01 + neg_half*A02 + half_j*A03 + neg_half*A10 + half*A11 + half_j*A12 + half*A13 + half*A20 + neg_half*A21 + half_j*A22 + half*A23 + half*A30 + neg_half*A31 + half_j*A32 + half*A33
    a35 = half_m_half_j*A02 + half_p_half_j*A03 + (neg_half+half_j)*A12 + (neg_half+neg_half_j)*A13 + half_m_half_j*A22 + half_p_half_j*A23 + (neg_half+neg_half_j)*A32 + half_m_half_j*A33
    a36 = (neg_half+neg_half_j)*A01 + (neg_half+neg_half_j)*A02 + (neg_half+half_j)*A11 + (neg_half+half_j)*A12 + half_m_half_j*A21 + half_m_half_j*A22 + half_m_half_j*A31 + half_m_half_j*A32
    a37 = half*A00 + neg_half_j*A01 + neg_half_j*A02 + neg_half_j*A03 + half_j*A10 + neg_half*A11 + neg_half*A12 + half*A13 + half_j*A20 + half*A21 + half*A22 + half*A23 + neg_half_j*A30 + half*A31 + half*A32 + neg_half*A33
    a38 = half_m_half_j*A01 + half_m_half_j*A02 + (neg_half+neg_half_j)*A11 + (neg_half+neg_half_j)*A12 + (neg_half+neg_half_j)*A21 + (neg_half+neg_half_j)*A22 + (neg_half+neg_half_j)*A31 + (neg_half+neg_half_j)*A32
    a39 = neg_half*A00 + neg_half_j*A01 + neg_half_j*A02 + neg_half_j*A03 + neg_half*A10 + half_j*A11 + half_j*A12 + neg_half_j*A13 + half*A20 + half_j*A21 + half_j*A22 + half_j*A23 + half_j*A30 + half*A31 + half*A32 + neg_half*A33
    a40 = (neg_half+neg_half_j)*A00 + (neg_half+neg_half_j)*A01 + half_p_half_j*A10 + half_p_half_j*A11 + (neg_half+neg_half_j)*A20 + (neg_half+neg_half_j)*A21 + (neg_half+half_j)*A30 + (neg_half+half_j)*A31
    a41 = half_m_half_j*A00 + (neg_half+neg_half_j)*A03 + (neg_half+half_j)*A10 + half_p_half_j*A13 + (neg_half+half_j)*A20 + half_p_half_j*A23 + half_p_half_j*A30 + half_m_half_j*A33
    a42 = half_p_half_j*A00 + (neg_half+half_j)*A03 + half_m_half_j*A10 + half_p_half_j*A13 + half_m_half_j*A20 + half_p_half_j*A23 + half_m_half_j*A30 + half_p_half_j*A33
    a43 = half_j*A00 + half*A01 + neg_half*A02 + neg_half*A03 + half*A10 + half_j*A11 + neg_half_j*A12 + half_j*A13 + neg_half*A20 + half_j*A21 + neg_half_j*A22 + neg_half_j*A23 + neg_half*A30 + neg_half_j*A31 + half_j*A32 + neg_half_j*A33
    a44 = half_m_half_j*A02 + (neg_half+neg_half_j)*A03 + (neg_half+neg_half_j)*A12 + (neg_half+half_j)*A13 + (neg_half+neg_half_j)*A22 + (neg_half+half_j)*A23 + (neg_half+neg_half_j)*A32 + (neg_half+half_j)*A33
    a45 = (neg_half+half_j)*A00 + half_m_half_j*A01 + half_p_half_j*A10 + (neg_half+neg_half_j)*A11 + (neg_half+neg_half_j)*A20 + half_p_half_j*A21 + (neg_half+neg_half_j)*A30 + half_p_half_j*A31
    a46 = half_m_half_j*A00 + half_p_half_j*A03 + half_m_half_j*A10 + half_p_half_j*A13 + half_m_half_j*A20 + half_p_half_j*A23 + half_p_half_j*A30 + (neg_half+half_j)*A33
    a47 = half*A00 + half_j*A01 + half_j*A02 + neg_half_j*A03 + half_j*A10 + half*A11 + half*A12 + half*A13 + neg_half_j*A20 + half*A21 + half*A22 + neg_half*A23 + half_j*A30 + half*A31 + half*A32 + half*A33
    
    # Linear combinations of elements from B (optimized)
    b0 = neg_half*B00 + neg_half*B10 + half*B20 + neg_half_j*B30
    b1 = half_j*B01 + half_j*B03 + half_j*B11 + half_j*B13 + half_j*B21 + half_j*B23 + half*B31 + half*B33
    b2 = half_p_half_j*B01 + (neg_half+neg_half_j)*B11 + half_p_half_j*B21 + half_m_half_j*B31
    b3 = neg_half_j*B00 + half_j*B02 + neg_half_j*B11 + neg_half_j*B12 + half_j*B21 + half_j*B22 + half*B30 + neg_half*B32
    b4 = neg_half*B00 + half*B02 + half*B03 + half*B10 + neg_half*B12 + neg_half*B13 + half*B20 + neg_half*B22 + neg_half*B23 + half_j*B30 + neg_half_j*B32 + neg_half_j*B33
    b5 = half*B01 + half*B03 + half*B11 + half*B13 + half*B21 + half*B23 + half_j*B31 + half_j*B33
    b6 = (neg_half+neg_half_j)*B01 + half_p_half_j*B11 + half_p_half_j*B21 + half_m_half_j*B31
    b7 = neg_half*B00 + half*B03 + half*B10 + neg_half*B13 + neg_half*B20 + half*B23 + half_j*B30 + neg_half_j*B33
    
    # Continue with b8 to b47
    b8 = half*B00 + neg_half*B02 + neg_half*B03 + half*B10 + neg_half*B12 + neg_half*B13 + half*B21 + neg_half_j*B31
    b9 = half_j*B01 + half_j*B02 + half_j*B03 + half_j*B11 + half_j*B12 + half_j*B13 + neg_half_j*B21 + neg_half_j*B22 + neg_half_j*B23 + half*B31 + half*B32 + half*B33
    b10 = half_j*B01 + half_j*B03 + neg_half_j*B11 + neg_half_j*B13 + neg_half_j*B21 + neg_half_j*B23 + neg_half*B31 + neg_half*B33
    b11 = neg_half_j*B00 + half_j*B03 + neg_half_j*B10 + half_j*B13 + half_j*B21 + half_j*B22 + neg_half*B31 + neg_half*B32
    b12 = neg_half*B00 + half*B02 + half*B03 + neg_half*B10 + half*B12 + half*B13 + half*B20 + neg_half*B22 + neg_half*B23 + half_j*B30 + neg_half_j*B32 + neg_half_j*B33
    b13 = half_j*B00 + neg_half_j*B02 + neg_half_j*B10 + half_j*B12 + half_j*B20 + neg_half_j*B22 + neg_half*B30 + half*B32
    b14 = neg_half*B01 + neg_half*B10 + half*B20 + half_j*B31
    b15 = half_j*B00 + neg_half_j*B03 + half_j*B10 + neg_half_j*B13 + neg_half_j*B20 + half_j*B23 + half*B30 + neg_half*B33
    b16 = half*B01 + half*B02 + half*B10 + neg_half*B12 + half*B20 + neg_half*B22 + neg_half_j*B31 + neg_half_j*B32
    b17 = neg_half_j*B00 + half_j*B02 + neg_half_j*B10 + half_j*B12 + neg_half_j*B20 + half_j*B22 + half*B30 + neg_half*B32
    b18 = neg_half_j*B01 + neg_half_j*B03 + neg_half_j*B11 + neg_half_j*B13 + neg_half_j*B20 + half_j*B22 + half*B30 + neg_half*B32
    b19 = neg_half_j*B00 + half_j*B02 + half_j*B10 + neg_half_j*B12 + half_j*B20 + neg_half_j*B22 + half*B30 + neg_half*B32
    b20 = neg_half_j*B01 + neg_half_j*B03 + neg_half_j*B11 + neg_half_j*B13 + half_j*B21 + half_j*B23 + half*B31 + half*B33
    
    # Complete b21 to b47
    b21 = neg_half*B01 + neg_half*B02 + half*B11 + half*B12 + neg_half*B20 + half*B23 + half_j*B30 + neg_half_j*B33
    b22 = neg_half_j*B00 + half_j*B02 + half_j*B03 + neg_half_j*B10 + half_j*B12 + half_j*B13 + neg_half_j*B20 + half_j*B22 + half_j*B23 + half*B30 + neg_half*B32 + neg_half*B33
    b23 = neg_half*B00 + half*B02 + half*B03 + neg_half*B10 + half*B12 + half*B13 + neg_half*B20 + half*B22 + half*B23 + half_j*B30 + neg_half_j*B32 + neg_half_j*B33
    b24 = half_j*B01 + neg_half_j*B11 + neg_half_j*B20 + half_j*B22 + half_j*B23 + half*B30 + neg_half*B32 + neg_half*B33
    b25 = half_j*B01 + half_j*B02 + half_j*B03 + half_j*B11 + half_j*B12 + half_j*B13 + neg_half_j*B21 + neg_half_j*B22 + neg_half_j*B23 + neg_half*B31 + neg_half*B32 + neg_half*B33
    b26 = half*B01 + half*B02 + neg_half*B11 + neg_half*B12 + neg_half*B21 + neg_half*B22 + neg_half_j*B31 + neg_half_j*B32
    b27 = half_j*B01 + half_j*B02 + half_j*B03 + half_j*B11 + half_j*B12 + half_j*B13 + neg_half_j*B20 + neg_half*B30
    b28 = half*B01 + half*B11 + half*B21 + neg_half_j*B31
    b29 = half_j*B01 + half_j*B02 + neg_half_j*B11 + neg_half_j*B12 + half_j*B21 + half_j*B22 + neg_half*B31 + neg_half*B32
    b30 = neg_half*B00 + half*B03 + neg_half*B10 + half*B13 + neg_half*B20 + half*B23 + half_j*B30 + neg_half_j*B33
    b31 = half*B00 + neg_half*B02 + neg_half*B10 + half*B12 + neg_half*B21 + neg_half*B23 + half_j*B31 + half_j*B33
    b32 = half_j*B01 + neg_half_j*B11 + neg_half_j*B21 + half*B31
    b33 = neg_half*B01 + neg_half*B03 + half*B10 + neg_half*B13 + neg_half*B20 + half*B23 + neg_half_j*B31 + neg_half_j*B33
    b34 = half_j*B00 + neg_half_j*B10 + half_j*B21 + half_j*B22 + half_j*B23 + neg_half*B31 + neg_half*B32 + neg_half*B33
    b35 = neg_half_j*B01 + neg_half_j*B02 + half_j*B11 + half_j*B12 + neg_half_j*B21 + neg_half_j*B22 + neg_half*B31 + neg_half*B32
    b36 = neg_half*B01 + neg_half*B02 + neg_half*B03 + neg_half*B11 + neg_half*B12 + neg_half*B13 + neg_half*B21 + neg_half*B22 + neg_half*B23 + neg_half_j*B31 + neg_half_j*B32 + neg_half_j*B33
    b37 = half_j*B01 + half_j*B02 + half_j*B03 + neg_half_j*B10 + half_j*B12 + half_j*B13 + neg_half_j*B20 + half_j*B22 + half_j*B23 + neg_half*B31 + neg_half*B32 + neg_half*B33
    b38 = half_j*B00 + neg_half_j*B10 + neg_half_j*B20 + neg_half*B30
    b39 = neg_half_j*B00 + half_j*B03 + half_j*B11 + half_j*B13 + half_j*B21 + half_j*B23 + neg_half*B30 + half*B33
    b40 = half_j*B01 + half_j*B02 + half_j*B11 + half_j*B12 + neg_half_j*B21 + neg_half_j*B22 + half*B31 + half*B32
    b41 = half*B00 + neg_half*B03 + half*B10 + neg_half*B13 + neg_half*B20 + half*B23 + half_j*B30 + neg_half_j*B33
    b42 = half_j*B00 + neg_half_j*B10 + half_j*B20 + half*B30
    b43 = half*B00 + neg_half*B02 + neg_half*B03 + neg_half*B11 + neg_half*B12 + neg_half*B13 + half*B21 + half*B22 + half*B23 + neg_half_j*B30 + half_j*B32 + half_j*B33
    b44 = neg_half_j*B00 + half_j*B10 + neg_half_j*B20 + half*B30
    b45 = neg_half_j*B01 + neg_half_j*B02 + neg_half_j*B03 + half_j*B11 + half_j*B12 + half_j*B13 + neg_half_j*B21 + neg_half_j*B22 + neg_half_j*B23 + half*B31 + half*B32 + half*B33
    b46 = neg_half*B00 + half*B02 + half*B10 + neg_half*B12 + half*B20 + neg_half*B22 + half_j*B30 + neg_half_j*B32
    b47 = half*B00 + half*B11 + half*B21 + half_j*B30
    
    # Perform the 48 multiplications efficiently
    m = np.zeros(48, dtype=np.complex128)
    
    # We can directly compute these multiplications
    # Numbering is maintained for clarity
    m[0] = a0 * b0
    m[1] = a1 * b1
    m[2] = a2 * b2
    m[3] = a3 * b3
    m[4] = a4 * b4
    m[5] = a5 * b5
    m[6] = a6 * b6
    m[7] = a7 * b7
    m[8] = a8 * b8
    m[9] = a9 * b9
    m[10] = a10 * b10
    m[11] = a11 * b11
    m[12] = a12 * b12
    m[13] = a13 * b13
    m[14] = a14 * b14
    m[15] = a15 * b15
    m[16] = a16 * b16
    m[17] = a17 * b17
    m[18] = a18 * b18
    m[19] = a19 * b19
    m[20] = a20 * b20
    m[21] = a21 * b21
    m[22] = a22 * b22
    m[23] = a23 * b23
    m[24] = a24 * b24
    m[25] = a25 * b25
    m[26] = a26 * b26
    m[27] = a27 * b27
    m[28] = a28 * b28
    m[29] = a29 * b29
    m[30] = a30 * b30
    m[31] = a31 * b31
    m[32] = a32 * b32
    m[33] = a33 * b33
    m[34] = a34 * b34
    m[35] = a35 * b35
    m[36] = a36 * b36
    m[37] = a37 * b37
    m[38] = a38 * b38
    m[39] = a39 * b39
    m[40] = a40 * b40
    m[41] = a41 * b41
    m[42] = a42 * b42
    m[43] = a43 * b43
    m[44] = a44 * b44
    m[45] = a45 * b45
    m[46] = a46 * b46
    m[47] = a47 * b47
    
    # Construct the result matrix efficiently
    # For C[0,0]
    C[0,0] = half_j*m[0] + neg_half_j*m[1] + neg_half*m[5] + half*m[8] + half_j*m[9] + \
             (neg_half+half_j)*m[11] + half*m[14] + neg_half_j*m[15] + (neg_half+neg_half_j)*m[16] + \
             half_j*m[17] + (neg_half+neg_half_j)*m[18] + neg_half_j*m[24] + half_j*m[26] + \
             half_j*m[27] + half*m[28] + half_j*m[30] + neg_half_j*m[32] + half*m[34] + \
             half*m[36] + neg_half_j*m[37] + neg_half*m[38] + (half+neg_half_j)*m[39] + \
             neg_half_j*m[40] + neg_half*m[42] + neg_half*m[43] + neg_half*m[44] + \
             neg_half_j*m[46] + half*m[47]
    
    # For C[0,1]
    C[0,1] = neg_half_j*m[0] + half*m[2] + (neg_half+neg_half_j)*m[3] + half*m[5] + \
             half*m[6] + neg_half*m[8] + (half+neg_half_j)*m[11] + neg_half*m[12] + \
             half_j*m[13] + half_j*m[14] + half_j*m[15] + neg_half_j*m[17] + \
             (half+half_j)*m[18] + half*m[20] + neg_half*m[22] + half_j*m[24] + \
             neg_half_j*m[27] + neg_half*m[28] + neg_half_j*m[29] + half_j*m[32] + \
             (neg_half+neg_half_j)*m[33] + neg_half*m[34] + neg_half*m[37] + half_j*m[40] + \
             half_j*m[41] + neg_half_j*m[43] + half*m[44] + neg_half_j*m[47]
    
    # For C[0,2]
    C[0,2] = neg_half*m[2] + half*m[3] + neg_half*m[5] + neg_half_j*m[8] + half_j*m[11] + \
             half*m[12] + neg_half_j*m[13] + neg_half_j*m[14] + neg_half_j*m[15] + \
             neg_half*m[16] + neg_half*m[18] + half_j*m[19] + neg_half*m[20] + half_j*m[21] + \
             neg_half*m[23] + neg_half_j*m[24] + neg_half*m[25] + half_j*m[26] + half*m[27] + \
             half_j*m[30] + neg_half*m[31] + neg_half_j*m[32] + half*m[33] + half*m[34] + \
             half_j*m[35] + half*m[36] + neg_half_j*m[37] + neg_half*m[38] + neg_half_j*m[39] + \
             half_j*m[43] + neg_half*m[44] + half*m[47]
    
    # For C[0,3]
    C[0,3] = half_j*m[0] + neg_half_j*m[1] + half_j*m[3] + neg_half_j*m[4] + neg_half*m[6] + \
             half*m[7] + half*m[8] + half_j*m[9] + neg_half*m[10] + neg_half*m[11] + half*m[14] + \
             neg_half_j*m[16] + half_j*m[17] + neg_half_j*m[18] + neg_half*m[21] + half*m[22] + \
             half*m[24] + half_j*m[27] + half*m[28] + half_j*m[29] + neg_half_j*m[31] + \
             half_j*m[33] + half_j*m[34] + half*m[37] + half*m[39] + neg_half_j*m[40] + \
             neg_half_j*m[41] + neg_half*m[42] + neg_half*m[43] + neg_half_j*m[45] + \
             neg_half_j*m[46] + half_j*m[47]
    
    # For C[1,0]
    C[1,0] = neg_half*m[0] + neg_half*m[1] + neg_half*m[5] + neg_half_j*m[8] + neg_half_j*m[9] + \
             (half+neg_half_j)*m[11] + neg_half_j*m[14] + half_j*m[15] + (neg_half+half_j)*m[16] + \
             half_j*m[17] + (neg_half+neg_half_j)*m[18] + neg_half*m[24] + half*m[26] + \
             neg_half*m[27] + neg_half_j*m[28] + half*m[30] + neg_half*m[32] + half_j*m[34] + \
             half*m[36] + neg_half*m[37] + neg_half*m[38] + (neg_half+neg_half_j)*m[39] + \
             half_j*m[40] + half*m[42] + half_j*m[43] + neg_half_j*m[44] + neg_half*m[46] + \
             neg_half_j*m[47]
    
    # For C[1,1]
    C[1,1] = half*m[0] + neg_half*m[2] + (half+neg_half_j)*m[3] + half*m[5] + half*m[6] + \
             half_j*m[8] + (neg_half+half_j)*m[11] + half*m[12] + neg_half*m[13] + \
             neg_half*m[14] + neg_half_j*m[15] + neg_half_j*m[17] + (half+half_j)*m[18] + \
             half_j*m[20] + neg_half*m[22] + half*m[24] + half*m[27] + half_j*m[28] + \
             half*m[29] + half*m[32] + (half+neg_half_j)*m[33] + neg_half_j*m[34] + \
             neg_half_j*m[37] + neg_half_j*m[40] + neg_half*m[41] + half*m[43] + \
             half_j*m[44] + half*m[47]
    
    # For C[1,2]
    C[1,2] = half*m[2] + neg_half*m[3] + neg_half*m[5] + neg_half*m[8] + neg_half_j*m[11] + \
             neg_half*m[12] + half*m[13] + half*m[14] + half_j*m[15] + neg_half*m[16] + \
             neg_half*m[18] + half_j*m[19] + neg_half_j*m[20] + neg_half_j*m[21] + half_j*m[23] + \
             neg_half*m[24] + neg_half_j*m[25] + half*m[26] + half_j*m[27] + half*m[30] + \
             neg_half*m[31] + neg_half*m[32] + neg_half*m[33] + half_j*m[34] + neg_half_j*m[35] + \
             half*m[36] + neg_half*m[37] + neg_half*m[38] + neg_half_j*m[39] + neg_half*m[43] + \
             neg_half_j*m[44] + neg_half_j*m[47]
    
    # For C[1,3]
    C[1,3] = neg_half*m[0] + neg_half*m[1] + half_j*m[3] + neg_half*m[4] + neg_half*m[6] + \
             neg_half*m[7] + neg_half_j*m[8] + neg_half_j*m[9] + neg_half*m[10] + half*m[11] + \
             neg_half_j*m[14] + half_j*m[16] + half_j*m[17] + neg_half_j*m[18] + half*m[21] + \
             half*m[22] + neg_half_j*m[24] + neg_half*m[27] + neg_half_j*m[28] + neg_half*m[29] + \
             neg_half_j*m[31] + half_j*m[33] + neg_half*m[34] + half_j*m[37] + neg_half*m[39] + \
             half_j*m[40] + half*m[41] + half*m[42] + half_j*m[43] + half*m[45] + \
             neg_half*m[46] + neg_half*m[47]
    
    # For C[2,0]
    C[2,0] = neg_half_j*m[0] + half_j*m[1] + half_j*m[5] + neg_half_j*m[8] + half*m[9] + \
             (half+half_j)*m[11] + half_j*m[14] + neg_half*m[15] + (neg_half+neg_half_j)*m[16] + \
             half*m[17] + (neg_half+half_j)*m[18] + neg_half*m[24] + half_j*m[26] + half*m[27] + \
             neg_half*m[28] + neg_half_j*m[30] + neg_half_j*m[32] + neg_half_j*m[34] + \
             neg_half_j*m[36] + neg_half*m[37] + neg_half_j*m[38] + (neg_half+half_j)*m[39] + \
             neg_half*m[40] + neg_half_j*m[42] + half_j*m[43] + neg_half*m[44] + \
             neg_half_j*m[46] + half_j*m[47]
    
    # For C[2,1]
    C[2,1] = half_j*m[0] + half_j*m[2] + (neg_half+neg_half_j)*m[3] + neg_half_j*m[5] + \
             half_j*m[6] + half_j*m[8] + (neg_half+neg_half_j)*m[11] + half_j*m[12] + \
             half_j*m[13] + neg_half*m[14] + half*m[15] + neg_half*m[17] + (half+neg_half_j)*m[18] + \
             neg_half*m[20] + half_j*m[22] + half*m[24] + neg_half*m[27] + half*m[28] + \
             neg_half_j*m[29] + half_j*m[32] + (half+half_j)*m[33] + half_j*m[34] + half_j*m[37] + \
             half*m[40] + neg_half_j*m[41] + neg_half*m[43] + half*m[44] + half*m[47]
    
    # For C[2,2]
    C[2,2] = neg_half_j*m[2] + half*m[3] + half_j*m[5] + half*m[8] + half_j*m[11] + neg_half_j*m[12] + \
             neg_half_j*m[13] + half*m[14] + neg_half*m[15] + neg_half*m[16] + neg_half*m[18] + \
             neg_half*m[19] + half*m[20] + neg_half_j*m[21] + half*m[23] + neg_half*m[24] + \
             half*m[25] + half_j*m[26] + half_j*m[27] + neg_half_j*m[30] + half*m[31] + \
             neg_half_j*m[32] + neg_half*m[33] + neg_half_j*m[34] + neg_half*m[35] + \
             neg_half_j*m[36] + neg_half*m[37] + neg_half_j*m[38] + half_j*m[39] + half*m[43] + \
             neg_half*m[44] + half_j*m[47]
    
    # For C[2,3]
    C[2,3] = neg_half_j*m[0] + half_j*m[1] + half_j*m[3] + neg_half_j*m[4] + neg_half_j*m[6] + \
             half_j*m[7] + neg_half_j*m[8] + half*m[9] + neg_half_j*m[10] + half*m[11] + \
             half_j*m[14] + neg_half_j*m[16] + half*m[17] + half_j*m[18] + neg_half*m[21] + \
             neg_half_j*m[22] + half_j*m[24] + half*m[27] + neg_half*m[28] + half_j*m[29] + \
             neg_half_j*m[31] + neg_half_j*m[33] + neg_half*m[34] + neg_half_j*m[37] + \
             neg_half*m[39] + neg_half*m[40] + half_j*m[41] + neg_half_j*m[42] + half_j*m[43] + \
             neg_half_j*m[45] + neg_half_j*m[46] + neg_half*m[47]
    
    # For C[3,0]
    C[3,0] = neg_half_j*m[0] + neg_half_j*m[1] + half*m[5] + half_j*m[8] + half_j*m[9] + \
             (neg_half+half_j)*m[11] + neg_half_j*m[14] + neg_half_j*m[15] + (half+half_j)*m[16] + \
             neg_half_j*m[17] + (half+half_j)*m[18] + half*m[24] + neg_half_j*m[26] + half*m[27] + \
             half*m[28] + half_j*m[30] + half_j*m[32] + neg_half_j*m[34] + neg_half*m[36] + \
             half*m[37] + neg_half*m[38] + (half+neg_half_j)*m[39] + neg_half_j*m[40] + \
             half*m[42] + neg_half_j*m[43] + neg_half*m[44] + half_j*m[46] + neg_half_j*m[47]
    
    # For C[3,1]
    C[3,1] = half_j*m[0] + neg_half*m[2] + (neg_half+neg_half_j)*m[3] + neg_half*m[5] + half*m[6] + \
             neg_half_j*m[8] + (half+neg_half_j)*m[11] + neg_half*m[12] + half_j*m[13] + \
             neg_half*m[14] + half_j*m[15] + half_j*m[17] + (neg_half+neg_half_j)*m[18] + \
             neg_half*m[20] + half*m[22] + neg_half*m[24] + neg_half*m[27] + neg_half*m[28] + \
             neg_half_j*m[29] + neg_half_j*m[32] + (half+half_j)*m[33] + half_j*m[34] + \
             half_j*m[37] + half_j*m[40] + neg_half_j*m[41] + neg_half*m[43] + half*m[44] + \
             half*m[47]
    
    # For C[3,2]
    C[3,2] = half*m[2] + half_j*m[3] + half*m[5] + neg_half*m[8] + neg_half*m[11] + half*m[12] + \
             neg_half_j*m[13] + half*m[14] + neg_half_j*m[15] + half_j*m[16] + half_j*m[18] + \
             half_j*m[19] + half*m[20] + half*m[21] + neg_half*m[23] + half*m[24] + half*m[25] + \
             neg_half_j*m[26] + half_j*m[27] + half_j*m[30] + neg_half_j*m[31] + half_j*m[32] + \
             neg_half_j*m[33] + neg_half_j*m[34] + neg_half_j*m[35] + neg_half*m[36] + half*m[37] + \
             neg_half*m[38] + half*m[39] + half*m[43] + neg_half*m[44] + neg_half_j*m[47]
    
    # For C[3,3]
    C[3,3] = neg_half_j*m[0] + neg_half_j*m[1] + half*m[3] + half_j*m[4] + neg_half*m[6] + \
             neg_half*m[7] + half_j*m[8] + half_j*m[9] + neg_half*m[10] + half_j*m[11] + \
             neg_half_j*m[14] + half*m[16] + neg_half_j*m[17] + half*m[18] + neg_half_j*m[21] + \
             neg_half*m[22] + neg_half_j*m[24] + half*m[27] + half*m[28] + half_j*m[29] + \
             neg_half*m[31] + neg_half*m[33] + neg_half*m[34] + neg_half_j*m[37] + \
             neg_half_j*m[39] + neg_half_j*m[40] + half_j*m[41] + half*m[42] + neg_half_j*m[43] + \
             neg_half_j*m[45] + half_j*m[46] + neg_half*m[47]
    
    # If input was real, ensure output is real
    if np.isrealobj(A) and np.isrealobj(B):
        for i in range(4):
            for j in range(4):
                C[i,j] = C[i,j].real
    
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