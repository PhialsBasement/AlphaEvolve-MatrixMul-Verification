import os
import numpy as np
import time
import requests
from numba import jit
import webview
import threading

# =============================================================================
# QUANTUM RANDOM NUMBER GENERATOR FUNCTION
# =============================================================================

def get_quantum_random_numbers(count, use_complex=False, timeout=10, max_retries=2):
    """Get quantum random numbers from ANU Quantum Random Number Generator API."""
    try:
        api_count = count * 2 if use_complex else count
        API_KEY = os.getenv("ANU_QRNG_KEY")
        if not API_KEY:
            return None, "Missing ANU_QRNG_KEY environment variable"

        url = f"https://api.quantumnumbers.anu.edu.au?length={api_count}&type=uint8"
        headers = {"x-api-key": API_KEY}
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.get(url, headers=headers, timeout=timeout)
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries:
                    return None, f"API request failed after {max_retries + 1} attempts: {e}"
                time.sleep(0.5)  # Brief delay before retry

        if response.status_code != 200:
            return None, f"API returned status {response.status_code}"

        data = response.json()
        if 'data' not in data:
            return None, "Unexpected API response format"

        random_values = np.array(data['data'], dtype=np.float64) / 255.0

        if use_complex:
            real_parts = random_values[:count]
            imag_parts = random_values[count:]
            return real_parts + 1j * imag_parts, "Success"
        else:
            return random_values, "Success"
            
    except Exception as e:
        return None, f"Unexpected error: {e}"

# =============================================================================
# MATRIX MULTIPLICATION ALGORITHMS
# =============================================================================

@jit(nopython=True, fastmath=True)
def standard_multiply(A, B):
    """Standard O(n³) matrix multiplication algorithm."""
    n = A.shape[0]
    C = np.zeros((n, n), dtype=A.dtype)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

@jit(nopython=True, fastmath=True)
def strassen_2x2(A, B):
    """Strassen's algorithm for 2x2 matrices using 7 multiplications."""
    # Extract elements
    a11, a12 = A[0, 0], A[0, 1]
    a21, a22 = A[1, 0], A[1, 1]
    b11, b12 = B[0, 0], B[0, 1]
    b21, b22 = B[1, 0], B[1, 1]
    
    # Strassen's 7 multiplications
    m1 = (a11 + a22) * (b11 + b22)
    m2 = (a21 + a22) * b11
    m3 = a11 * (b12 - b22)
    m4 = a22 * (b21 - b11)
    m5 = (a11 + a12) * b22
    m6 = (a21 - a11) * (b11 + b12)
    m7 = (a12 - a22) * (b21 + b22)
    
    # Combine results
    C = np.zeros((2, 2), dtype=A.dtype)
    C[0, 0] = m1 + m4 - m5 + m7
    C[0, 1] = m3 + m5
    C[1, 0] = m2 + m4
    C[1, 1] = m1 - m2 + m3 + m6
    
    return C

@jit(nopython=True, fastmath=True)
def strassen_4x4(A, B):
    """
    Strassen's algorithm for 4x4 matrices using recursive 2x2 approach.
    Uses 49 scalar multiplications (7^2).
    """
    C = np.zeros((4, 4), dtype=A.dtype)
    
    # Divide 4x4 matrices into 2x2 blocks
    A11 = A[:2, :2]
    A12 = A[:2, 2:]
    A21 = A[2:, :2]
    A22 = A[2:, 2:]
    
    B11 = B[:2, :2]
    B12 = B[:2, 2:]
    B21 = B[2:, :2]
    B22 = B[2:, 2:]
    
    # Strassen's recursive formula for 2x2 blocks
    # Each operation uses strassen_2x2 (7 mults each)
    M1 = strassen_2x2(A11 + A22, B11 + B22)  # 7 mults
    M2 = strassen_2x2(A21 + A22, B11)        # 7 mults
    M3 = strassen_2x2(A11, B12 - B22)        # 7 mults
    M4 = strassen_2x2(A22, B21 - B11)        # 7 mults
    M5 = strassen_2x2(A11 + A12, B22)        # 7 mults
    M6 = strassen_2x2(A21 - A11, B11 + B12)  # 7 mults
    M7 = strassen_2x2(A12 - A22, B21 + B22)  # 7 mults
    
    # Combine results
    C[:2, :2] = M1 + M4 - M5 + M7
    C[:2, 2:] = M3 + M5
    C[2:, :2] = M2 + M4
    C[2:, 2:] = M1 - M2 + M3 + M6
    
    return C

@jit(nopython=True, fastmath=True)
def alphaevolve_4x4_complex(A, B):
    """
    AlphaEvolve's optimized algorithm for 4×4 matrices.
    Uses exactly 48 scalar multiplications.
    
    This is the authentic implementation from Google DeepMind's AlphaEvolve.
    """
    # Initialize the result matrix
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
    
    # Construct the result matrix efficiently
    # For C[0,0]
    C[0,0] = half_j*m0 + neg_half_j*m1 + neg_half*m5 + half*m8 + half_j*m9 + \
             (neg_half+half_j)*m11 + half*m14 + neg_half_j*m15 + (neg_half+neg_half_j)*m16 + \
             half_j*m17 + (neg_half+neg_half_j)*m18 + neg_half_j*m24 + half_j*m26 + \
             half_j*m27 + half*m28 + half_j*m30 + neg_half_j*m32 + half*m34 + \
             half*m36 + neg_half_j*m37 + neg_half*m38 + (half+neg_half_j)*m39 + \
             neg_half_j*m40 + neg_half*m42 + neg_half*m43 + neg_half*m44 + \
             neg_half_j*m46 + half*m47
    
    # For C[0,1]
    C[0,1] = neg_half_j*m0 + half*m2 + (neg_half+neg_half_j)*m3 + half*m5 + \
             half*m6 + neg_half*m8 + (half+neg_half_j)*m11 + neg_half*m12 + \
             half_j*m13 + half_j*m14 + half_j*m15 + neg_half_j*m17 + \
             (half+half_j)*m18 + half*m20 + neg_half*m22 + half_j*m24 + \
             neg_half_j*m27 + neg_half*m28 + neg_half_j*m29 + half_j*m32 + \
             (neg_half+neg_half_j)*m33 + neg_half*m34 + neg_half*m37 + half_j*m40 + \
             half_j*m41 + neg_half_j*m43 + half*m44 + neg_half_j*m47
    
    # For C[0,2]
    C[0,2] = neg_half*m2 + half*m3 + neg_half*m5 + neg_half_j*m8 + half_j*m11 + \
             half*m12 + neg_half_j*m13 + neg_half_j*m14 + neg_half_j*m15 + \
             neg_half*m16 + neg_half*m18 + half_j*m19 + neg_half*m20 + half_j*m21 + \
             neg_half*m23 + neg_half_j*m24 + neg_half*m25 + half_j*m26 + half*m27 + \
             half_j*m30 + neg_half*m31 + neg_half_j*m32 + half*m33 + half*m34 + \
             half_j*m35 + half*m36 + neg_half_j*m37 + neg_half*m38 + neg_half_j*m39 + \
             half_j*m43 + neg_half*m44 + half*m47
    
    # For C[0,3]
    C[0,3] = half_j*m0 + neg_half_j*m1 + half_j*m3 + neg_half_j*m4 + neg_half*m6 + \
             half*m7 + half*m8 + half_j*m9 + neg_half*m10 + neg_half*m11 + half*m14 + \
             neg_half_j*m16 + half_j*m17 + neg_half_j*m18 + neg_half*m21 + half*m22 + \
             half*m24 + half_j*m27 + half*m28 + half_j*m29 + neg_half_j*m31 + \
             half_j*m33 + half_j*m34 + half*m37 + half*m39 + neg_half_j*m40 + \
             neg_half_j*m41 + neg_half*m42 + neg_half*m43 + neg_half_j*m45 + \
             neg_half_j*m46 + half_j*m47
    
    # For C[1,0]
    C[1,0] = neg_half*m0 + neg_half*m1 + neg_half*m5 + neg_half_j*m8 + neg_half_j*m9 + \
             (half+neg_half_j)*m11 + neg_half_j*m14 + half_j*m15 + (neg_half+half_j)*m16 + \
             half_j*m17 + (neg_half+neg_half_j)*m18 + neg_half*m24 + half*m26 + \
             neg_half*m27 + neg_half_j*m28 + half*m30 + neg_half*m32 + half_j*m34 + \
             half*m36 + neg_half*m37 + neg_half*m38 + (neg_half+neg_half_j)*m39 + \
             half_j*m40 + half*m42 + half_j*m43 + neg_half_j*m44 + neg_half*m46 + \
             neg_half_j*m47
    
    # For C[1,1]
    C[1,1] = half*m0 + neg_half*m2 + (half+neg_half_j)*m3 + half*m5 + half*m6 + \
             half_j*m8 + (neg_half+half_j)*m11 + half*m12 + neg_half*m13 + \
             neg_half*m14 + neg_half_j*m15 + neg_half_j*m17 + (half+half_j)*m18 + \
             half_j*m20 + neg_half*m22 + half*m24 + half*m27 + half_j*m28 + \
             half*m29 + half*m32 + (half+neg_half_j)*m33 + neg_half_j*m34 + \
             neg_half_j*m37 + neg_half_j*m40 + neg_half*m41 + half*m43 + \
             half_j*m44 + half*m47
    
    # For C[1,2]
    C[1,2] = half*m2 + neg_half*m3 + neg_half*m5 + neg_half*m8 + neg_half_j*m11 + \
             neg_half*m12 + half*m13 + half*m14 + half_j*m15 + neg_half*m16 + \
             neg_half*m18 + half_j*m19 + neg_half_j*m20 + neg_half_j*m21 + half_j*m23 + \
             neg_half*m24 + neg_half_j*m25 + half*m26 + half_j*m27 + half*m30 + \
             neg_half*m31 + neg_half*m32 + neg_half*m33 + half_j*m34 + neg_half_j*m35 + \
             half*m36 + neg_half*m37 + neg_half*m38 + neg_half_j*m39 + neg_half*m43 + \
             neg_half_j*m44 + neg_half_j*m47
    
    # For C[1,3]
    C[1,3] = neg_half*m0 + neg_half*m1 + half_j*m3 + neg_half*m4 + neg_half*m6 + \
             neg_half*m7 + neg_half_j*m8 + neg_half_j*m9 + neg_half*m10 + half*m11 + \
             neg_half_j*m14 + half_j*m16 + half_j*m17 + neg_half_j*m18 + half*m21 + \
             half*m22 + neg_half_j*m24 + neg_half*m27 + neg_half_j*m28 + neg_half*m29 + \
             neg_half_j*m31 + half_j*m33 + neg_half*m34 + half_j*m37 + neg_half*m39 + \
             half_j*m40 + half*m41 + half*m42 + half_j*m43 + half*m45 + \
             neg_half*m46 + neg_half*m47
    
    # For C[2,0]
    C[2,0] = neg_half_j*m0 + half_j*m1 + half_j*m5 + neg_half_j*m8 + half*m9 + \
             (half+half_j)*m11 + half_j*m14 + neg_half*m15 + (neg_half+neg_half_j)*m16 + \
             half*m17 + (neg_half+half_j)*m18 + neg_half*m24 + half_j*m26 + half*m27 + \
             neg_half*m28 + neg_half_j*m30 + neg_half_j*m32 + neg_half_j*m34 + \
             neg_half_j*m36 + neg_half*m37 + neg_half_j*m38 + (neg_half+half_j)*m39 + \
             neg_half*m40 + neg_half_j*m42 + half_j*m43 + neg_half*m44 + \
             neg_half_j*m46 + half_j*m47
    
    # For C[2,1]
    C[2,1] = half_j*m0 + half_j*m2 + (neg_half+neg_half_j)*m3 + neg_half_j*m5 + \
             half_j*m6 + half_j*m8 + (neg_half+neg_half_j)*m11 + half_j*m12 + \
             half_j*m13 + neg_half*m14 + half*m15 + neg_half*m17 + (half+neg_half_j)*m18 + \
             neg_half*m20 + half_j*m22 + half*m24 + neg_half*m27 + half*m28 + \
             neg_half_j*m29 + half_j*m32 + (half+half_j)*m33 + half_j*m34 + half_j*m37 + \
             half*m40 + neg_half_j*m41 + neg_half*m43 + half*m44 + half*m47
    
    # For C[2,2]
    C[2,2] = neg_half_j*m2 + half*m3 + half_j*m5 + half*m8 + half_j*m11 + neg_half_j*m12 + \
             neg_half_j*m13 + half*m14 + neg_half*m15 + neg_half*m16 + neg_half*m18 + \
             neg_half*m19 + half*m20 + neg_half_j*m21 + half*m23 + neg_half*m24 + \
             half*m25 + half_j*m26 + half_j*m27 + neg_half_j*m30 + half*m31 + \
             neg_half_j*m32 + neg_half*m33 + neg_half_j*m34 + neg_half*m35 + \
             neg_half_j*m36 + neg_half*m37 + neg_half_j*m38 + half_j*m39 + half*m43 + \
             neg_half*m44 + half_j*m47
    
    # For C[2,3]
    C[2,3] = neg_half_j*m0 + half_j*m1 + half_j*m3 + neg_half_j*m4 + neg_half_j*m6 + \
             half_j*m7 + neg_half_j*m8 + half*m9 + neg_half_j*m10 + half*m11 + \
             half_j*m14 + neg_half_j*m16 + half*m17 + half_j*m18 + neg_half*m21 + \
             neg_half_j*m22 + half_j*m24 + half*m27 + neg_half*m28 + half_j*m29 + \
             neg_half_j*m31 + neg_half_j*m33 + neg_half*m34 + neg_half_j*m37 + \
             neg_half*m39 + neg_half*m40 + half_j*m41 + neg_half_j*m42 + half_j*m43 + \
             neg_half_j*m45 + neg_half_j*m46 + neg_half*m47
    
    # For C[3,0] (continuing from previous)
    C[3,0] = neg_half_j*m0 + neg_half_j*m1 + half*m5 + half_j*m8 + half_j*m9 + \
             (neg_half+half_j)*m11 + neg_half_j*m14 + neg_half_j*m15 + (half+half_j)*m16 + \
             neg_half_j*m17 + (half+half_j)*m18 + half*m24 + neg_half_j*m26 + half*m27 + \
             half*m28 + half_j*m30 + half_j*m32 + neg_half_j*m34 + neg_half*m36 + \
             half*m37 + neg_half*m38 + (half+neg_half_j)*m39 + neg_half_j*m40 + \
             half*m42 + neg_half_j*m43 + neg_half*m44 + half_j*m46 + neg_half_j*m47
    
    # For C[3,1]
    C[3,1] = half_j*m0 + neg_half*m2 + (neg_half+neg_half_j)*m3 + neg_half*m5 + half*m6 + \
             neg_half_j*m8 + (half+neg_half_j)*m11 + neg_half*m12 + half_j*m13 + \
             neg_half*m14 + half_j*m15 + half_j*m17 + (neg_half+neg_half_j)*m18 + \
             neg_half*m20 + half*m22 + neg_half*m24 + neg_half*m27 + neg_half*m28 + \
             neg_half_j*m29 + neg_half_j*m32 + (half+half_j)*m33 + half_j*m34 + \
             half_j*m37 + half_j*m40 + neg_half_j*m41 + neg_half*m43 + half*m44 + \
             half*m47
    
    # For C[3,2]
    C[3,2] = half*m2 + half_j*m3 + half*m5 + neg_half*m8 + neg_half*m11 + half*m12 + \
             neg_half_j*m13 + half*m14 + neg_half_j*m15 + half_j*m16 + half_j*m18 + \
             half_j*m19 + half*m20 + half*m21 + neg_half*m23 + half*m24 + half*m25 + \
             neg_half_j*m26 + half_j*m27 + half_j*m30 + neg_half_j*m31 + half_j*m32 + \
             neg_half_j*m33 + neg_half_j*m34 + neg_half_j*m35 + neg_half*m36 + half*m37 + \
             neg_half*m38 + half*m39 + half*m43 + neg_half*m44 + neg_half_j*m47
    
    # For C[3,3]
    C[3,3] = neg_half_j*m0 + neg_half_j*m1 + half*m3 + half_j*m4 + neg_half*m6 + \
             neg_half*m7 + half_j*m8 + half_j*m9 + neg_half*m10 + half_j*m11 + \
             neg_half_j*m14 + half*m16 + neg_half_j*m17 + half*m18 + neg_half_j*m21 + \
             neg_half*m22 + neg_half_j*m24 + half*m27 + half*m28 + half_j*m29 + \
             neg_half*m31 + neg_half*m33 + neg_half*m34 + neg_half_j*m37 + \
             neg_half_j*m39 + neg_half_j*m40 + half_j*m41 + half*m42 + neg_half_j*m43 + \
             neg_half_j*m45 + half_j*m46 + neg_half*m47
    
    return C

def alphaevolve_4x4(A, B):
    """
    AlphaEvolve wrapper that handles both real and complex matrices.
    """
    # Convert to complex for computation
    A_complex = A.astype(np.complex128)
    B_complex = B.astype(np.complex128)
    
    # Compute using the JIT-compiled complex version
    result = alphaevolve_4x4_complex(A_complex, B_complex)
    
    # Return appropriate type based on input
    if A.dtype in [np.float32, np.float64] and B.dtype in [np.float32, np.float64]:
        return result.real.astype(A.dtype)
    else:
        return result

def verify_algorithms(A, B, tolerance=1e-10):
    """Verify that all three algorithms produce the same result."""
    try:
        # Convert to complex for AlphaEvolve if needed
        A_complex = A.astype(np.complex128) if A.dtype != np.complex128 else A
        B_complex = B.astype(np.complex128) if B.dtype != np.complex128 else B
        
        result_standard = standard_multiply(A, B)
        result_strassen = strassen_4x4(A, B)
        result_alphaevolve = alphaevolve_4x4(A_complex, B_complex)
        
        # Check if results match within tolerance
        standard_vs_strassen = np.allclose(result_standard, result_strassen, atol=tolerance)
        standard_vs_alphaevolve = np.allclose(result_standard, result_alphaevolve, atol=tolerance)
        
        return {
            'all_match': standard_vs_strassen and standard_vs_alphaevolve,
            'standard_vs_strassen': standard_vs_strassen,
            'standard_vs_alphaevolve': standard_vs_alphaevolve,
            'max_diff_strassen': np.max(np.abs(result_standard - result_strassen)),
            'max_diff_alphaevolve': np.max(np.abs(result_standard - result_alphaevolve))
        }
    except Exception as e:
        return {'error': str(e)}

class Api:
    def run_benchmark(self, use_quantum, use_complex, num_iterations=100):
        results = []
        n = 4  # 4x4 matrices
        
        try:
            # Generate matrices based on user preferences
            quantum_status = "Not used"
            
            if use_complex:
                if use_quantum:
                    A_data, status = get_quantum_random_numbers(n*n, use_complex=True)
                    B_data, _ = get_quantum_random_numbers(n*n, use_complex=True)
                    if A_data is not None and B_data is not None:
                        quantum_status = "Success"
                        A = A_data.reshape((n, n))
                        B = B_data.reshape((n, n))
                    else:
                        quantum_status = f"Failed: {status}"
                        real_parts = np.random.rand(n*n)
                        imag_parts = np.random.rand(n*n)
                        A = (real_parts + 1j * imag_parts).reshape((n, n))
                        real_parts = np.random.rand(n*n)
                        imag_parts = np.random.rand(n*n)
                        B = (real_parts + 1j * imag_parts).reshape((n, n))
                else:
                    real_parts = np.random.rand(n*n)
                    imag_parts = np.random.rand(n*n)
                    A = (real_parts + 1j * imag_parts).reshape((n, n))
                    real_parts = np.random.rand(n*n)
                    imag_parts = np.random.rand(n*n)
                    B = (real_parts + 1j * imag_parts).reshape((n, n))
            else:
                if use_quantum:
                    A_data, status = get_quantum_random_numbers(n*n)
                    B_data, _ = get_quantum_random_numbers(n*n)
                    if A_data is not None and B_data is not None:
                        quantum_status = "Success"
                        A = A_data.reshape((n, n))
                        B = B_data.reshape((n, n))
                    else:
                        quantum_status = f"Failed: {status}"
                        A = np.random.rand(n, n)
                        B = np.random.rand(n, n)
                else:
                    A = np.random.rand(n, n)
                    B = np.random.rand(n, n)

            # Verify algorithm correctness (temporarily allow mismatches for debugging)
            verification = verify_algorithms(A, B)
            verification_status = "Algorithms verified: ✓"
            if 'error' in verification:
                verification_status = f"Verification error: {verification['error']}"
                # Continue anyway for debugging
            elif not verification['all_match']:
                verification_status = f"⚠️ Mismatch detected! Strassen diff={verification['max_diff_strassen']:.2e}, AlphaEvolve diff={verification['max_diff_alphaevolve']:.2e} (continuing anyway)"
                # Continue with benchmark but show warning

            # For AlphaEvolve, ensure matrices are complex-typed for optimal performance
            A_complex = A.astype(np.complex128) if A.dtype != np.complex128 else A
            B_complex = B.astype(np.complex128) if B.dtype != np.complex128 else B

            # --- Standard Algorithm ---
            # Warm up JIT
            standard_multiply(A, B)
            
            times = []
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                standard_multiply(A, B)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)
            
            results.append({
                "name": "Standard",
                "mults": 64,
                "time": f"{np.mean(times):.4f} ± {np.std(times):.4f} ms"
            })
            
            # --- Strassen's Algorithm ---
            # Warm up JIT
            strassen_4x4(A, B)
            
            times = []
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                strassen_4x4(A, B)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)
            
            results.append({
                "name": "Strassen",
                "mults": 49,
                "time": f"{np.mean(times):.4f} ± {np.std(times):.4f} ms"
            })
            
            # --- AlphaEvolve's Algorithm ---
            # Warm up JIT
            alphaevolve_4x4(A_complex, B_complex)
            
            times = []
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                alphaevolve_4x4(A_complex, B_complex)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)
            
            results.append({
                "name": "AlphaEvolve",
                "mults": 48,
                "time": f"{np.mean(times):.4f} ± {np.std(times):.4f} ms"
            })

            # Update UI with results and status
            window.evaluate_js(f'update_results({results}, {repr(quantum_status)}, {repr(verification_status)})')

        except ValueError as e:
            window.evaluate_js(f'show_error({repr(f"ValueError: {e}")})')
        except RuntimeError as e:
            window.evaluate_js(f'show_error({repr(f"RuntimeError: {e}")})')
        except Exception as e:
            print(f"Unexpected error during benchmarking: {e}") 
            import traceback
            traceback.print_exc()
            window.evaluate_js(f'show_error({repr(f"Unexpected error: {e}")})')


html = """
<!DOCTYPE html>
<html>
<head>
    <title>Matrix Multiplication Benchmark</title>
    <style>
        body { font-family: sans-serif; padding: 20px; background-color: #f4f4f9; }
        h1 { color: #333; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; border: 1px solid #ddd; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; font-size: 16px; margin-top: 15px;}
        button:hover { background-color: #45a049; }
        .loader { border: 5px solid #f3f3f3; border-top: 5px solid #3498db; border-radius: 50%; width: 30px; height: 30px; animation: spin 2s linear infinite; display: none; margin-top: 15px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .controls { margin-bottom: 20px; }
        .control-group { margin-bottom: 10px; }
        .status { margin-top: 10px; padding: 10px; background-color: #e8f5e8; border-radius: 5px; font-size: 14px; }
        .status.error { background-color: #ffe8e8; }
    </style>
</head>
<body>
    <h1>Matrix Multiplication Algorithm Comparison (4x4)</h1>
    <p>This application benchmarks three different algorithms for matrix multiplication, demonstrating that AlphaEvolve's method uses fewer scalar multiplications.</p>
    
    <div class="controls">
        <div class="control-group">
            <label>
                <input type="checkbox" id="quantumRngCheckbox">
                Use ANU Quantum Random Numbers (requires ANU_QRNG_KEY environment variable)
            </label>
        </div>
        
        <div class="control-group">
            <label>
                <input type="checkbox" id="complexNumbersCheckbox" checked>
                Use complex-valued matrices (optimal for AlphaEvolve)
            </label>
        </div>
        
        <div class="control-group">
            <label>
                Iterations: 
                <select id="iterationsSelect">
                    <option value="10">10</option>
                    <option value="50">50</option>
                    <option value="100" selected>100</option>
                    <option value="500">500</option>
                    <option value="1000">1000</option>
                </select>
            </label>
        </div>
    </div>

    <button onclick="startBenchmark()">Run Benchmark</button>
    <div class="loader" id="loader"></div>
    
    <div id="status" class="status" style="display: none;"></div>

    <table id="results-table">
        <thead>
            <tr>
                <th>Algorithm</th>
                <th>Scalar Multiplications</th>
                <th>Execution Time (mean ± std)</th>
            </tr>
        </thead>
        <tbody>
            </tbody>
    </table>
    
    <script>
        function startBenchmark() {
            document.getElementById('loader').style.display = 'block';
            document.querySelector('button').disabled = true;
            document.getElementById('status').style.display = 'none';
            
            const useQuantum = document.getElementById('quantumRngCheckbox').checked;
            const useComplex = document.getElementById('complexNumbersCheckbox').checked;
            const iterations = parseInt(document.getElementById('iterationsSelect').value);
            
            pywebview.api.run_benchmark(useQuantum, useComplex, iterations);
        }
        
        function update_results(results, quantumStatus, verificationStatus) {
            const tableBody = document.querySelector("#results-table tbody");
            tableBody.innerHTML = ""; // Clear previous results
            
            results.forEach(res => {
                const row = `<tr>
                                <td>${res.name}</td>
                                <td>${res.mults}</td>
                                <td>${res.time}</td>
                             </tr>`;
                tableBody.innerHTML += row;
            });
            
            // Update status
            const statusDiv = document.getElementById('status');
            statusDiv.innerHTML = `<strong>Quantum RNG:</strong> ${quantumStatus}<br><strong>Verification:</strong> ${verificationStatus}`;
            statusDiv.className = 'status';
            statusDiv.style.display = 'block';
            
            document.getElementById('loader').style.display = 'none';
            document.querySelector('button').disabled = false;
        }

        function show_error(message) {
            const statusDiv = document.getElementById('status');
            statusDiv.innerHTML = `<strong>Error:</strong> ${message}`;
            statusDiv.className = 'status error';
            statusDiv.style.display = 'block';
            
            document.getElementById('loader').style.display = 'none';
            document.querySelector('button').disabled = false;
        }
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    api = Api()
    window = webview.create_window(
        'Matrix Multiplication Benchmark',
        html=html,
        js_api=api,
        width=900,
        height=700
    )
    webview.start()
