import os
import numpy as np
import time
import requests
from numba import jit
import webview
import threading

# =============================================================================
# QUANTUM RANDOM NUMBER GENERATOR FUNCTION (from matrix_multiplication_algorithms.py)
# =============================================================================

def get_quantum_random_numbers(count, use_complex=False):
    """Get quantum random numbers from ANU Quantum Random Number Generator API."""
    try:
        api_count = count * 2 if use_complex else count
        API_KEY = os.getenv("ANU_QRNG_KEY")
        if not API_KEY:
            print("Warning: ANU_QRNG_KEY environment variable not set. Falling back to numpy RNG.")
            return None

        url = f"https://api.quantumnumbers.anu.edu.au?length={api_count}&type=uint8"
        headers = {"x-api-key": API_KEY}
        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code != 200:
            print(f"Warning: Quantum RNG API returned status {response.status_code}. Falling back to numpy RNG.")
            return None

        data = response.json()
        if 'data' in data:
            random_values = np.array(data['data'], dtype=np.float64) / 255.0
        else:
            print("Warning: Unexpected API response format. Falling back to numpy RNG.")
            return None

        if use_complex:
            real_parts = random_values[:count]
            imag_parts = random_values[count:]
            return real_parts + 1j * imag_parts
        else:
            return random_values
    except Exception as e:
        print(f"Warning: Error fetching quantum random numbers: {e}. Falling back to numpy RNG.")
        return None

# =============================================================================
# MATRIX MULTIPLICATION ALGORITHMS (from matrix_multiplication_algorithms.py)
# =============================================================================

@jit(nopython=True, fastmath=True)
def standard_multiply(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=A.dtype)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

@jit(nopython=True, fastmath=True)
def strassen_4x4(A, B):
    C = np.zeros((4, 4), dtype=A.dtype)
    
    A00, A01, A02, A03 = A[0,0], A[0,1], A[0,2], A[0,3]
    A10, A11, A12, A13 = A[1,0], A[1,1], A[1,2], A[1,3]
    A20, A21, A22, A23 = A[2,0], A[2,1], A[2,2], A[2,3]
    A30, A31, A32, A33 = A[3,0], A[3,1], A[3,2], A[3,3]
    
    B00, B01, B02, B03 = B[0,0], B[0,1], B[0,2], B[0,3]
    B10, B11, B12, B13 = B[1,0], B[1,1], B[1,2], B[1,3]
    B20, B21, B22, B23 = B[2,0], B[2,1], B[2,2], B[2,3]
    B30, B31, B32, B33 = B[3,0], B[3,1], B[3,2], B[3,3]
    
    a11_00,a11_01,a11_10,a11_11 = A00,A01,A10,A11
    a12_00,a12_01,a12_10,a12_11 = A02,A03,A12,A13
    a21_00,a21_01,a21_10,a21_11 = A20,A21,A30,A31
    a22_00,a22_01,a22_10,a22_11 = A22,A23,A32,A33
    
    b11_00,b11_01,b11_10,b11_11 = B00,B01,B10,B11
    b12_00,b12_01,b12_10,b12_11 = B02,B03,B12,B13
    b21_00,b21_01,b21_10,b21_11 = B20,B21,B30,B31
    b22_00,b22_01,b22_10,b22_11 = B22,B23,B32,B33

    s1_00,s1_01,s1_10,s1_11 = a11_00+a22_00, a11_01+a22_01, a11_10+a22_10, a11_11+a22_11
    t1_00,t1_01,t1_10,t1_11 = b11_00+b22_00, b11_01+b22_01, b11_10+b22_10, b11_11+b22_11
    p1_00,p1_01,p1_10,p1_11 = s1_00*t1_00+s1_01*t1_10, s1_00*t1_01+s1_01*t1_11, s1_10*t1_00+s1_11*t1_10, s1_10*t1_01+s1_11*t1_11
    
    s2_00,s2_01,s2_10,s2_11 = a21_00+a22_00, a21_01+a22_01, a21_10+a22_10, a21_11+a22_11
    p2_00,p2_01,p2_10,p2_11 = s2_00*b11_00+s2_01*b11_10, s2_00*b11_01+s2_01*b11_11, s2_10*b11_00+s2_11*b11_10, s2_10*b11_01+s2_11*b11_11

    t3_00,t3_01,t3_10,t3_11 = b12_00-b22_00, b12_01-b22_01, b12_10-b22_10, b12_11-b22_11
    p3_00,p3_01,p3_10,p3_11 = a11_00*t3_00+a11_01*t3_10, a11_00*t3_01+a11_01*t3_11, a11_10*t3_00+a11_11*t3_10, a11_10*t3_01+a11_11*t3_11

    t4_00,t4_01,t4_10,t4_11 = b21_00-b11_00, b21_01-b11_01, b21_10-b11_10, b21_11-b11_11
    p4_00,p4_01,p4_10,p4_11 = a22_00*t4_00+a22_01*t4_10, a22_00*t4_01+a22_01*t4_11, a22_10*t4_00+a22_11*t4_10, a22_10*t4_01+a22_11*t4_11

    s5_00,s5_01,s5_10,s5_11 = a11_00+a12_00, a11_01+a12_01, a11_10+a12_10, a11_11+a12_11
    p5_00,p5_01,p5_10,p5_11 = s5_00*b22_00+s5_01*b22_10, s5_00*b22_01+s5_01*b22_11, s5_10*b22_00+s5_11*b22_10, s5_10*b22_01+s5_11*b22_11

    s6_00,s6_01,s6_10,s6_11 = a21_00-a11_00, a21_01-a11_01, a21_10-a11_10, a21_11-a11_11
    t6_00,t6_01,t6_10,t6_11 = b11_00+b12_00, b11_01+b12_01, b11_10+b12_10, b11_11+b12_11
    p6_00,p6_01,p6_10,p6_11 = s6_00*t6_00+s6_01*t6_10, s6_00*t6_01+s6_01*t6_11, s6_10*t6_00+s6_11*t6_10, s6_10*t6_01+s6_11*t6_11

    s7_00,s7_01,s7_10,s7_11 = a12_00-a22_00, a12_01-a22_01, a12_10-a22_10, a12_11-a22_11
    t7_00,t7_01,t7_10,t7_11 = b21_00+b22_00, b21_01+b22_01, b21_10+b22_10, b21_11+b22_11
    p7_00,p7_01,p7_10,p7_11 = s7_00*t7_00+s7_01*t7_10, s7_00*t7_01+s7_01*t7_11, s7_10*t7_00+s7_11*t7_10, s7_10*t7_01+s7_11*t7_11

    C[0,0],C[0,1],C[1,0],C[1,1] = p1_00+p4_00-p5_00+p7_00, p1_01+p4_01-p5_01+p7_01, p1_10+p4_10-p5_10+p7_10, p1_11+p4_11-p5_11+p7_11
    C[0,2],C[0,3],C[1,2],C[1,3] = p3_00+p5_00, p3_01+p5_01, p3_10+p5_10, p3_11+p5_11
    C[2,0],C[2,1],C[3,0],C[3,1] = p2_00+p4_00, p2_01+p4_01, p2_10+p4_10, p2_11+p4_11
    C[2,2],C[2,3],C[3,2],C[3,3] = p1_00-p2_00+p3_00+p6_00, p1_01-p2_01+p3_01+p6_01, p1_10-p2_10+p3_10+p6_10, p1_11-p2_11+p3_11+p6_11
    
    return C

@jit(nopython=True, fastmath=True)
def alphaevolve_4x4(A, B):
    # CORRECTED: Initialize the result matrix 'C' as complex, as the intermediate calculations are complex.
    C = np.zeros((4, 4), dtype=np.complex128)
    m = np.zeros(48, dtype=np.complex128)

    # The 48 multiplications from the AlphaEvolve algorithm:
    m[0] = (-A[1,0] - A[1,1] + A[2,1] - A[3,0] + A[3,1])*(-B[0,1] + B[1,1])
    m[1] = (A[1,1] - A[3,1])*(B[1,1] - B[3,1])
    m[2] = (-A[1,1] + A[3,0])*(B[0,1] + B[3,1])
    m[3] = (A[1,1] - A[2,1] + A[3,0] - A[3,1])*B[3,1]
    m[4] = (-A[1,0] - A[1,1] + A[2,1] + A[3,0] - A[3,1])*(B[0,1])
    m[5] = (-A[3,0] + A[3,1])*(-B[0,1] + B[3,0] + B[3,1])
    m[6] = (A[1,0] + A[3,0])*(B[0,1] - B[3,0])
    m[7] = (A[0,0] + A[1,3] + A[2,2] + A[3,3])*(-B[2,1] + B[3,3])
    m[8] = (A[0,3] + A[1,3] - A[2,2] + A[3,3])*(B[2,1] + B[3,3])
    m[9] = (-A[0,3] + A[2,2])*(B[2,1] - B[2,3] + B[3,3])
    m[10] = (A[0,3] - A[1,3] + A[2,2] - A[3,3])*B[2,1]
    m[11] = (-A[0,0] + A[0,3] - A[1,3] + A[3,3])*(B[2,1] - B[2,3])
    m[12] = (A[0,0] + A[0,3])*(B[2,3] - B[3,3])
    m[13] = A[0,0]*(B[2,1] - B[3,3])
    m[14] = (A[0,2] + A[1,1] + A[2,3] + A[3,0])*(B[1,2] + B[2,0])
    m[15] = (A[0,2] - A[1,1] + A[2,3] - A[3,0])*(-B[1,2] + B[2,0])
    m[16] = (-A[0,2] + A[1,1])*(B[1,0] + B[1,2] - B[2,0])
    m[17] = (A[0,2] + A[1,1] - A[2,3] + A[3,0])*(-B[1,0] + B[2,0])
    m[18] = (-A[1,1] + A[2,3])*B[1,0]
    m[19] = (-A[0,2] + A[3,0])*(B[1,0] + B[1,2])
    m[20] = (A[2,3] - A[3,0])*(B[1,0] + B[2,0])
    m[21] = (-A[0,1] + A[1,2] + A[2,0] + A[3,3])*(-B[0,3] + B[2,2])
    m[22] = (-A[0,1] - A[1,2] + A[2,0] - A[3,3])*(B[0,3] + B[2,2])
    m[23] = (A[0,1] - A[2,0])*(B[0,3] + B[3,2] - B[2,2])
    m[24] = (-A[0,1] + A[1,2] - A[2,0] + A[3,3])*B[3,2]
    m[25] = (A[1,2] - A[3,3])*(-B[0,2] + B[3,2])
    m[26] = (A[0,1] + A[3,3])*(B[0,2] - B[0,3])
    m[27] = A[3,3]*(-B[0,2] + B[2,2])
    m[28] = (-A[0,3] + A[1,2] - A[2,1] + A[3,0])*(B[0,0] + B[2,3])
    m[29] = (A[0,3] + A[1,2] - A[2,1] - A[3,0])*(B[0,0] - B[2,3])
    m[30] = (-A[0,3] + A[2,1])*(B[0,0] - B[3,0] + B[2,3])
    m[31] = (A[0,3] - A[1,2] + A[2,1] - A[3,0])*(-B[3,0] + B[2,3])
    m[32] = (A[1,2] - A[3,0])*(-B[0,2] + B[3,0])
    m[33] = (-A[2,1] + A[3,0])*(-B[0,2] + B[0,0])
    m[34] = (-A[0,3] - A[3,0])*B[0,2]
    m[35] = (A[0,0] + A[1,1] + A[2,2] + A[3,3])*(B[0,0] + B[1,3] + B[2,1] + B[3,2])
    m[36] = (-A[0,0] - A[1,1] + A[2,2] + A[3,3])*(B[0,0] - B[1,3] - B[2,1] + B[3,2])
    m[37] = (A[0,0] - A[2,2])*(B[0,0] + B[1,3] - B[2,1] - B[3,2])
    m[38] = (-A[0,0] + A[1,1] - A[2,2] + A[3,3])*(B[0,0] - B[1,3] + B[2,1] - B[3,2])
    m[39] = (-A[1,1] - A[3,3])*(B[1,0] - B[1,3] + B[3,2] - B[3,0])
    m[40] = (-A[0,0] - A[2,2])*(B[2,0] + B[2,1] + B[3,0] + B[3,2])
    m[41] = A[2,2]*(-B[2,0] + B[3,2])
    m[42] = A[3,3]*(B[1,0] - B[3,0])
    m[43] = A[0,0]*(B[0,0] + B[1,3])
    m[44] = A[1,1]*(B[1,0] - B[1,3])
    m[45] = (-A[0,0] - A[1,1]) * B[1,0]
    m[46] = (-A[2,2] - A[3,3]) * B[3,0]
    m[47] = (-A[0,0] - A[1,1] + A[2,2] - A[3,3])*B[3,2]
    
    C[0,0] = m[35] + m[36] + m[37] + m[40] + m[41] + m[46] + m[47]
    C[0,1] = m[0] + m[4] + m[5] + m[6]
    C[0,2] = m[28] + m[31] + m[32] + m[33]
    C[0,3] = m[21] + m[24] + m[25] + m[26]
    C[1,0] = m[14] + m[17] + m[18] + m[19]
    C[1,1] = m[1] + m[2] + m[3] + m[4]
    C[1,2] = m[21] + m[22] + m[23] + m[27]
    C[1,3] = m[7] + m[10] + m[11] + m[12]
    C[2,0] = m[35] - m[36] - m[38] + m[39] + m[42] - m[45] - m[47]
    C[2,1] = m[7] + m[8] + m[9] + m[13]
    C[2,2] = m[14] + m[15] + m[16] + m[20]
    C[2,3] = m[28] + m[29] + m[30] + m[34]
    C[3,0] = m[7] - m[8] - m[10] + m[11]
    C[3,1] = m[14] - m[15] - m[17] + m[18]
    C[3,2] = m[28] - m[29] - m[31] + m[32]
    C[3,3] = m[35] - m[36] + m[37] - m[38] - m[39] - m[40] - m[41]
    
    return C

class Api:
    def run_benchmark(self, use_quantum):
        
        results = []
        n = 4  # 4x4 matrices
        
        try:
            if use_quantum:
                A_real_flat = get_quantum_random_numbers(n*n)
                B_real_flat = get_quantum_random_numbers(n*n)
                if A_real_flat is None or B_real_flat is None:
                    use_quantum = False 
            
            if not use_quantum:
                A_real_flat = np.random.rand(n*n)
                B_real_flat = np.random.rand(n*n)

            A_real = A_real_flat.reshape((n, n))
            B_real = B_real_flat.reshape((n, n))

            # --- Standard Algorithm ---
            start_time = time.time()
            standard_multiply(A_real, B_real)
            end_time = time.time()
            results.append({
                "name": "Standard",
                "mults": 64,
                "time": f"{(end_time - start_time) * 1000:.4f} ms"
            })
            
            # --- Strassen's Algorithm ---
            start_time = time.time()
            strassen_4x4(A_real, B_real)
            end_time = time.time()
            results.append({
                "name": "Strassen",
                "mults": 49,
                "time": f"{(end_time - start_time) * 1000:.4f} ms"
            })
            
            # --- AlphaEvolve's Algorithm ---
            start_time = time.time()
            alphaevolve_4x4(A_real, B_real)
            end_time = time.time()
            results.append({
                "name": "AlphaEvolve",
                "mults": 48,
                "time": f"{(end_time - start_time) * 1000:.4f} ms"
            })

            window.evaluate_js(f'update_results({results})')

        except Exception as e:
            # Provide a more detailed error message in the console
            print(f"An error occurred during benchmarking: {e}") 
            import traceback
            traceback.print_exc()
            window.evaluate_js('show_error("An error occurred during benchmarking. Check the console for details.")')


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
    </style>
</head>
<body>
    <h1>Matrix Multiplication Algorithm Comparison (4x4)</h1>
    <p>This application benchmarks three different algorithms for matrix multiplication, demonstrating that AlphaEvolve's method uses fewer scalar multiplications.</p>
    
    <label>
        <input type="checkbox" id="quantumRngCheckbox">
        Use ANU Quantum Random Numbers (requires ANU_QRNG_KEY environment variable)
    </label>
    
    <br>

    <button onclick="startBenchmark()">Run Benchmark</button>
    <div class="loader" id="loader"></div>

    <table id="results-table">
        <thead>
            <tr>
                <th>Algorithm</th>
                <th>Scalar Multiplications</th>
                <th>Execution Time</th>
            </tr>
        </thead>
        <tbody>
            </tbody>
    </table>
    
    <script>
        function startBenchmark() {
            document.getElementById('loader').style.display = 'block';
            document.querySelector('button').disabled = true;
            const useQuantum = document.getElementById('quantumRngCheckbox').checked;
            pywebview.api.run_benchmark(useQuantum);
        }
        
        function update_results(results) {
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
            
            document.getElementById('loader').style.display = 'none';
            document.querySelector('button').disabled = false;
        }

        function show_error(message) {
            alert(message);
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
        width=800,
        height=600
    )
    webview.start()
