# AlphaEvolveVerify

Verification of Google DeepMind's AlphaEvolve 48-multiplication matrix algorithm, a breakthrough in matrix multiplication after 56 years.

## Overview

This repository contains code to verify and optimize the groundbreaking 4×4 matrix multiplication algorithm discovered by Google DeepMind's AlphaEvolve. In 2025, AlphaEvolve found a method to multiply 4×4 matrices using only 48 scalar multiplications, improving on Strassen's algorithm (49 multiplications) for the first time since 1969. And while yes, my implementation of AlphaEvolves Algo is slower than Strassens algo, it is a PoC that this algo does in fact work as advertised by google(in 48 steps).

The repository includes:

1. **Matrix Multiplication Verification (MMV)** - Code to test and benchmark the AlphaEvolve algorithm against standard and Strassen's algorithms
2. **Tensor Decomposition Analyzer (TDA)** - A tool to reverse-engineer the tensor decomposition into an optimized direct implementation

## The Breakthrough

Matrix multiplication is one of the most fundamental operations in computing:
- Standard algorithm: 64 multiplications (for 4×4 matrices)
- Strassen's algorithm (1969): 49 multiplications
- AlphaEvolve's algorithm (2025): 48 multiplications

While a single multiplication reduction might seem minor, it represents the first improvement to Strassen's algorithm for 4×4 complex matrices in over five decades and has significant implications for larger matrices when applied recursively.

## Matrix Multiplication Verification (MMV)

The `matrix_multiplication_algorithms.py` file contains implementations of three algorithms:
- Standard matrix multiplication (64 multiplications)
- Strassen's algorithm (49 multiplications)
- AlphaEvolve's algorithm (48 multiplications)

Features:
- Accuracy verification with quantum random matrices (via ANU Quantum RNG API)
- Performance benchmarking
- Support for both real and complex matrices

Usage:
```bash
python matrix_multiplication_algorithms.py
```

## Tensor Decomposition Analyzer (TDA)

The `decomposition_analyzer.py` script reverse-engineers the tensor decomposition provided by AlphaEvolve into an optimized direct implementation.

Features:
- Converts mathematical tensor representation to readable Python code
- Generates an optimized direct implementation without loops
- Significantly improves performance and numerical stability

Usage:
```bash
python decomposition_analyzer.py
```

## Results

Our verification confirms AlphaEvolve's breakthrough and demonstrates:

1. **Correctness**: The algorithm produces accurate results for both real and complex matrices
2. **Numerical Stability**: Optimized implementation achieves machine precision (error ~10^-16)
3. **Performance**: The optimized direct implementation outperforms the tensor-based approach

## Requirements

- Python 3.6+
- NumPy
- Requests (for quantum RNG)

## Installation

```bash
git clone https://github.com/PhialsBasement/AlphaEvolve-MatrixMul-Verification.git
cd AlphaEvolve-MatrixMul-Verification
pip install numpy requests
```
![image](https://github.com/user-attachments/assets/692fbfef-60b0-46c7-8528-85d13e521e31)


## Acknowledgements

- Google DeepMind for the AlphaEvolve algorithm and tensor decomposition
- The Australian National University for the Quantum Random Numbers API
- Claude (Anthropic) for assistance in reverse engineering

## Contributing

Contributions are welcome! Areas for improvement include:
- More efficient implementations of the algorithms
- Applications to larger matrices
- Integration with popular numerical libraries
(and obviously if theres something wrong with the algo pls let me know or submit a PR request) 

## License

MIT License

## Citation
```
@techreport{alphaevolve,
      author={Novikov, Alexander and V\~{u}, Ng\^{a}n and Eisenberger, Marvin and Dupont, Emilien and Huang, Po-Sen and Wagner, Adam Zsolt and Shirobokov, Sergey and Kozlovskii, Borislav and Ruiz, Francisco J. R. and Mehrabian, Abbas and Kumar, M. Pawan and See, Abigail and Chaudhuri, Swarat and Holland, George and Davies, Alex and Nowozin, Sebastian and Kohli, Pushmeet and Balog, Matej},
      title={Alpha{E}volve: A coding agent for scientific and algorithmic discovery},
      year={2025},
      month={05},
      url={https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf},
}
```
