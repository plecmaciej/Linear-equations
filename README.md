# Solving Systems of Linear Equations

## Project Overview

This project focuses on solving systems of linear equations of the form `Ax = b` using both **iterative** and **direct** numerical methods. Implemented in Python, it leverages matrix libraries to analyze the accuracy and performance of each approach.

## Implemented Methods

- **Iterative Methods**
  - Jacobi Method
  - Gauss-Seidel Method

- **Direct Method**
  - LU Factorization

Each method is evaluated based on:
- Accuracy (residual norm)
- Number of iterations
- Execution time

##  Task Breakdown

###  Task A – Matrix & Vector Generation

- Generate a square matrix `A` of size `N x N`, where `N = 955`.
- Diagonal element `a1 = 12` (based on index number).
- Vector `b` defined as: b_n = sin(4 * n)

###  Task B – Iterative Methods: Jacobi & Gauss-Seidel

- Gauss-Seidel converges in fewer iterations (16 vs 23 for Jacobi).
- Execution times are similar, with Gauss-Seidel being slightly faster.

###  Task C – Residual Norm Analysis

- Modified matrix parameters to analyze convergence.
- Residuals diverge → both methods fail to converge.

###  Task D – LU Factorization

- Implemented a direct method using LU decomposition.
- Achieved a very low residual norm → highly accurate.

###  Task E – Time vs Matrix Size

- Compared execution times for different sizes of `N = 100, 500, ..., 3000`.
- Iterative methods are generally faster.
- LU becomes inefficient for large `N`, but remains robust.

###  Task F – Summary

- Iterative methods are efficient (complexity O(n²)), but may diverge.
- LU factorization is more accurate (O(n³)), suitable for challenging systems.

##  Conclusion

Iterative methods like Jacobi and Gauss-Seidel provide fast solutions for well-conditioned systems. However, LU factorization offers better stability and accuracy, especially when iterative approaches fail to converge.

