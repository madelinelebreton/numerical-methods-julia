# MECHTRON 3X03 Assignment 1
# Author: Madeline LeBreton using resources from course content
# 16/10/2025

using LinearAlgebra

"""
Implement the function signatures defined here in your solution file <FIRSTNAME>_LASTNAME_a1.jl.
DO NOT MODIFY THE SIGNATURES.

Use test_a1.jl to test your solutions.
"""


"""
Perform backward substitution to solve the system U*x = b for x.
Inputs:
    U: an nxn upper diagonal square matrix (assume the diagonal is nonzero)
    b: an nx1 vector
Output: 
    x: the solution to U*x = b
"""
function backward_substitution(U, b)
    # solve for the values of x, starting with the last one
    n = size(b,1) # number of rows
    x = zeros(eltype(b), n) # create variable to store elements of x

    for k in n:-1:1 # for each row (bottom-up), solve for x_k
        x[k] = b[k] # start with kth value in b 
        for j in k+1:n # for each column (element [n,n] doesn't require this step)
            x[k] -= U[k,j] * x[j] # subtract each value of x that has previously been solved for 
        end

        x[k] /= U[k,k] # divide by element on diagonal
    end
    return x
end

"""
Perform forward substitution to solve the system L*x = b for x.
Inputs:
    L: an nxn lower diagonal square matrix (assume the diagonal is nonzero)
    b: an nx1 vector
Output: 
    x: the solution to L*x = b
"""
function forward_substitution(L, b)

    n = size(b, 1) # number of rows
    x = zeros(eltype(b), n) # create variable to store values of x
    # working top-down
    for k in 1:n
        x[k] = b[k] # start with kth value of b 
        for j in 1:k-1 # last entry doesn't require this step
            x[k] -= L[k,j] * x[j] # subtract values of x that have already been solved for
        end
        x[k] /= L[k,k]
    end
    return x
end

"""
Gaussian elimination without pivoting. 
You may assume A_in has full rank and never has a zero pivot. 

Input:
    A: a full rank nxn matrix with no zero pivots.
    b: a nx1 vector

Output:
    A_out: the input matrix A in row-echelon form
    b_out: the input vector b after having undergone the transformations putting A into row-echelon form
"""
function gaussian_elimination(A, b)
    # “clear” each column starting with the 1st by multiplying and subtracting rows
    n = size(A,1);
    for k in 1:n-1 # each row except the last 
        for i in k+1:n # remove a multiple of the kth row
            m_ik = A[i,k] / A[k,k] # use multiplier m to clear kth column
            
            # update A and b 
            for j in k:n # for each row below
                A[i,j] = A[i,j] - A[k,j] * m_ik # update the row by subtracting a multiple of the jth row
            end
            b[i] = b[i] - m_ik * b[k] # update b with same transformation by multiple m_ik
            
        end
    end
    
    # return the updated variables in row-echelon form
    A_out = copy(A) 
    b_out = copy(b)
    return A_out, b_out
end

"""
LU decomposition with partial pivoting. 

Input: 
    A_in: an nxn full-rank matrix

Returns:
    L: nxn lower triangular matrix
    U: nxn upper triangular matrix
    p: permuted vector of indices 1:n representing the pivots 
       (i.e., your solution should (approximately) satisfy A[p, :] = L*U) 
"""
function lu_partial_pivoting(A)
    # A=LU, where L and U are the results of recording GE matrix operations
    # pivoting: at step k, eliminate with the row that has the largest absolute value entry in the kth column
    
    A_copy = copy(A) # don't modify the original matrix
    n = size(A, 1) # number of rows

    # preallocate variables
    L = zeros(eltype(A_copy), n, n) # will store lower triangular matrix
    U = zeros(eltype(A), n, n) # will store upper triangular matrix
    p = collect(1:n) # will store vector of pivots

    for k in 1:n # each instance of p represents the row order
        pivot_row = k - 1 + findmax(abs.(A[k:n, k]))[2]  # select entries in column k, from submatrix of rows k to n (previous rows already eliminated)
        #pivot_row += (k-1) # adjust submatrix indexing

        # swap rows in A, and record swaps in pivot vector
        A[[k, pivot_row], :] = A[[pivot_row, k], :]
        p[[k, pivot_row]] = p[[pivot_row, k]]

        # check for singularity
        if abs(A[k,k]) < eps(eltype(A))  
            error("Matrix is singular to working precision at pivot $k")
        end

        # swap L entries for previous columns (columns from 1 to k-1)
        if k > 1 # no previous columns if k=1, so nothing to swap
            L[[k, pivot_row], 1:k-1] = L[[pivot_row, k], 1:k-1] # swap row k with pivot row to bring largest entry to diagonal
        end

        # elimination to zero out entries below pivot
        for i in k+1:n
            L[i,k] = A[i,k] / A[k,k]
            A[i,k:n] -= L[i,k] * A[k,k:n]
        end

        
    end

    # after all eliminations
    U = triu(A)
    for i in 1:n
        L[i,i] = 1.0
    end

    return L, U, p
end
"""
Solve A*x = b for x given the LU decomposition (approximately) satisfying A[p, :] = L*U.

Inputs:
    L: lower triangular matrix
    U: upper triangular matrix
    p: vector of permuted indices representing the pivots

Output:
    x: solution to A*x = b (recall that A[p, :] = L*U)
"""
function lu_solve(L, U, p, b)

    n = size(L, 1) # number of rows and columns
    b_perm = b[p]  # reorder b according to the pivot vector (p maps from original to pivoted rows)

    # we have L and U, where LU=A
    # solve Lz=b using forward substitution
    z = forward_substitution(L, b_perm)

    # solve Ux=z using backward substitution
    x = backward_substitution(U, z)

    return x
end
