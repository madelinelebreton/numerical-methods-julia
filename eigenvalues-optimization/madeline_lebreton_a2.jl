# Assignment 2: Eigenvalues and optimization
# Author: Madeline LeBreton with materials from MECHTRON 3X03
# Date: 18/11/2025

using LinearAlgebra

"""
Computes the maximum magnitude eigenpair for the real symmetric matrix A 
with the power method. Ensures that the error tolerance is satisfied by 
using the Bauer-Fike theorem.

Inputs:
    A: real symmetric matrix
    tol: error tolerance; i.e., the maximum eigenvalue estimate 
         must be within tol of the true maximum eigenvalue

Outputs:
    λ: the estimate of the maximum magnitude eigenvalue
    v: the estimate of the eigenvector corresponding to λ
"""

function power_method_symmetric(A, tol)
    v0 = zeros(size(A, 1)) # normalized initial guess eigenvector of 1s
    v0[1] = 1
    v = v0
    bound = tol + 1 # ensure at least one loop
    λ = 0.0 # initialize max eigenvalue

    while(bound >= tol) # termination criterion: bauer-fike theorem with input tol
        v = A * v # update vector of next iteration
        v = v / norm(v) # normalize with max value
        λ = v'*A*v

        # calculate bauer-fike bound 
        r = A*v - λ*v
        bound = norm(r) / norm(v)
    end
    return λ, v
end



"""
Compute the eigenpairs of the k extremal eigenvalues (i.e., the k eigenvalues of real 
symmetric matrix A with the greatest absolute value).

Inputs:
    A: nxn real symmetric matrix
    k: number of extremal (i.e., maximum magnitude) eigenvalues to compute
    tol: error tolerance; i.e., each eigenvalue estimate 
         must be within tol of a true eigenvalue

Outputs:
    λ: vector of k real elements containing the estimates of the extremal eigenvalues;
       λ[i] contains the ith largest eigenvalue by absolute value
    V: nxk matrix where V[:, i] is the eigenvector for the ith largest eigenvalue 
       by absolute value

"""
function extremal_eigenpairs(A, k, tol)

    A = copy(A)
    k = Int(k)
    v = zeros(size(A, 1))
    λ = zeros(k)
    V = zeros(size(A, 1), k)

    for i in 1:(k)
        lambda_i, v_i = power_method_symmetric(A, tol) # gets largest eigenvalue and eigenvector

        λ[i] = lambda_i # sets λ[i] as the next largest abs eigenvalue

        V[:, i] = v_i # sets V[:, i] as the corresponding eigenvector

        A -= lambda_i * (v_i * v_i') # remove the found eigenvalue and decrease rank of A by 1, moving closer to rank Q2 rank 1 case

        if lambda_i == 0 # no remaining nonzero eigenvalues 
            break
        end

    end
    return λ, V
end


"""
Helper Jacobian function for Newton's Method. 

Inputs: 
    x: specific position of the receiver in R^n
    P: nxn matrix with rows p_i
"""

function jacobian(x, P)
    n = size(P, 1)
    m = size(P, 2)
    J = zeros(m, n) # this will be the Jacobian matrix to return

    for i in 1:m # each iteration corresponds to the gradient of one f_i
        r = x - P[:, i] # vector from transmitter to receiver
        J[i, :] = (r / norm(r))'
    end

    return J # return the Jacobian matrix
end

"""
Use Newton's method to solve the nonlinear system of equations described in Problems 4-5.
This should work for Euclidean distance measurements in any dimension n.

Inputs:
    x0: initial guess for the position of the receiver in R^n
    P: nxn matrix with known locations of transmitting beacons as columns
    d: vector in R^n where d[i] contains the distance from beacon P[:, i] to x
    tol: Euclidean error tolerance (stop when norm(F(x)) <= tol)
    max_iters: maximum iterations of Newton's method to try

Returns:
    x_trace: Vector{Vector{Float64}} containing each Newton iterate x_k in R^n. 

"""
function newton(x0, P, d, tol, max_iters)
    x = copy(x0)
    x_trace = [x]
    error = Inf
    iters = 0
    n = size(P, 1)
    J = zeros(n, n)

    while ((error > tol) && (iters < max_iters))
        F_i = [norm(x_trace[end] - P[:, i]) - d[i] for i in 1:3] # evaluate F

        J = jacobian(x, P) # evaluate current jacobian using helper function

        # solve linear system F = J*D 
        D = J\F_i

        # iteratively move step size D closer to solution
        x -= D 

        push!(x_trace, copy(x)) # save iterate

        error = norm(F_i) # check stopping criterion
        iters += 1 # make sure max number of iterations isn't exceeded

    end

    return x_trace
end



"""
Use Newton's method to solve the nonlinear optimization problem described in Problems 6-7.
This should work for Euclidean distance measurements in any dimension n, and any number 
of noisy measurements m.

Inputs:
    x0: initial guess for the position of the receiver in R^n
    P: nxm matrix with known locations of transmitting beacons as columns
    d: vector in R^m where d[i] contains the noisy distance from beacon P[:, i] to x
    tol: gradient Euclidean error tolerance (stop when norm(∇f(x)) <= tol)
    max_iters: maximum iterations of Newton's method to try

Returns:
    x_trace: Vector{Vector{Float64}} containing each Newton iterate x_k in R^n. 

"""
function newton_optimizer(x0, P, d, tol, max_iters)
    x = copy(x0)
    x_trace = [x] # stores each iteration
    error = Inf
    iters = 0

    m = size(P, 2)

    while ((error > tol) && (iters < max_iters))
        J = jacobian(x, P) # using the jacobian helper function

        G = grad_f(x, P, d) # using the gradient helper function

        H = 2 * (J' * J) # using the gauss-newton hessian approximation H = J' J because my code had errors otherwise

        delta_x = H \ G # one step of newton's method: H * delta_x = G
        #println("\nIteration $iters \ngradient: $G \ndelta_x: $delta_x")

        x -= delta_x # update x vector 

        push!(x_trace, copy(x)) # store the iterate

        error = norm(G)
        iters += 1

        #println("\nH: $(H)")
        #println("eigenvalues of H: $(eigvals(H))")
    end

    return x_trace
end



# gradient helper function
function grad_f(x, P, d)
    n = size(P, 1) # p is nxm matrix of locations
    m = size(P, 2)
    G = zeros(n) # this matrix will store the gradient

    for i in 1:m
        r_vec = (x - P[:, i]) # vector that goes from beacon to x
        l = norm(r_vec) # residual length
        r_i = l - d[i] # residual
        u = r_vec / l # unit vector
        G .+= 2 * r_i * u # add gradient contribution
    end

    return G
end



"""
Use gradient descent as described in Problem 8 to solve the nonlinear optimization problem from Problem 6.
This should work for Euclidean distance measurements in any dimension n, and any number 
    of noisy measurements m.

Inputs:
    x0: initial guess for the position of the receiver in R^n
    P: nxm matrix with known locations of transmitting beacons as columns
    d: vector in R^m where d[i] contains the noisy distance from beacon P[:, i] to x
    tol: gradient Euclidean error tolerance (stop when norm(∇f(x)) <= tol)
    max_iters: maximum iterations of gradient descent to try
	gamma: step size constant

Returns:
    x_trace: Vector{Vector{Float64}} containing each gradient descent iterate x_k in R^n. 

"""

function gradient_descent(x0, P, d, tol, max_iters, gamma)
    x = copy(x0)
    x_trace = [x]
    iters = 0

    while iters < max_iters
        G = grad_f(x, P, d) # compute gradient using the helper function
        if norm(G) <= tol # if stopping criterion has been met
            break
        end

        x = x - gamma * G # update the gradient descent 
        push!(x_trace, copy(x)) # store the iterate
        iters += 1
    end
    
    return x_trace # return vector containing each iterate
end
