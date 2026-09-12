# 3X03 assignment 3: Integration and Numerical Integration
# author: madeline lebreton
# date: 25/11/2025

""" 
helper function for recursive divided differences calls, using indexing
"""

function divided_diffs(i, j, x, y)

    # base case c1 = y1 
    if i == j
        c_new = y[i]          
        return c_new
    end

    # base case first divided difference
    if (i + 1) == j
        c_new = (y[j] - y[i]) / (x[j] - x[i])
        return c_new
    end

    # recursive calls
    right = divided_diffs(i + 1, j, x, y)
    left  = divided_diffs(i, j - 1, x, y)

    c_new = (right - left) / (x[j] - x[i])
    #push!(c, c_new)

    return c_new  # return new coefficient
end


""" 
Computes the coefficients of Newton's interpolating polynomial. 
    Inputs 
        x: vector with distinct elements x[i] 
        y: vector of the same size as x 
    Output 
        c: vector with the coefficients of the polynomial
"""
function newton_int(x, y)
    n = length(x) # need to compute this many coefficients

    # initialize 2D table: index [i,j] holds f[x_i,...,x_j]
    table = zeros(n, n)

    # fill first column with y-values
    table[:, 1] = y

    # fill upper triangle with divided differences
    for j in 2:n # column
        for i in (j:n) # row
            table[i, j] = (table[i, j-1] - table[i-1, j-1]) / (x[i] - x[i-j+1])
        end
    end

    # coefficients are the diagonal entries
    c = [table[j, j] for j in 1:n]

    return c
end



"""
Evaluates a polynomial with Newton coefficients c 
defined over nodes x using Horner's rule on the points in X.
Inputs 
    c: vector with n coefficients 
    x: vector of n distinct points used to compute c in newton_int 
    X: vector of m points 
Output 
    p: vector of m points
"""
function horner(c, x, X)
    n = length(x)
    p = zeros(length(X))

    for k in 1:(length(X))
        t = X[k]
        b = c[n] # b_0 = c[n]
        for i in (n-1):-1:1
            b = c[i] + (t - x[i]) * b # horners method b_1 = c[n-1] + (t - x[n-1])*b, b_2 = c[n-2] + (t - x[n-2])*b , ...
        end
        p[k] = b
    end

    return p
end

"""
Compute the integral ∫f(x)dx over [a, b] with the composite trapezoidal 
rule using r subintervals.

Inputs:
    
    f: function to integrate
    a: lower bound of the definite integral
    b: upper bound of the definite integral
    r: number of subintervals
"""
function composite_trapezoidal_rule(f, a, b, r)
    dx = (b-a)/r # size of subinterval
    approximate_integral = 0 # this will hold the sum of all trapezoids

    for i in 1:(r)
        l = a + (i-1)*dx # left side of trapezoid
        r = l + dx 
        approximate_integral += (r - l)/2 * (f(l)+f(r)) # add the contribution to the trapezoid
        #println("approximate integral trapezoid: $approximate_integral")

    end

    return approximate_integral
    
end

"""
Compute the integral ∫f(x)dx over [a, b] with the composite midpoint 
rule using r subintervals.

Inputs:
    
    f: function to integrate
    a: lower bound of the definite integral
    b: upper bound of the definite integral
    r: number of subintervals
"""
function composite_midpoint_rule(f, a, b, r)

    dx = (b-a)/r # size of subinterval
    approximate_integral = 0 # this will hold the sum of all rectangles

    for i in 1:(r)
        l = a + (i-1)*dx # left side of trapezoid
        m = l + dx/2
        approximate_integral += (dx)*f(m) # add the contribution to the integral (base x height)
        #println("approximate integral midpoint: $approximate_integral")
    end

    return approximate_integral
end

"""
Compute the integral ∫f(x)dx over [a, b] with the composite Simpson's 
rule using r subintervals. Note that r must be even because each 
application of Simpson's rule uses a subinterval of length 2*(b-a)/r.
In other words, the midpoints used by the basic Simpson's rule are 
included in the r+1 points on which we evaluate f(x).

Inputs:
    
    f: function to integrate
    a: lower bound of the definite integral
    b: upper bound of the definite integral
    r: even number of subintervals
"""
function composite_simpsons_rule(f, a, b, r)

    dx = (b - a) / r  # width of subinterval
    approximate_integral = 0

    for i in 0:(r/2 - 1)
        l = a + 2*i*dx
        m = l + dx
        r = l + 2*dx
        approximate_integral += (dx/3) * (f(l) + 4*f(m) + f(r))
    end

    return approximate_integral
end
