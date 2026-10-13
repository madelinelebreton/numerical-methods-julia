# Floating Point Computation & Direct Linear System Solvers

Analysis of floating point error and a direct linear solver in Julia from scratch.

**Implemented**
- Backward substitution for upper triangular systems
- Gaussian elimination, with and without partial pivoting
- Benchmarking of solution error against matrix condition number across hundreds of randomly generated linear systems

**Files**
- `madeline_lebreton_a1.jl`: implementation
- `madeline_lebreton_a1.pdf`: write up with derivations, error analysis, and plots
- `figures/`: supporting plots
  
  
  
<p align="center">
  <br/>
  Error vs. condition number comparison for Gaussian Elimination and Partial Pivoting
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/d441aaa7-98e9-4132-b255-0d095cd93c2d" />
</p>

