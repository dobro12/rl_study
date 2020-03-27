#cg 는 wiki랑 일치
p = b.copy()
r = b.copy()
x = np.zeros_like(b)
rdotr = r.dot(r)

for i in range(cg_iters):
    z = f_Ax(p)
    v = rdotr / p.dot(z)
    x += v*p
    r -= v*z
    newrdotr = r.dot(r)
    mu = newrdotr/rdotr
    p = r + mu*p

    rdotr = newrdotr
    if rdotr < residual_tol:
        break

#f_Ax(p) : #fisher_vector_product
def fisher_vector_product(p):
    return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p
#cg_damping              conjugate gradient damping -> A에 미세한 항등행렬을 더해주는 효과.!!! -> 1e-2 줌