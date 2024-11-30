import cupy as cp

def modify_init(u2d, v2d, k2d, om2d, vis2d):
    # set inlet field in entire domain
    k2d = cp.ones((ni, nj))
    om2d = cp.ones((ni, nj))

    vis2d = k2d / om2d + viscos

    return u2d, v2d, k2d, om2d, vis2d

def modify_inlet():
    global y_rans, y_rans, u_rans, v_rans, k_rans, om_rans, uv_rans, k_bc_west, eps_bc_west, om_bc_west

    return u_bc_west, v_bc_west, k_bc_west, om_bc_west, u2d_face_w, convw

def modify_conv(convw, convs):
    # since we are solving for fully-developed channel flow, we know that the convection terms are zero
    convs = cp.zeros((ni, nj))
    convw = cp.zeros((ni, nj))

    return convw, convs

def modify_u(su2d, sp2d):
    global file1

    # add a driving pressure gradient term
    su2d = su2d + vol

    # we know that the convection and diffusion term in the x direction are zero
    aw2d = cp.zeros((ni, nj))
    ae2d = cp.zeros((ni, nj))

    # wall shear stress calculations
    tauw_south = viscos * cp.sum(as_bound * u2d[:, 0]) / x2d[-1, 0]
    tauw_north = viscos * cp.sum(an_bound * u2d[:, -1]) / x2d[-1, 0]

    print(f"{'tau wall, south: '} {tauw_south:.3f},{'  tau wall, north: '} {tauw_north:.3f}")

    return su2d, sp2d

def modify_v(su2d, sp2d):
    return su2d, sp2d

def modify_p(su2d, sp2d):
    return su2d, sp2d

def modify_k(su2d, sp2d):
    # we know that the convection and diffusion term in the x direction are zero
    aw2d = cp.zeros((ni, nj))
    ae2d = cp.zeros((ni, nj))

    return su2d, sp2d

def modify_om(su2d, sp2d):
    # we know that the convection and diffusion term in the x direction are zero
    aw2d = cp.zeros((ni, nj))
    ae2d = cp.zeros((ni, nj))

    return su2d, sp2d

def modify_outlet(convw):
    # since we are solving for fully-developed channel flow, we know that the convection terms are zero
    convw = cp.zeros((ni + 1, nj))

    return convw

def fix_omega():
    return aw2d, ae2d, as2d, an2d, ap2d, su2d, sp2d

def modify_vis(vis2d):
    return vis2d

def fix_k():
    return aw2d, ae2d, as2d, an2d, ap2d, su2d, sp2d
