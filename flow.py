import dolfin as df

Ly = 4.0  # Domain size y
Lx = 0.5  # Domain size x
Lm = 2.0  # Membrane size
Ny = 64   # resolution y
Nx = 16   # resolution x

dt = 0.1
nu = 0.1
t_tot = 10.
K_T = 0.01

inner_iterations = 2

p_left_top = -5.0
p_left_btm = 5.0
p_right_top = 5.0
p_right_btm = -5.0

T_btm = 1.0
T_top = 0.0

kappa_u = 0.005
kappa_T = 0.05

df.parameters['allow_extrapolation'] = True

class GenericBC(df.SubDomain):
    def __init__(self, Lx, Ly, Lm):
        self.Lx = Lx
        self.Ly = Ly
        self.Lm = Lm
        df.SubDomain.__init__(self)

class Left(GenericBC):
    def inside(self, x, on_bnd):
        return on_bnd and df.near(x[0], 0)

class Bottom(GenericBC):
    def inside(self, x, on_bnd):
        return on_bnd and df.near(x[1], 0)

class Top(GenericBC):
    def inside(self, x, on_bnd):
        return on_bnd and df.near(x[1], self.Ly)

class Middle(GenericBC):
    def inside(self, x, on_bnd):
        return (on_bnd and
                x[0] > self.Lx - 2*df.DOLFIN_EPS and
                x[0] < self.Lx + 2*df.DOLFIN_EPS and
                (x[1] < (self.Ly-self.Lm)/2+df.DOLFIN_EPS_LARGE or
                 x[1] > (self.Ly+self.Lm)/2-df.DOLFIN_EPS_LARGE))

class Membrane(GenericBC):
    def inside(self, x, on_bnd):
        return (on_bnd and
                x[0] > self.Lx - 2*df.DOLFIN_EPS and
                x[0] < self.Lx + 2*df.DOLFIN_EPS and
                (x[1] >= (self.Ly-self.Lm)/2 and
                 x[1] <= (self.Ly+self.Lm)/2))

class Right(GenericBC):
    def inside(self, x, on_bnd):
        return on_bnd and df.near(x[0], 2*self.Lx)


class SubProblem:
    def __init__(self, side, Lx, Ly, Lm):
        u_el = df.VectorElement("CG", "triangle", 2)
        p_el = df.FiniteElement("CG", "triangle", 1)
        w_el = df.MixedElement([u_el, p_el])
        T_el = df.FiniteElement("CG", "triangle", 1)

        top = Top(Lx, Ly, Lm)
        bottom = Bottom(Lx, Ly, Lm)
        left = Left(Lx, Ly, Lm)
        right = Right(Lx, Ly, Lm)
        middle = Middle(Lx, Ly, Lm)
        membrane = Membrane(Lx, Ly, Lm)

        if side == "left":
            self.mesh = df.RectangleMesh(df.Point(0., 0.),
                                         df.Point(Lx+df.DOLFIN_EPS, Ly), Nx, Ny)
        elif side == "right":
            self.mesh = df.RectangleMesh(df.Point(Lx-df.DOLFIN_EPS, 0.),
                                         df.Point(2*Lx, Ly), Nx, Ny)
        else:
            exit("Unknown side")

        self.subd = df.MeshFunction("size_t", self.mesh,
                                    self.mesh.topology().dim()-1)
        self.subd.rename("subd", "subd")
        self.subd.set_all(0)

        top.mark(self.subd, 1)
        bottom.mark(self.subd, 2)
        if side == "left":
            left.mark(self.subd, 3)
        else:
            right.mark(self.subd, 3)
        middle.mark(self.subd, 4)
        membrane.mark(self.subd, 5)

        df.XDMFFile(self.mesh.mpi_comm(),
                    "subd_{}.xdmf".format(side)).write(self.subd)

        self.W = df.FunctionSpace(self.mesh, w_el)

        self.w_ = df.Function(self.W)
        self.w_1 = df.Function(self.W)

        u, p = df.TrialFunctions(self.W)
        v, q = df.TestFunctions(self.W)
        self.u_, self.p_ = df.split(self.w_)
        u_1, p_1 = df.split(self.w_1)
        self.ux_mem = df.Function(self.W.sub(0).sub(0).collapse())

        self.S = df.FunctionSpace(self.mesh, T_el)
        self.T_ = df.Function(self.S, name="T")
        self.T_1 = df.Function(self.S)
        T = df.TrialFunction(self.S)
        s = df.TestFunction(self.S)
        self.JTx_mem = df.Function(self.S)

        n = df.FacetNormal(self.mesh)
        ds = df.Measure("ds", domain=self.mesh,
                        subdomain_data=self.subd)

        self.p_top = df.Constant(0.)
        self.p_btm = df.Constant(0.)

        F_up = (df.dot(u - u_1, v) / dt * df.dx
                + df.inner(df.grad(u) * u_1, v) * df.dx
                + nu * df.inner(df.grad(u), df.grad(v)) * df.dx
                - df.div(v) * p * df.dx
                - df.div(u) * q * df.dx)
        F_up += self.p_top * df.dot(n, v) * ds(1) \
            + self.p_btm * df.dot(n, v) * ds(2)

        a_up, L_up = df.lhs(F_up), df.rhs(F_up)

        self.T_top = df.Constant(0.)
        self.T_btm = df.Constant(0.)

        F_T = ((T - self.T_1) * s * df.dx
               - df.dot(u_1 * T - K_T * df.grad(T), df.grad(s)) * df.dx)
        if side == "left":
            F_T += self.T_btm * df.dot(n, u_1) * s * ds(2) \
                + T * df.dot(n, u_1) * s * ds(1)
            F_T += self.JTx_mem * s * ds(5)
        else:
            F_T += self.T_top * df.dot(n, u_1) * s * ds(1) \
                + T * df.dot(n, u_1) * s * ds(2)
            F_T += -self.JTx_mem * s * ds(5)
        a_T, L_T = df.lhs(F_T), df.rhs(F_T)

        noslip = df.Constant((0., 0.))
        bcp_top = df.DirichletBC(self.W.sub(1), self.p_top,
                                 self.subd, 1)
        bcp_btm = df.DirichletBC(self.W.sub(1), self.p_btm,
                                 self.subd, 2)
        if side == "left":
            bcu_side = df.DirichletBC(self.W.sub(0), noslip,
                                      self.subd, 3)
        else:
            bcu_side = df.DirichletBC(self.W.sub(0), noslip,
                                      self.subd, 3)
        bcu_middle = df.DirichletBC(self.W.sub(0), noslip,
                                    self.subd, 4)
        bcux_membrane = df.DirichletBC(self.W.sub(0).sub(0),
                                       self.ux_mem,
                                       self.subd, 5)
        bcuy_membrane = df.DirichletBC(self.W.sub(0).sub(1), 0.,
                                       self.subd, 5)

        bcs_up = [bcp_top, bcp_btm,
                  bcu_side, bcu_middle,
                  bcux_membrane, bcuy_membrane]

        problem_up = df.LinearVariationalProblem(a_up, L_up,
                                                 self.w_, bcs=bcs_up)
        self.solver_up = df.LinearVariationalSolver(problem_up)

        if side == "left":
            bcs_T = df.DirichletBC(self.S, self.T_btm, self.subd, 2)
        else:
            bcs_T = df.DirichletBC(self.S, self.T_top, self.subd, 1)

        problem_T = df.LinearVariationalProblem(a_T, L_T,
                                                self.T_, bcs=bcs_T)
        self.solver_T = df.LinearVariationalSolver(problem_T)

        self.xdmf_u = df.XDMFFile(self.mesh.mpi_comm(),
                                  "u_{}.xdmf".format(side))
        self.xdmf_p = df.XDMFFile(self.mesh.mpi_comm(),
                                  "p_{}.xdmf".format(side))
        self.xdmf_T = df.XDMFFile(self.mesh.mpi_comm(),
                                  "T_{}.xdmf".format(side))
        self.xdmf_u.parameters["rewrite_function_mesh"] = False
        self.xdmf_u.parameters["flush_output"] = True
        self.xdmf_p.parameters["rewrite_function_mesh"] = False
        self.xdmf_p.parameters["flush_output"] = True
        self.xdmf_T.parameters["rewrite_function_mesh"] = False
        self.xdmf_T.parameters["flush_output"] = True

    def solve(self):
        self.solver_up.solve()
        self.U_, self.P_ = self.w_.split(deepcopy=True)
        self.U_.rename("u", "u")
        self.P_.rename("p", "p")
        self.solver_T.solve()

    def dump(self):
        self.xdmf_u.write(self.U_, t)
        self.xdmf_p.write(self.P_, t)
        self.xdmf_T.write(self.T_, t)

    def update(self):
        self.w_1.assign(self.w_)
        self.T_1.assign(self.T_)


sp_left = SubProblem("left", Lx, Ly, Lm)
sp_left.p_top.assign(p_left_top)
sp_left.p_btm.assign(p_left_btm)
sp_left.T_btm.assign(T_btm)

sp_right = SubProblem("right", Lx, Ly, Lm)
sp_right.p_top.assign(p_right_top)
sp_right.p_btm.assign(p_right_btm)
sp_right.T_top.assign(T_top)

mesh_membrane = df.RectangleMesh(df.Point(Lx-df.DOLFIN_EPS, 0),
                                 df.Point(Lx+df.DOLFIN_EPS, Ly),
                                 1, Ny)
S_mem = df.FunctionSpace(mesh_membrane, "Lagrange", 1)
p_mem_left = df.Function(S_mem, name="p_left")
p_mem_right = df.Function(S_mem, name="p_right")
T_mem_left = df.Function(S_mem, name="T_left")
T_mem_right = df.Function(S_mem, name="T_right")

xdmf_mem = df.XDMFFile(mesh_membrane.mpi_comm(), "p_mem.xdmf")

t = 0.0
while t <= t_tot:
    t += dt
    print(t)
    for i in range(inner_iterations):
        sp_left.solve()
        sp_right.solve()

        p_mem_left.interpolate(sp_left.P_)
        p_mem_right.interpolate(sp_right.P_)

        ux_mem = df.project(kappa_u*(p_mem_left-p_mem_right),
                            p_mem_left.function_space())

        sp_left.ux_mem.interpolate(ux_mem)
        sp_right.ux_mem.interpolate(ux_mem)

        T_mem_left.interpolate(sp_left.T_)
        T_mem_right.interpolate(sp_right.T_)
        JTx_mem = df.project(kappa_T*(T_mem_left-T_mem_right),
                             T_mem_left.function_space())

        sp_left.JTx_mem.interpolate(JTx_mem)
        sp_right.JTx_mem.interpolate(JTx_mem)

    sp_left.dump()
    sp_right.dump()

    sp_left.update()
    sp_right.update()
