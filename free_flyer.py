import jax.numpy as np
import jax

def free_flyer(state,control,dt):
	m = 7.2
	J1 = 0.07
	J2 = 0.07
	J3 = 0.17

# control variables
	u1=control[0]
	u2=control[1]
	u3=control[2]
	u4=control[3]
	u5=control[4]
	u6=control[5]

# state variables
	x00=state[0]
	x10=state[1]
	x20=state[2]
	x30=state[3]
	x40=state[4]
	x50=state[5]
	x60=state[6]
	x70=state[7]
	x80=state[8]
	x90=state[9]
	x100=state[10]
	x110=state[11]
	x120=state[12]
# writing dynamics
	dxdt = np.zeros(13)
	dxdt = np.array(dxdt,dtype = np.float32)
	xx = dt*x30
	dxdt = jax.ops.index_update(dxdt,0,xx)
	xx = dt*x40
	dxdt = jax.ops.index_update(dxdt,1,xx)
	xx = dt*x50
	dxdt = jax.ops.index_update(dxdt,2,xx)
	xx = 1.0*dt*u1/m
	dxdt = jax.ops.index_update(dxdt,3,xx)
	xx = 1.0*dt*u2/m
	dxdt = jax.ops.index_update(dxdt,4,xx)
	xx = 1.0*dt*u3/m
	dxdt = jax.ops.index_update(dxdt,5,xx)
	xx = dt*(-0.5*x100*x70 - 0.5*x110*x80 - 0.5*x120*x90)
	dxdt = jax.ops.index_update(dxdt,6,xx)
	xx = dt*(0.5*x100*x60 - 0.5*x110*x90 + 0.5*x120*x70)
	dxdt = jax.ops.index_update(dxdt,7,xx)
	xx = dt*(0.5*x100*x90 + 0.5*x110*x60 - 0.5*x120*x70)
	dxdt = jax.ops.index_update(dxdt,8,xx)
	xx = dt*(-0.5*x100*x80 + 0.5*x110*x70 + 0.5*x120*x60)
	dxdt = jax.ops.index_update(dxdt,9,xx)
	xx = -J2*dt*x110*x120/J1 + J3*dt*x110*x120/J1 + dt*u4/J1
	dxdt = jax.ops.index_update(dxdt,10,xx)
	xx = -J1*dt*x100*x120/J2 + J3*dt*x100*x120/J2 + dt*u5/J2
	dxdt = jax.ops.index_update(dxdt,11,xx)
	xx = J1*dt*x100*x110/J3 - J2*dt*x100*x110/J3 + dt*u6/J3
	dxdt = jax.ops.index_update(dxdt,12,xx)
	return dxdt