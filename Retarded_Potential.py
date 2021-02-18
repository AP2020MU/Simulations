# 遅延ポテンシャルのシミュレート
# 参考資料(FDTD法)
# https://qiita.com/sandshiP/items/2b8b10265d0c11597081
# https://github.com/sandship/practice_fdtd/tree/master/imp
# http://www.emlab.cei.uec.ac.jp/denjikai/fself.dtd.pdf
# https://qiita.com/atily17/items/aeff9e1c4609e33f2d72
# https://qiita.com/tommyecguitar/items/010b1ef05c530b1284cb


from matplotlib import cm
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import product

c = 2.99792458e8 # 光速[m/s]
eps0 = 8.8541878128e-12 # 真空の誘電率[F/m]
mu0 = 1.25663706212e-06 # 真空の透磁率[N/A^2]



dx = 0.01
dy = 0.01
dz = 0.01
dt = 0.99/(c * np.sqrt((1.0/dx ** 2 + 1.0/dy ** 2 + 1.0/dz ** 2))) # 時間差分間隔[s] Courantの安定条件を満たす
print("dt=",dt)

ctx = (c*dt/dx)**2 #constant
cty = (c*dt/dy)**2 #constant
ctz = (c*dt/dz)**2 #constant
cte = (c*dt)**2 / eps0

nx = 100
ny = 100
nz = 100
nt = 200


x = np.array([i*dx for i in range(nx)])
y = np.array([i*dy for i in range(ny)])
X, Y = np.meshgrid(x,y)

# ポテンシャル
u = np.zeros((nt,nx,ny,nz))
# 電荷密度
rho = np.zeros((nt,nx,ny,nz))
Q = 1 # 電荷[C]
# 点電荷を扱うと必然的に無限大の発散が出てきてしまうためナイーブな実装では壊れる
# そこでFDTD法のように電荷の存在位置を半格子分だけずらして離散化することを考える
# この考え方によれば点電荷のδ関数的な電荷密度は，広がりを持つものとして定式化されるため
# 点電荷近傍では真のポテンシャル/電場を再現しない可能性があることに留意

#rho[:,nx//2,ny//2,nz//2] = Q/(dx*dy*dz) # 電荷密度[C/m^3]

"""
# 静止した点電荷
for i,j,k in product([0,1],repeat=3):
    rho[:,nx//2+i,ny//2+j,nz//2+k] = Q/(dx*dy*dz)/8
"""
"""
# 正負の電荷が実空間で振動
omega = 10**10 # 角振動数[rad/s]
for n in range(nt):
    for i,j,k in product([0,1],repeat=3):
        rho[n,nx//2+i+int(30*np.sin(omega*n*dt)),ny//2+j,nz//2+k] = Q/(dx*dy*dz)/8
        rho[n,nx//2+i-int(30*np.sin(omega*n*dt)),ny//2+j,nz//2+k] = Q/(dx*dy*dz)/8
"""
"""
# 固定された2電荷の電荷量が振動
omega = 10**10 # 角振動数[rad/s]
for n in range(nt):
    for i,j,k in product([0,1],repeat=3):
        rho[n,nx//2+i+nx//5,ny//2+j,nz//2+k] = Q*np.sin(omega*n*dt)/(dx*dy*dz)/8
        rho[n,nx//2+i-nx//5,ny//2+j,nz//2+k] = -Q*np.sin(omega*n*dt)/(dx*dy*dz)/8
"""

# 電荷量が振動する静止した点電荷
omega = 10**10 # 角振動数[rad/s]
for n in range(nt):
    for i,j,k in product([0,1],repeat=3):
        rho[n,nx//2+i,ny//2+j,nz//2+k] = Q*np.sin(omega*n*dt)/(dx*dy*dz)/8


#rho[:,nx//2,ny//2,nz//2] = 5.0e-6*Q/eps0

# ポテンシャルの初期状態
#u_0 = np.exp(-((X-2)**2)*10)*np.exp(-((Y-2)**2)*10)*2
u_0 = np.zeros((nx,ny,nz))
u_1 = np.zeros((nx,ny,nz))
#np.sin(X)*np.sin(Y)
#np.exp(-(X**2)*10)*np.exp(-(Y**2)*10)
#np.exp(-((X-2)**2)*10)*np.exp(-((Y-2)**2)*10)

u[0] = u_0
u[1] = u[0] + dt * u_1


for t in tqdm(range(1,nt-1)):
    # ポテンシャルの更新
    u[t+1] = 2*(1-ctx-cty-ctz)*u[t] - u[t-1] \
            + ctx*(np.roll(u[t],shift=1,axis=0) + np.roll(u[t],shift=-1,axis=0)) \
            + cty*(np.roll(u[t],shift=1,axis=1) + np.roll(u[t],shift=-1,axis=1)) \
            + ctz*(np.roll(u[t],shift=1,axis=2) + np.roll(u[t],shift=-1,axis=2)) \
            + cte*rho[t]
    
    # Neumann条件
    u[t+1,0,:,:] = u[t+1,1,:,:]
    u[t+1,:,0,:] = u[t+1,:,1,:]
    u[t+1,:,:,0] = u[t+1,:,:,1]
    u[t+1,nx-1,:,:] = u[t+1,nx-2,:,:]
    u[t+1,:,ny-1,:] = u[t+1,:,ny-2,:]
    u[t+1,:,:,nz-1] = u[t+1,:,:,nz-2]
    """
    u[t,0,0] = (u[t,1,0]+u[t,0,1])/2
    u[t,nx-1,0] = (u[t,nx-2,0]+u[t,nx-1,1])/2
    u[t,0,ny-1] = (u[t,1,ny-1]+u[t,0,ny-2])/2
    u[t,nx-1,ny-1] = (u[t,nx-2,ny-1]+u[t,nx-1,ny-2])/2
    """

# 解析解におけるポテンシャル
real_phi = np.zeros((nx,ny,nz))
dr = np.sqrt(dx**2+dy**2+dz**2)
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            real_phi[i,j,k] = Q/(4*np.pi*eps0*dr*np.sqrt((i-0.5-nx//2)**2 + (j-0.5-ny//2)**2 + (k-0.5-nz//2)**2))
            """
            if (i,j,k) == (nx//2,ny//2,nz//2):
                real_phi[i,j,k] = Q/(4*np.pi*eps0*dx*dy*dz)
                real_phi[i,j,k] = u[10,nx//2,ny//2,nz//2]
            else:
                real_phi[i,j,k] = Q/(4*np.pi*eps0*dr*np.sqrt((i-nx//2)**2 + (j-ny//2)**2 + (k-nz//2)**2))

            """
# 電場
Ex = np.zeros((nt,nx,ny,nz))
Ey = np.zeros((nt,nx,ny,nz))
Ez = np.zeros((nt,nx,ny,nz))
# 解析解における電場
real_Ex = np.zeros((nx,ny,nz))
for t in tqdm(range(1,nt-1)):
    Ex[t] = (u[t]-np.roll(u[t],shift=1,axis=0))/dx
    Ey[t] = (u[t]-np.roll(u[t],shift=1,axis=1))/dy
    Ez[t] = (u[t]-np.roll(u[t],shift=1,axis=2))/dz

real_Ex = (real_phi-np.roll(real_phi,shift=1,axis=0))/dx


# 描画
fig = plt.figure()
fig.set_dpi(100)
ax = fig.gca(projection='3d')
lim_val = max(abs(np.max(u)),abs(np.min(u)))
lim_val = max(abs(np.max(Ex)),abs(np.min(Ex)))
absE = np.sqrt(Ex**2+Ey**2+Ez**2)
def animate(i):
    ax.clear()
    ax.set_title(f"{i}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    #ax.plot_surface(X, Y, u[i], rstride=1, cstride=1, cmap=plt.cm.coolwarm,vmax=1,vmin=-1)
    #ax.plot_wireframe(X, Y, u[i,:,:,nz//2], rstride=1, cstride=1,color='blue', linewidth=0.2)
    #ax.plot_wireframe(X, Y, real_phi[:,:,nz//2], rstride=1, cstride=1,color='red', linewidth=0.2)
    #ax.plot_wireframe(X, Y, u[i,:,:,nz//2]-real_phi[:,:,nz//2], rstride=1, cstride=1,color='green', linewidth=0.2)
    #ax.plot_wireframe(X, Y, Ex[i,:,:,nz//2]-real_Ex[:,:,nz//2], rstride=1, cstride=1,color='green', linewidth=0.2)
    #ax.plot_wireframe(X, Y, Ex[i,:,:,nz//2], rstride=1, cstride=1,color='red', linewidth=0.2)
    #ax.plot_wireframe(X, Y, real_Ex[:,:,nz//2], rstride=1, cstride=1,color='blue', linewidth=0.2)
    ax.plot_wireframe(X, Y, absE[i,:,:,nz//2], rstride=1, cstride=1,color='red', linewidth=0.2)
    ax.set_zlim(-lim_val, lim_val)
    
anim = animation.FuncAnimation(fig,animate,frames=nt-1,interval=10)
#anim.save("wave2D-N2.gif", writer="imagemagick")
plt.show()