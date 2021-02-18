# 電磁場のFDTD法によるシミュレート
# 参考資料
# https://qiita.com/sandshiP/items/2b8b10265d0c11597081
# https://github.com/sandship/practice_fdtd/tree/master/imp
# http://www.emlab.cei.uec.ac.jp/denjikai/fdtd.pdf
# https://qiita.com/atily17/items/aeff9e1c4609e33f2d72
# https://qiita.com/tommyecguitar/items/010b1ef05c530b1284cb
# http://sudalab.is.s.u-tokyo.ac.jp/~matsumoto/SS2017/01_MM.pdf

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm


class FDTD3D:
    def __init__(self,nx=100,ny=100,nz=100,nt=100):
        c = 2.99792458e8 # 光速[m/s]
        eps0 = 8.8541878128e-12 # 真空の誘電率[F/m]
        mu0 = 1.25663706212e-06 # 真空の透磁率[N/A^2]

        self.nx = nx # x方向計算点数
        self.ny = ny # y方向計算点数
        self.nz = nz # z方向計算点数
        self.nt = nt # 計算ステップ数

        self.dx = 0.01 # x方向空間差分間隔[m]
        self.dy = 0.01 # y方向空間差分間隔[m]
        self.dz = 0.01 # z方向空間差分間隔[m]
        self.dt = 0.99/(c * np.sqrt((1.0/self.dx ** 2 + 1.0/self.dy ** 2 + 1.0/self.dz ** 2))) # 時間差分間隔[s] Courantの安定条件を満たす

        # 電気定数初期化と更新係数の計算
        self.eps = np.full((self.nx, self.ny, self.nz), eps0) # 誘電率
        self.mu = np.full((self.nx, self.ny, self.nz), mu0) # 透磁率
        self.sigma = np.full((self.nx, self.ny, self.nz), 0.0) # 導電率
        self.sigmam = np.full((self.nx, self.ny, self.nz), 0.0) # 導磁率：本来の差分式には現れないがPML境界条件の計算に使う

        
        # 境界条件としてBerengerのPML(Perfect Matched Layer)吸収境界条件を課す
        # PML用のパラメータ
        M = 2     # 吸収境界の導電率の上昇曲線の次数(2,3次が一般的)
        R = 1e-6  # 境界面において実現したい反射係数
        pmlN = 8  # PMLの層数，大きいほど計算コストが増えるが，反射率低減可

        # 厳密にはc*mu0ではなく波動アドミタンスY=np.sqrt(self.eps/self.mu)を用いる
        # また，誘電率・透磁率は各点の値を使うべきだが，境界は真空であると仮定して簡単化している
        # self.dx==self.dy==self.dzを仮定しているのも同様の理由
        sigma_max = (M+1) * (-np.log(R)) / (2*pmlN*self.dx*c*mu0)
        for ln in range(pmlN):
            sigma_value = ((pmlN - ln)/pmlN)**M * sigma_max # PML吸収境界のsigma

            self.sigma[ln, :, :] = sigma_value
            self.sigma[:, ln, :] = sigma_value
            self.sigma[:, :, ln] = sigma_value

            self.sigma[-(ln+1), :, :] = sigma_value
            self.sigma[:, -(ln+1), :] = sigma_value
            self.sigma[:, :, -(ln+1)] = sigma_value

            self.sigmam[ln, :, :] = mu0/eps0*sigma_value
            self.sigmam[:, ln, :] = mu0/eps0*sigma_value
            self.sigmam[:, :, ln] = mu0/eps0*sigma_value

            self.sigmam[-(ln+1), :, :] = mu0/eps0*sigma_value
            self.sigmam[:, -(ln+1), :] = mu0/eps0*sigma_value
            self.sigmam[:, :, -(ln+1)] = mu0/eps0*sigma_value
        

        # 電磁場の記録(それぞれshape==(nt+1,nx,ny,nz)の4次元ndarray)
        self.Ex = np.zeros(shape=(self.nt+1, self.nx, self.ny, self.nz))
        self.Ey = np.zeros(shape=(self.nt+1, self.nx, self.ny, self.nz))
        self.Ez = np.zeros(shape=(self.nt+1, self.nx, self.ny, self.nz))
        self.Hx = np.zeros(shape=(self.nt+1, self.nx, self.ny, self.nz))
        self.Hy = np.zeros(shape=(self.nt+1, self.nx, self.ny, self.nz))
        self.Hz = np.zeros(shape=(self.nt+1, self.nx, self.ny, self.nz))

        self.Jx = np.zeros(shape=(self.nt+1, self.nx, self.ny, self.nz))
        self.Jy = np.zeros(shape=(self.nt+1, self.nx, self.ny, self.nz))
        self.Jz = np.zeros(shape=(self.nt+1, self.nx, self.ny, self.nz))
 
        self.divE = np.zeros(shape=(self.nt+1, self.nx, self.ny, self.nz))
        self.divH = np.zeros(shape=(self.nt+1, self.nx, self.ny, self.nz))
        self.rotEx = np.zeros(shape=(self.nt+1, self.nx, self.ny, self.nz))
        self.rotEy = np.zeros(shape=(self.nt+1, self.nx, self.ny, self.nz))
        self.rotEz = np.zeros(shape=(self.nt+1, self.nx, self.ny, self.nz))
        self.rotHx = np.zeros(shape=(self.nt+1, self.nx, self.ny, self.nz))
        self.rotHy = np.zeros(shape=(self.nt+1, self.nx, self.ny, self.nz))
        self.rotHz = np.zeros(shape=(self.nt+1, self.nx, self.ny, self.nz))

        self.PEC = np.full((self.nx,self.ny,self.nz),False) # 電場が侵入できない仮想物質：完全電気導体が存在する領域でTrue


    
    def excite_EH(self,nE,nH,t): # 時刻tにおける電磁場の励起
        # nEは電場の離散固有時，nHは磁場の離散固有時
        # t == (nE+nH)/2*self.dt を満たす

        c = 2.99792458e8 # 光速[m/s]
        eps0 = 8.8541878128e-12 # 真空の誘電率[F/m]
        mu0 = 1.25663706212e-06 # 真空の透磁率[N/A^2]

        # 電場のz成分の励振
        omega = 1.0e10
        E0 = 30
        self.Ez[nE, self.nx//2, self.ny//2, self.nz//2] = E0*np.sin(-omega*t)

        """
        # 電場のy成分の励振
        omega = 1.0e9*2.0*np.pi
        E0 = 30
        self.Ey[nE,self.nx//2, self.ny//2, self.nz//2] = E0*np.sin(-omega*t)
        """

        # xy平面に平行な2枚の極板によるz軸方向の一様電場
        #self.Ez[nE, self.nx//4:self.nx//4*3, self.ny//4:self.ny//4*3, self.nz//5*2:self.nz//5*3] = 20

        """
        # xy平面内をx軸負方向から伝搬する直線偏光の平面波
        k = 2*np.pi/(0.1) # 波長10cmの波数
        omega = c*k # 角振動数
        E0 = 20 # 電場強度[V/m]
        self.Ez[nE, self.nx//4, :, :] = E0*np.sin(-omega*t)
        self.Hy[nH, self.nx//4, :, :] = -c*eps0*E0*np.sin(-omega*t)
        """
        """
        # 原点を通るz方向の電流
        I0 = 1 # [A]
        self.Jz[nH, self.nx//2, self.ny//2, :] = I0/(self.dx*self.dy) # [A/m^2]
        """
        """
        # 原点に1Cの電荷を配置
        q = 1
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    if (i,j,k) == (self.nx//2,self.ny//2,self.nz//2):
                        continue
                    r2 = ((i-self.nx//2)*self.dx)**2 + ((j-self.ny//2)*self.dy)**2 + ((k-self.nz//2)*self.dz)**2
                    coef = q/(4*np.pi*eps0) / (r2**(3/2))
                    Ex[i,j,k] = coef * ((i-self.nx//2)*self.dx)
                    Ey[i,j,k] = coef * ((j-self.ny//2)*self.dy) 
                    Ez[i,j,k] = coef * ((k-self.nz//2)*self.dz) 
        """
    
    def set_PEC(self,PEC): # PECの設置．PECが存在する領域でTrueとなる配列で指定する
        # PEC:np.ndarray(shape=(self.nx,self.ny,self.nz),dtype=bool)
        self.PEC = PEC
    
    def set_dielectric(self,epsr): # 誘電体の設置．比誘電率の配列で指定する
        # epsr:np.ndarray(shape=(self.nx,self.ny,self.nz),dtype=float) 
        eps0 = 8.8541878128e-12 # 真空の誘電率[F/m]
        self.eps = epsr*eps0

    def excute(self):
        # 差分化したMaxwell方程式の係数
        ch1 = (2.0*self.mu - self.sigmam*self.dt) / (2.0*self.mu + self.sigmam*self.dt)
        ch2 = 2.0 * self.dt / (2.0*self.mu + self.sigmam*self.dt) # 厳密な差分式では符号がマイナスになるはずだが，そうすると壊れるので便宜的に符号をプラスにとっている(なぜか参考文献でもこの式変形で符号ミスをして結果的に上手くいっている．なんで？)
        dhx = ch2 / self.dx
        dhy = ch2 / self.dy
        dhz = ch2 / self.dz

        ce1 = (2.0*self.eps - self.sigma*self.dt) / (2.0*self.eps + self.sigma*self.dt)
        ce2 = 2.0 * self.dt / (2.0*self.eps + self.sigma*self.dt)
        dex = ce2 / self.dx
        dey = ce2 / self.dy
        dez = ce2 / self.dz
        t = 0.0
        for n in tqdm(range(self.nt)): # tqdmでprogressバーの表示
            # 外部から電磁場を励起
            self.excite_EH(nE=n,nH=n,t=t)
            t += self.dt/2

            # 電場各成分計算
            self.Ex[n+1] = ce1 * self.Ex[n] + dey * (self.Hz[n] - np.roll(self.Hz[n], shift=1, axis=1))\
                                            - dez * (self.Hy[n] - np.roll(self.Hy[n], shift=1, axis=2)) + ce2*self.Jx[n]

            self.Ey[n+1] = ce1 * self.Ey[n] + dez * (self.Hx[n] - np.roll(self.Hx[n], shift=1, axis=2))\
                                            - dex * (self.Hz[n] - np.roll(self.Hz[n], shift=1, axis=0)) + ce2*self.Jy[n]

            self.Ez[n+1] = ce1 * self.Ez[n] + dex * (self.Hy[n] - np.roll(self.Hy[n], shift=1, axis=0))\
                                            - dey * (self.Hx[n] - np.roll(self.Hx[n], shift=1, axis=1)) + ce2*self.Jz[n]
            
            # np.rollで周期境界的になっているので端面を補正
            self.Ex[n+1,:,0,:] -=  dey[:,0,:] * (-np.roll(self.Hz[n], shift=1, axis=1)[:,0,:])
            self.Ex[n+1,:,:,0] -= -dez[:,:,0] * (-np.roll(self.Hy[n], shift=1, axis=2)[:,:,0])
            self.Ey[n+1,0,:,:] -= -dex[0,:,:] * (-np.roll(self.Hz[n], shift=1, axis=0)[0,:,:])
            self.Ey[n+1,:,:,0] -=  dez[:,:,0] * (-np.roll(self.Hx[n], shift=1, axis=2)[:,:,0])
            self.Ez[n+1,0,:,:] -=  dex[0,:,:] * (-np.roll(self.Hy[n], shift=1, axis=0)[0,:,:])
            self.Ez[n+1,:,0,:] -= -dey[:,0,:] * (-np.roll(self.Hx[n], shift=1, axis=1)[:,0,:])
            
            # PEC内部で電場は0
            self.Ex[n+1][self.PEC] = 0
            self.Ey[n+1][self.PEC] = 0
            self.Ez[n+1][self.PEC] = 0

            # 外部から電磁場を励起
            self.excite_EH(nE=n+1,nH=n,t=t) # 電場の時刻は磁場の時刻とずれていることに注意
            t += self.dt/2

            # 磁場各成分計算
            self.Hx[n+1] = ch1 * self.Hx[n] + dhy * (self.Ez[n+1] - np.roll(self.Ez[n+1], shift=-1, axis=1))\
                                            - dhz * (self.Ey[n+1] - np.roll(self.Ey[n+1], shift=-1, axis=2))

            self.Hy[n+1] = ch1 * self.Hy[n] + dhz * (self.Ex[n+1] - np.roll(self.Ex[n+1], shift=-1, axis=2))\
                                            - dhx * (self.Ez[n+1] - np.roll(self.Ez[n+1], shift=-1, axis=0))

            self.Hz[n+1] = ch1 * self.Hz[n] + dhx * (self.Ey[n+1] - np.roll(self.Ey[n+1], shift=-1, axis=0))\
                                            - dhy * (self.Ex[n+1] - np.roll(self.Ex[n+1], shift=-1, axis=1))
            
            # np.rollで周期境界的になっているので端面を補正
            self.Hx[n+1,:,-1,:] -=  dhy[:,-1,:] * (-np.roll(self.Ez[n+1], shift=-1, axis=1)[:,-1,:])
            self.Hx[n+1,:,:,-1] -= -dhz[:,:,-1] * (-np.roll(self.Ey[n+1], shift=-1, axis=2)[:,:,-1])
            self.Hy[n+1,-1,:,:] -= -dhx[-1,:,:] * (-np.roll(self.Ez[n+1], shift=-1, axis=0)[-1,:,:])
            self.Hy[n+1,:,:,-1] -=  dhz[:,:,-1] * (-np.roll(self.Ex[n+1], shift=-1, axis=2)[:,:,-1])
            self.Hz[n+1,-1,:,:] -=  dhx[-1,:,:] * (-np.roll(self.Ey[n+1], shift=-1, axis=0)[-1,:,:])
            self.Hz[n+1,:,-1,:] -= -dhy[:,-1,:] * (-np.roll(self.Ex[n+1], shift=-1, axis=1)[:,-1,:])
            

            # divとrotの計算
            self.divE[n] = (self.Ex[n] - np.roll(self.Ex[n],shift=1,axis=0))/self.dx + (self.Ey[n] - np.roll(self.Ey[n],shift=1,axis=1))/self.dy + (self.Ez[n] - np.roll(self.Ez[n],shift=1,axis=2))/self.dz
            self.rotEx[n] = (self.Ez[n] - np.roll(self.Ez[n],shift=1,axis=1))/self.dy - (self.Ey[n] - np.roll(self.Ey[n],shift=1,axis=2))/self.dz
            self.rotEy[n] = (self.Ex[n] - np.roll(self.Ex[n],shift=1,axis=2))/self.dz - (self.Ez[n] - np.roll(self.Ez[n],shift=1,axis=0))/self.dx
            self.rotEz[n] = (self.Ey[n] - np.roll(self.Ey[n],shift=1,axis=0))/self.dx - (self.Ex[n] - np.roll(self.Ex[n],shift=1,axis=1))/self.dy

            self.divH[n] = (self.Hx[n] - np.roll(self.Hx[n],shift=1,axis=0))/self.dx + (self.Hy[n] - np.roll(self.Hy[n],shift=1,axis=1))/self.dy + (self.Hz[n] - np.roll(self.Hz[n],shift=1,axis=2))/self.dz
            self.rotHx[n] = (self.Hz[n] - np.roll(self.Hz[n],shift=1,axis=1))/self.dy - (self.Ey[n] - np.roll(self.Hy[n],shift=1,axis=2))/self.dz
            self.rotHy[n] = (self.Hx[n] - np.roll(self.Hx[n],shift=1,axis=2))/self.dz - (self.Ez[n] - np.roll(self.Hz[n],shift=1,axis=0))/self.dx
            self.rotHz[n] = (self.Hy[n] - np.roll(self.Hy[n],shift=1,axis=0))/self.dx - (self.Ex[n] - np.roll(self.Hx[n],shift=1,axis=1))/self.dy



    def visualize_3plane(self,title,lim_val=None): # xy平面を正面とした正投影図(第三角法)で電磁場の時間変遷をアニメ化
        try:
            data = self.__dict__[title]
        except:
            print("電場Eのz成分Ezを表示します")
            data = self.Ez
            title = "Ez"
        if lim_val is None:
            lim_val = max(np.abs(np.max(data)),np.abs(np.min(data)))

        fig, ax = plt.subplots(2,2,figsize=(9,8))
        ax[0,1].axis('off')
        #plt.tight_layout()

        x = np.array([i*self.dx for i in range(self.nx)])
        y = np.array([i*self.dy for i in range(self.ny)])
        z = np.array([i*self.dy for i in range(self.ny)])
        xy, yx = np.meshgrid(x,y)
        yz, zy = np.meshgrid(y,z)
        zx, xz = np.meshgrid(z,x)


        # ax.pcolor()でshading="flat"にすると最後の1行1列のデータが落ちるのでWarningが出る
        cax0 = ax[0,0].pcolor(xz, zx, data[0][:,self.ny//2,:], shading="nearest", cmap="jet", vmin=-lim_val, vmax=lim_val)
        cax2 = ax[1,0].pcolor(xy, yx, data[0][:,:,self.nz//2].T, shading="nearest", cmap="jet", vmin=-lim_val, vmax=lim_val)
        cax3 = ax[1,1].pcolor(zy, yz, data[0][self.nx//2,:,:].T, shading="nearest", cmap="jet", vmin=-lim_val, vmax=lim_val)
        #fig.colorbar(cax2)
        #fig.colorbar(cax3)
        fig.colorbar(cax0,ax=ax.flat)

        ax[0,0].set_aspect('equal')
        ax[1,0].set_aspect('equal')
        ax[1,1].set_aspect('equal')

        #ax.plot(self.mx, self.my, '^', lw=2, color='black', ms=5, label='マイク')
        #ax.legend()
    
        def animate(i):
            i = i%len(data)

            ax[0,0].set_title(title+f"\nt={i}dt")
            ax[0,0].set_xlabel("x")
            ax[0,0].set_ylabel("z")
            cax0.set_array(data[i][:,self.ny//2,:].flatten())

            #ax[1,0].set_title(title+f"\nt={i}dt")
            ax[1,0].set_xlabel("x")
            ax[1,0].set_ylabel("y")
            cax2.set_array(data[i][:,:,self.nz//2].T.flatten())

            #ax[1,1].set_title(title+f"\nt={i}dt")
            ax[1,1].set_xlabel("z")
            ax[1,1].set_ylabel("y")
            cax3.set_array(data[i][self.nx//2,:,:].T.flatten())

        ani = animation.FuncAnimation(fig, animate, interval=100, frames=1000) # interval[ms]
        plt.show()
    


    def visualize_vectorfield(self,name="E"): # (規格化した)3次元ベクトル場で電磁場の時間変遷を3Dアニメ化
        if not name in ("E","H","rotE","rotH"):
            print("電場Eのベクトル場を表示します")
            name = "E"
        fig = plt.figure()
        x = np.array([i*self.dx for i in range(self.nx)])
        y = np.array([i*self.dy for i in range(self.ny)])
        z = np.array([i*self.dz for i in range(self.nz)])
        idx = np.arange(0,self.nx,25)

        #X,Y,Z = np.meshgrid(x[idx],y[idx],z[idx]) # <-間違い
        Y,Z,X = np.meshgrid(y[idx],z[idx],x[idx])
        IDX = np.ix_(idx,idx,idx)
        ax = fig.gca(projection='3d')
        
        def animate(i):
            i = i%self.nt
            ax.clear()
            ax.set_title(name+f"\nt={i}dt")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_xlim([x[0], x[-1]])
            ax.set_ylim([y[0], y[-1]])
            ax.set_zlim([z[0], z[-1]])

            # ベクトル場を描画
            if name=="E":
                ax.quiver(X,Y,Z,self.Ex[i][IDX],self.Ey[i][IDX],self.Ez[i][IDX],length=0.1,normalize=True,color="blue")
            elif name=="H":
                ax.quiver(X,Y,Z,self.Hx[i][IDX],self.Hy[i][IDX],self.Hz[i][IDX],length=0.1,normalize=True,color="blue")
            elif name=="rotE":
                ax.quiver(X,Y,Z,self.rotEx[i][IDX],self.rotEy[i][IDX],self.rotEz[i][IDX],length=0.1,normalize=True,color="blue")
            elif name=="rotH":
                ax.quiver(X,Y,Z,self.rotHx[i][IDX],self.rotHy[i][IDX],self.rotHz[i][IDX],length=0.1,normalize=True,color="blue")
            else:
                print("ここは実行されない")
                raise RuntimeError
        ani = animation.FuncAnimation(fig, animate, interval=100, frames=1000) # interval[ms]
        plt.show()

    def visualize_wireframe(self,title,plane="xy",lim_val=None): # 電磁場を(xy,yz,zx)平面で切ったときの値の時間変遷をアニメ化
        try:
            data = self.__dict__[title]
        except:
            print("電場Eのz成分Ezを表示します")
            data = self.Ez
            title = "Ez"
        if lim_val is None:
            lim_val = max(np.abs(np.max(data)),np.abs(np.min(data)))

        if plane=="xy":
            u = np.array([i*self.dx for i in range(self.nx)])
            v = np.array([i*self.dy for i in range(self.ny)])
        elif plane=="yz":
            u = np.array([i*self.dy for i in range(self.ny)])
            v = np.array([i*self.dz for i in range(self.nz)])
        elif plane=="zx":
            u = np.array([i*self.dz for i in range(self.nz)])
            v = np.array([i*self.dx for i in range(self.nx)])
        else:
            print("xy平面上の値を表示します")
            u = np.array([i*self.dx for i in range(self.nx)])
            v = np.array([i*self.dy for i in range(self.ny)])

        fig = plt.figure()
        V,U = np.meshgrid(v,u)
        
        ax = fig.gca(projection='3d')
    
        def animate(i):
            i = i%len(data)
            ax.clear()
            ax.set_title(title+f"\nt={i}dt")
            ax.set_xlabel(plane[0])
            ax.set_ylabel(plane[1])
            ax.set_zlabel(title)
            ax.set_zlim(-lim_val, lim_val)
            if plane=="yz":
                ax.plot_wireframe(U,V,data[i][self.nx//2,:,:], rstride=1, cstride=1, color='blue', linewidth=0.2) 
            elif plane=="zx":
                ax.plot_wireframe(U,V,data[i][:,self.ny//2,:], rstride=1, cstride=1, color='blue', linewidth=0.2) 
            else:
                ax.plot_wireframe(U,V,data[i][:,:,self.nz//2], rstride=1, cstride=1, color='blue', linewidth=0.2) 
        ani = animation.FuncAnimation(fig, animate, interval=100, frames=1000) # interval[ms]
        plt.show()



    def visualize_scatter(self,title): # 電磁場を(xy,yz,zx)平面で切ったときの値の時間変遷をアニメ化
        try:
            data = self.__dict__[title]
        except:
            print("電場Eのz成分Ezを表示します")
            data = self.Ez
            title = "Ez"

        fig = plt.figure()
        
        x = np.array([i*self.dx for i in range(self.nx)])
        y = np.array([i*self.dy for i in range(self.ny)])
        z = np.array([i*self.dz for i in range(self.nz)])
        idx = np.arange(0,self.nx,20)
        IDX = np.ix_(idx,idx,idx)

        
        Y,Z,X = np.meshgrid(y[idx],z[idx],x[idx])

        lim_val = max(np.abs(np.max(data)),np.abs(np.min(data)))
        ax = fig.gca(projection='3d')
        sc = ax.scatter(X,Y,Z, c=data[0][IDX], alpha=0.5, cmap='jet')
        #fig.colorbar(sc)
    
        def animate(i):
            i = i%len(data)
            ax.clear()
            ax.set_title(title+f"\nt={i}dt")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")            
            ax.scatter(X,Y,Z, c=data[i][IDX], alpha=0.5, cmap='jet')

        ani = animation.FuncAnimation(fig, animate, interval=200, frames=1000) # interval[ms]
        plt.show()



if __name__ == "__main__":
    # 調整するところはexcite_EHと誘電率の空間分布
    nx = 100; ny = 100; nz = 100; nt = 150

    F = FDTD3D(nx=nx,ny=ny,nz=nz,nt=nt)

    # PECの設置
    PEC = np.full((nx,ny,nz),False)
    PEC[nx//3:nx//3+2, ny//6:ny//6*4, nz//6:nz//6*4] = True
    PEC[nx//3*2:nx//3*2+2, ny//6:ny//6*4, nz//6:nz//6*4] = True
    F.set_PEC(PEC)

    # 誘電体の設置
    Dielectric = np.ones((nx,ny,nz))
    Dielectric[:nx//3,:,:] = 10
    F.set_dielectric(Dielectric)

    # 実行
    F.excute()
    
    # 描画
    limval = 0.1
    #limval = None
    F.visualize_3plane(title="Ex",lim_val=limval)
    F.visualize_wireframe(title="Ex",plane="xy",lim_val=limval)
    F.visualize_3plane(title="Ey",lim_val=limval)
    F.visualize_wireframe(title="Ey",plane="xy",lim_val=limval)
    F.visualize_3plane(title="Ez",lim_val=limval)
    F.visualize_wireframe(title="Ez",plane="xy",lim_val=limval)
    
    #F.visualize_3plane(title="Hx")
    #F.visualize_wireframe(title="Hx",plane="xy")
    #F.visualize_3plane(title="Hy")
    #F.visualize_wireframe(title="Hy",plane="xy")
    #F.visualize_3plane(title="Hz")
    #F.visualize_wireframe(title="Hz",plane="xy")
    
    #F.visualize_vectorfield(name="E")
    #F.visualize_vectorfield(name="H")
    #F.visualize_vectorfield(name="rotE")    
    #F.visualize_vectorfield(name="rotH")
    
    #F.visualize_scatter(title="divE")
    #F.visualize_scatter(title="divH")





