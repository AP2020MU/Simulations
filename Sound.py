# FDTD法(Finite-Difference Time-Domain method)による音響シミュレーション
# https://github.com/samuiui/blogs/blob/master/blog_fdtd/fdtd_notebook.ipynb
# を元に，多くの変更を加えてクラス化した
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as Path
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib import cm
from scipy.stats import norm
import japanize_matplotlib



class SoundSimulation:
    def __init__(self, temperature=15, time_step=2000):
        atm = 1.013 
        self.temperature = temperature
        self.C = 331.5 + 0.60*temperature 

        # 大気密度
        self.Rho = (1.293*atm)/(1+(temperature/273.15)) 
        # 気体弾性率
        self.K = self.Rho*self.C*self.C

        print(f"sound velocity: {self.C} [m/s]")
        print(f"atmosphere density: {round(self.Rho, 3)} [kg/m^3]")
        print(f"bulk module: {round(self.K, 0)} [Pa]")

        # space discrete width [m]
        self.dx = 0.01
        self.dy = self.dx

        # time discrete width [s]
        self.dt = 0.00001

        # sampling rate [Hz]
        fs = 1/self.dt
        print(f"sampling rate: {round(fs)} [Hz]")

        # number of calculation
        self.time_step = time_step
        sec = self.time_step*self.dt
        print(f"calclate time: {round(sec, 6)} [sec]")


        # CFL条件(Courant-Friedrichs-Lewy Condition)
        # 情報を伝播させる速さが実際の音が伝わる速さより遅くならないようにするための条件
        # 情報が伝播する速さは空間分解能を時間分解能を割った時の商，音の伝わる速さは音速になる
        # 空間的に細かく追跡する場合は時間幅を狭くする必要があるため所望する時間までの計算時間が増加
        if self.dx/self.dt > self.C:
            print("CFL condition: OK ")
        else :
            print("CFL condition: NG")

        # 各時刻における音圧場
        self.P_history = []

    def make_room(self, shape_num=0, Lx=4., Ly=3., source=(0.3,0.4), mic=(1.0,1.0), ab=0.00001, Zr=0.1): # Zr:空気に対する壁の音響インピーダンス比
        # 室形状設定
        # shape_numで選択する
        # 0(default):矩形領域
        # 1:下辺が「く」
        # 2:下辺が曲面
        # 3:左辺がギザギザ
        # 4:矩形+突起
        if shape_num == 1: # 下辺が「く」
            verts = np.array([
                [-Lx/2,  Ly/2], # left, top
                [-Lx/2, -Ly/2], # left, bottom
                [0, 0],   # KU point
                [ Lx/2, -Ly/2], # right, bottom
                [ Lx/2,  Ly/2], # right, top
                [-Lx/2,  Ly/2]  # return to start point
                ])
            codes = np.array([1, 2, 2, 2, 2, 79]) # 1:MOVETO, 2:LINETO, 3:CURVETO 79:CLOSEPOLY
        elif shape_num == 2: # 下辺が曲面
            verts = np.array([
                [-2,  1.5], # left, top
                [-2, -1.5], # left, bottom
                [0, 1.0],   # curve point
                [ 2, -1.5], # right, bottom
                [ 2,  1.5], # right, top
                [-2,  1.5]  # return to start point
                ])
            codes = np.array([1, 2, 3, 2, 2, 79]) # 1:MOVETO, 2:LINETO, 3:CURVETO 79:CLOSEPOLY
        elif shape_num == 3: # 左辺がギザギザ
            GIZA = 6
            verts = []
            d = Ly/GIZA/2
            for i in range(GIZA):
                verts.append([-Lx/2,Ly/2-d*2*i])
                verts.append([-Lx/2+d,Ly/2-d*(2*i+1)])
            verts.append([-Lx/2, -Ly/2])
            verts.append([ Lx/2, -Ly/2])
            verts.append([ Lx/2,  Ly/2])
            verts.append([-Lx/2,  Ly/2])
            verts = np.array(verts)
            codes = np.array([1] + [2]*(GIZA*2+2) + [79]) # 1:MOVETO, 2:LINETO, 3:CURVETO 79:CLOSEPOLY
        elif shape_num == 4: # 矩形+突起
            verts = np.array([
                [-Lx/2,  Ly/2], # left, top
                [-Lx/2, -Ly/2], # left, bottom
                [-Lx/40, -Ly/2], # wall, left, bottom
                [-Lx/40, -Ly/6], # wall, left, top
                [ Lx/40, -Ly/6], # wall, right, top
                [ Lx/40, -Ly/2], # wall ,right, bottom
                [ Lx/2, -Ly/2], # right, bottom
                [ Lx/2,  Ly/2], # right, top
                [-Lx/2,  Ly/2]  # return to start point
                ])
            codes = np.array([1, 2, 2, 2, 2, 2, 2, 2, 79]) # 1:MOVETO, 2:LINETO, 79:CLOSEPOLY
        else: # 矩形領域
            verts = np.array([
                [-Lx/2,  Ly/2], # left, top
                [-Lx/2, -Ly/2], # left, bottom
                [ Lx/2, -Ly/2], # right, bottom
                [ Lx/2,  Ly/2], # right, top
                [-Lx/2,  Ly/2]  # return to start point
                ])
            codes = np.array([1, 2, 2, 2, 79]) # 1:MOVETO, 2:LINETO, 79:CLOSEPOLY



        # 音源の座標
        self.sx, self.sy = source
        
        # 受音体の座標
        self.mx, self.my = mic

        # 音響インピーダンス
        self.Z = self.Rho*self.C*((1+np.sqrt(1-ab)) / (1-np.sqrt(1-ab))) # 原典のコード
        
        # 粒子速度反射率(応用音響学第7回参照)
        self.Ru = -(Zr-1)/(Zr+1)
        
        """
        # 室形状表示
        pth = Path.Path(verts, codes)
        patch = patches.PathPatch(pth, facecolor='none', lw=2, ls='--')
        plt.gca().add_patch(patch)
        plt.xlim(-3,3)
        plt.ylim(-2,2)
        plt.axis('equal')
        plt.show()
        """

        # 1. 頂点の最大/最小値を取得
        self.pth = Path.Path(verts, codes)
        self.x_min = min(verts[:,0])
        self.x_max = max(verts[:,0])
        self.y_min = min(verts[:,1])
        self.y_max = max(verts[:,1])

        # 2. 最小値から最大値の間を空間解像度幅で分割した配列を作成
        self.x = np.arange(self.x_min+0.5*self.dx, self.x_max+0.5*self.dx, self.dx)
        self.y = np.arange(self.y_min+0.5*self.dy, self.y_max+0.5*self.dy, self.dy)

        self.Nx = len(self.x)
        self.Ny = len(self.y)

        # 点(x[i],y[j])が設定した領域内部の点かどうかの配列
        bools = np.zeros((self.Nx,self.Ny), dtype="bool")

        # 3. 1で求めた凸包的矩形領域のうち，どの点が内部の点か調べる
        for j in range(self.Ny):
            points = [[self.x[i], self.y[j]] for i in range(self.Nx)]
            bools[:,j] = self.pth.contains_points(points)

        # 4. 内部と外部の境界を記録しておく
        # bound_*に入っている点は全て領域内部の点で，そこから1マス動くと外に出る
        bound_x_left = []
        bound_x_right = []
        for i in range(self.Nx-1):
            for j in range(self.Ny):
                if bools[i,j] == 0 and bools[i+1,j] == 1:
                    bound_x_left.append([i+1,j])
                elif bools[i,j] == 1 and bools[i+1,j] == 0:
                    bound_x_right.append([i,j])
        self.bound_x_left = np.array(bound_x_left)
        self.bound_x_right = np.array(bound_x_right)


        bound_y_down = []
        bound_y_up = []
        for i in range(self.Nx):
            for j in range(self.Ny-1):
                if bools[i,j] == 0 and bools[i,j+1] == 1:
                    bound_y_down.append([i,j+1])
                elif bools[i,j] == 1 and bools[i,j+1] == 0:
                    bound_y_up.append([i,j])
        self.bound_y_down = np.array(bound_y_down)
        self.bound_y_up = np.array(bound_y_up)

    def make_signal(self):
        # 音源の生成
        # 正弦パルスによる音圧の初期化
        # 音響分野でよく用いられるものは数セルにわたってなめらかな分布をもつ初期条件を与える手法
        ang = np.arange(-np.pi, np.pi, 2.*np.pi/50.)
        self.sig = 1+np.cos(ang)


        """
        # 点音源の差分化
        # 点音源での音圧を，点音源から放出される体積速度$Q(t)$で表現
        # 特に差分化したQ(n)=ガウシアンパルスの場合
        ang = np.arange(-np.pi,np.pi,2.*np.pi/50.)
        sig = norm.pdf(ang)
        sig *= 4
        """

        """
        # 音源の波形と周波数特性
        plt.subplot(2,1,1)
        plt.plot(sig)
        plt.xlabel("Sample")
        plt.ylabel("Relative Sound Pressure")

        plt.subplot(2,1,2)
        plt.magnitude_spectrum(sig, Fs=1/self.dt, scale='dB')

        plt.show()
        """

    def FDTD(self):
        # 計算用アレイの定義・初期化
        P = np.zeros((self.Nx,self.Ny),"float64") # 音圧場
        self.P_history.append(P)
        Ux = np.zeros((self.Nx+1,self.Ny),"float64") # x軸方向の速度
        Uy = np.zeros((self.Nx,self.Ny+1),"float64") # y軸方向の速度

        # マイクの位置に対応するindexとそこでの音圧変化
        micx = int((self.mx - self.x_min)/self.dx)
        micy = int((self.my - self.y_min)/self.dy)
        mic_received = []

        # 音源の位置に対応するindex
        ssx = int((self.sx - self.x_min)/self.dx)
        ssy = int((self.sy - self.y_min)/self.dy)

        # 逐次計算
        for t in range(self.time_step):
            sys.stdout.flush()
            sys.stdout.write("\r{}".format("calculating: "+str(t+1)+"/"+str(self.time_step)))
            
            # 音源から一定時間の間，音圧を供給(div P)
            if t<len(self.sig):
                P[ssx,ssy] += self.sig[t]
                
            # 各軸方向の運動方程式を差分化して，粒子速度の更新式を導いた
            Ux[1:self.Nx,:] = Ux[1:self.Nx,:] - self.dt/self.Rho/self.dx*(P[1:self.Nx,:]-P[:self.Nx-1,:])  # x軸の粒子速度更新
            Uy[:,1:self.Ny] = Uy[:,1:self.Ny] - self.dt/self.Rho/self.dy*(P[:,1:self.Ny]-P[:,:self.Ny-1])  # y軸の粒子速度更新
            
            ### 境界条件は少し考える必要がありそう
            ### 反射・屈折・エネルギー保存の法則が壊れそう
            # 境界におけるx方向の粒子速度
            for x, y in self.bound_x_left:
                Ux[x,y] = -P[x,y]/self.Z # 原典のコード
                #Ux[x,y] += Ux[x,y]*self.Ru
            for x, y in self.bound_x_right:
                Ux[x+1,y] = 0.0
            
            # 境界におけるy方向の粒子速度
            for x, y in self.bound_y_down:
                Uy[x,y] = 0.0 # 原典のコードによれば境界の内側でy軸方向の粒子速度は0になるらしいけど正しい？
                #Uy[x,y] = -P[x,y]/self.Z
                #Uy[x,y] += Uy[x,y]*self.Ru
            for x, y in self.bound_y_up:
                Uy[x,y+1] = 0.0

            # 各軸方向の連続の式を差分化して，音圧の更新式を導いた
            P[:self.Nx,:self.Ny] = P[:self.Nx,:self.Ny] - self.K*self.dt/self.dx*(Ux[1:self.Nx+1,:]-Ux[:self.Nx,:]) - self.K*self.dt/self.dy*(Uy[:,1:self.Ny+1]-Uy[:,:self.Ny])
            self.P_history.append(P.copy())
            if t > 100:
                mic_received.append(P[micx, micy])
        self.mic_received = mic_received
        self.P = P
        

    
    def visualize(self):
        # 可視化パート
        
        """
        # 最終的な音圧場
        yy, xx = np.meshgrid(self.y,self.x)
        plt.pcolor(xx, yy, self.P, cmap=cm.jet)

        patch = patches.PathPatch(self.pth, facecolor='none', lw=2, ls='--')
        plt.gca().add_patch(patch)
        plt.axis('equal')
        #plt.clim(-0.05, 0.05)
        plt.colorbar()
        plt.plot(self.sx, self.sy, 'o', lw=2, color='black', ms=10, label='音源')
        plt.plot(self.mx, self.my, '^', lw=2, color='black', ms=10, label='マイク')
        plt.title(f'音圧分布\n(1気圧 {self.temperature}℃)')
        plt.legend()
        plt.show()
        """

        # 音圧場の時間変遷
        fig, ax = plt.subplots()
        yy, xx = np.meshgrid(self.y,self.x)
        data = self.P_history
        #print(self.P_history)
        cax = ax.pcolor(xx, yy, data[0], cmap=cm.jet)
        fig.colorbar(cax)

        ax.set_aspect('equal')
        patch = patches.PathPatch(self.pth, facecolor='none', lw=2, ls='--')
        plt.gca().add_patch(patch)
        ax.plot(self.sx, self.sy, 'o', lw=2, color='black', ms=5, label='音源')
        ax.plot(self.mx, self.my, '^', lw=2, color='black', ms=5, label='マイク')
        ax.legend()
    
        def animate(i):
            #plt.rcParams['figure.figsize'] = (7,6)
            i = i%len(self.P_history)
            ax.set_title(f"音圧分布\n(1気圧 {self.temperature}℃)：t={i}")
            # 注意！
            # ax.pcolorは元データに対して1まわり大きい配列で実現されるので，
            # ただ単にflatten()してset_arrayに渡すと，端っこの余りの影響で画像が崩れる．
            # 以下のように行と列のサイズをそれぞれ1だけ小さくしておくこと
            cax.set_array(data[i][:self.Nx-1,:self.Ny-1].flatten()) 

        ani = animation.FuncAnimation(fig, animate, interval=1, frames=self.time_step)
        plt.show()

        # マイクロホン位置での音圧
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.plot(self.mic_received)
        plt.show()


S = SoundSimulation(temperature=20,time_step=2000)

S.make_room(shape_num=0,Lx=3,Ly=3,source=(0.,0.), mic=(1.0,1.0),Zr=1)
S.make_signal()
S.FDTD()
S.visualize()