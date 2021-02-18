"""
2次元Ising模型のMonte Carlo Simulation
"""
from random import random, randrange
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import datetime
from collections import deque
import japanize_matplotlib 
import os
import scipy.special as sps

# $ pip install japanize_matplotlib
# でインストール可能．実行環境ですでにmatplotlibのグラフ上に日本語フォントを表示できるならこれは不要．
# japanize_matplotlibをインストールしたくない，または，matplotlibで日本語を表示できない場合は，
# 下のクラスメソッドのうち"visualize_*()"の中のtitleやlabelから日本語を消す必要がある

# グラフ描画
def make_figure(X,Y,titlename='',Xlabel='',Ylabel='',label='',plot_type='scatter',point_size=5,pcolor='tab:blue',lcolor='tab:blue',lwidth=1.0,fig_size=(8,6),Xlog=False,Ylog=False,Yerr=None,consecutive=None,filename=''): 
    # consecutive==Noneのとき，1回の描画で完結
    ## consecutive==1のとき，連続して描画する際の1枚目を表す
    ## consecutive==0のとき，連続して描画する際の中間に来る描画を表す
    ## consecutive==-1のとき，連続して描画する際の最後の描画を表す
    plt.rcParams["font.size"] = 12
    if consecutive is None or consecutive == 1:
        plt.figure(figsize=fig_size) 
    if plot_type == 'scatter':
        plt.scatter(X,Y,s=point_size,label=label,color=pcolor)
    elif plot_type == 'plot':
        plt.plot(X,Y,markersize=point_size,label=label,color=lcolor,linewidth=lwidth)
    elif plot_type == 'errorbar':
        plt.errorbar(X,Y,yerr=Yerr,markersize=point_size, capsize=3, fmt='o', ecolor='black', markeredgecolor="black", color=pcolor, elinewidth=0.5, capthick=0.5,label=label)

    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    if Xlog:
        plt.xscale('log')
        plt.xlim(np.min(X)/5,np.max(X)*5)
    if Ylog:
        plt.yscale('log')
        plt.ylim(np.min(Y)/5,np.max(Y)*5)
    plt.title(titlename)
    if not label == '':
        plt.legend()
    if consecutive is None or consecutive == -1:
        if filename != '':
            plt.savefig(filename+'.png', dpi=300)
        plt.show()

class Ising2():
    def __init__(self,Ny,Nx,J,H,dir_name,max_kBT=6,steps_kBT=200,N_MCS=40000,sampling_period=1,equilibrium_num=2,Q=500,cold=True):
        # Ising Modelにおける通常のHamiltonian: H=-JΣ(<i,j>:隣接スピン対)SiSj-BΣSi
        def Hamiltonian(spin_table):
            def closest_spin_pair(spins): # 最近隣接スピン対<i,j>について,Si*Sjを足し上げる．また，周期境界条件を課した
                # スピン行列を1つ下にずらしたものと1つ右にずらしたものを足して，
                # 元の行列とアダマール積(要素ごとの積で行列積ではない)をとると，最近隣接スピン対<i,j>についてSi*Sjを足し上げたものが出てくる
                # Pythonではfor文より行列操作の方が圧倒的に速い(サイズにもよるが数十倍程度も違う)のでこの方式を採用
                up = np.roll(spins, 1, axis=0)
                left = np.roll(spins, 1, axis=1)
                ans = np.sum(spins*(up+left))
                return ans
            return -self.J*closest_spin_pair(spin_table)-self.H*np.sum(spin_table)
        def delta_Hamiltonian(s,i,j): # もし(i,j)のスピンを反転させたら生じるであろうエネルギー変化
            return 2*s[i,j]*self.J*(s[(i+1)%self.Ny,j]+s[i-1,j]+s[i,(j+1)%self.Nx]+s[i,j-1]) + 2*self.H*s[i,j]

        self.Ny = Ny # 行方向(y軸方向)のスピンの数
        self.Nx = Nx # 列方向(x軸方向)のスピンの数
        self.Ntot = Ny*Nx # 総スピン数
        """
        関連して,二重ループの時は,
        for i in range(self.Ny):
            for j in range(self.Nx):
                pass
        となり, i<->y, j<->x, という対応になることに注意
        """
        self.J = J # スピン間相互作用エネルギー
        self.H = H # 外部磁場
        self.max_kBT = max_kBT # 計算する最大のkBT
        self.steps_kBT = steps_kBT # (0,max_kBT]の分割数(刻み)
        self.N_MCS = N_MCS # 総モンテカルロステップ(可視化の時に使う)
        self.Ps = sampling_period # サンプリング周期
        self.equilibrium_num = equilibrium_num # 平衡状態に至るまでの総モンテカルロステップに対する割合の逆数
        self.Q = Q # サンプル状態数(計算で用いることはないがデータ保存時のファイル名に使用)
        ### N_MCS == Ps * equilibrium_num * Q
        self.cold = cold # True:kBT=0.0001からスタートする, False:kBT=max_kBTからスタートするか
        
        self.Hamiltonian = Hamiltonian
        self.delta_Hamiltonian = delta_Hamiltonian
        self.MCmode = '' # 熱浴法(GibbsSampler，Metropolis，SwendsenWang，Wolffから選択(self.MonteCarlo(MCmode='')で選択)

        self.dir_path = './'+dir_name+'/' # データ保存先のディレクトリパス(カレントディレクトリ内部に該当するディレクトリ存在する必要がある．存在しないとエラー)
        if not os.path.isdir(self.dir_path):
            print('対象となるディレクトリが存在しません．"dir_name"を変更してください')
            raise FileNotFoundError

        self.kBTc = 2*self.J/np.log(1+np.sqrt(2))

        #self.kBT_list = np.linspace(0.0001,max_kBT, steps_kBT) # 温度を0.0001からmax_kBTまで(kBT単位で)steps_kBT刻みで変動させる(kBT=0にすると比熱が吹き飛ぶ)
        
        min_exp = -4
        steps_under_kBTc = int(steps_kBT/(np.log(self.kBTc)-min_exp+np.log(self.max_kBT-self.kBTc)-min_exp)*(np.log(self.kBTc)-min_exp))
        # kBT_listは，T=Tc近傍が密になるように非線形的に増加する
        # 具体的には，kBTc=0.0001~(kBTc-e^min_exp),(kBTc+e^min_exp)~Max_kBT の2つの区間で対数スケールで等間隔になっている
        # 参考値:e^(-3)~0.050，e^(-4)~0.018，e^(-5)~0.0067
        self.kBT_list = np.concatenate([self.kBTc-np.logspace(np.log(self.kBTc-0.0001),min_exp,num=steps_under_kBTc,base=np.e), [self.kBTc], self.kBTc+np.logspace(min_exp,np.log(self.max_kBT-self.kBTc),num=self.steps_kBT-steps_under_kBTc-1,base=np.e)])
        
        if not self.cold:
            self.kBT_list = self.kBT_list[::-1] # 逆順(高温側からシミュレート)
            self.start = 'hotstart' # グラフなどの体裁に使用
        else:
            self.start = 'coldstart' # グラフなどの体裁に使用

        self.E_history = [] # kBT=0~max_kBTにおける1スピンあたりエネルギーの変遷
        self.M_history = [] # kBT=0~max_kBTにおける1スピンあたり磁化の変遷
        self.C_history = [] # kBT=0~max_kBTにおける1スピンあたり比熱の変遷
        self.X_history = [] # kBT=0~max_kBTにおける1スピンあたり磁化率の変遷
        self.S_history = [] # kBT=0~max_kBTにおけるスピン全体の変遷

    # Ising模型MCMC法の実行関数
    def MonteCarlo(self,MCmode,is_save=True): 
        self.MCmode = MCmode
        if is_save:
            try:
                os.mkdir(self.dir_path+self.MCmode+'/')
            except FileExistsError:
                print('過去に保存したデータが上書きされる可能性があるのでプログラムを終了します．実行用の空ディレクトリを新たに用意してください')
                raise FileExistsError

        
        # スピンの初期化
        # hotstart()はcoldstart()に比べて35倍ほど遅い上に,収束が遅い(?,不安定になった)ので非推奨
        def coldstart(): # 完全に揃った状態(s=1)を初期状態とする
            return np.ones((self.Ny,self.Nx),dtype=int)
        def hotstart(): # 完全にランダムなスピン状態を初期状態とする
            return np.random.choice([-1,1],(self.Ny,self.Nx))

        # Wolffのアルゴリズム
        # 高温になるほどクラスターの大きさが小さくなるので高速に動作する
        def Wolff_algorithm(s_before, e_2JkBT, E_kBT): # 実質的にgrid上のbfs((sx,sy)からスタート)
            ### Wolff algorithmは,下のようにランダムに一つのスピンを選んでそれが属するclusterを確率的にひっくり返すことを繰り返す
            r = randrange(self.Ntot-1) # 0からNy*Nx-1まででランダムな数が選ばれる
            # つまり,スピンの位置(i,j)が一様な確率で選ばれる
            sy = r // self.Nx # 第i行
            sx = r % self.Nx # 第j列
            dire = ((0, -1), (-1, 0), (1, 0), (0, 1))
            cluster = np.full((self.Ny, self.Nx), -1) # cluster外は-1, clusterに含まれるなら1
            cluster[sy][sx] = 1
            que = deque([(sy, sx)]) # (sx, sy)からスタート
            s_trial = s_before.copy()
            while que:
                y, x = que.popleft()
                for dy, dx in dire:
                    ny,nx = (y+dy)%self.Ny,(x+dx)%self.Nx
                    # スピンがs[i,j]と同じで未訪問なら確率的に採択し,距離を更新してdequeに入れる
                    if s_trial[ny,nx] == s_trial[sy,sx] and cluster[ny,nx] == -1:
                        if random() < 1-e_2JkBT: # 確率p=1-exp(-2βJ)でclusterに入れる
                            cluster[ny,nx] = 1
                            que.append((ny, nx))
            
            s_trial = np.where(cluster==1,s_trial*-1,s_trial) # 今考えているclusterに属する(cluster==1)スピンを反転
            return s_trial, self.Hamiltonian(s_trial)


        class UnionFind:
            def __init__(self, n): # O(n)
                # parent[i]にはi番目のノードの親の番号を格納し，
                # 自分が根だった場合は-(自分が属する連結集合のサイズ)とする
                self.parent = [-1 for _ in range(n)]
                self.n = n
            def root(self, x): # 要素xの根の番号を返す O(α(n))
                if self.parent[x] < 0: # 自分が根のとき
                    return x
                else:
                    # 要素xの親を要素xの根に付け替えることで次の呼び出しの高速化
                    self.parent[x] = self.root(self.parent[x]) # 要素xの親を要素xの根に変えておく(付け替える)
                    return self.parent[x]
            def size(self, x): # 要素xの所属するグループの要素数を調べる O(α(n))
                return -self.parent[self.root(x)] #根のparentにサイズが格納されている
            def merge(self, x, y): # xとyを結合する O(α(n))
                x = self.root(x)
                y = self.root(y)
                if x == y:
                    return False 
                if self.parent[x] > self.parent[y]: # 大きい方(x)に小さい方(y)をぶら下げる
                    x, y = y, x 
                self.parent[x] += self.parent[y]
                self.parent[y] = x
                return True 
            
        
        # Swendsen-Wangのアルゴリズム
        def SwendsenWang_algorithm(s_before, e_2JkBT, E_kBT): 
            ### Swendsen-Wang algorithmは，全てのスピンをclusteringして一度に全部独立に確率的にひっくり返すことを繰り返す
            dire = ((1, 0), (0, 1))
            s_trial = s_before.copy()
            UF = UnionFind(self.Ntot)
            for y in range(self.Ny):
                for x in range(self.Nx):
                    for dy, dx in dire: # 各スピンについて横と下を見ていけば，全てのスピン対について結合判定ができる(周期境界条件を考慮の上)
                        ny,nx = (y+dy)%self.Ny,(x+dx)%self.Nx
                        # s[ny][nx]がs[y][x]と同じなら確率的に採択し，距離を更新してdequeに入れる
                        if s_trial[ny,nx] == s_trial[y,x]:
                            if random() < 1-e_2JkBT: # 確率p=1-np.exp(-2*self.J/kBT)で結合(union)
                                UF.merge(y*self.Nx+x,(ny)*self.Nx+(nx))

            cluster = np.full((self.Ny, self.Nx), -1) # クラスター番号を-1で初期化
            num_cluster = 0 # 各clusterに0から番号を振っていく
            dict_parent = dict()
            
            for i in range(self.Ntot): # まず，dict_parentをつくる
                if UF.parent[i] < 0: # 親の場合
                    dict_parent[i] = num_cluster
                    num_cluster += 1
                
            for i in range(self.Ntot): # 各スピンが属するclusterの番号を記録する
                y,x = i//self.Nx, i%self.Nx
                cluster[y][x] = dict_parent[UF.root(i)]

            for c in range(num_cluster):
                if random() < 0.5: # 確率1/2でスピンを反転
                    s_trial = np.where(cluster==c,s_trial*-1,s_trial) # i番目のclusterに属する(cluster==i)スピンを確率1/2で反転していく
                
            return s_trial, self.Hamiltonian(s_trial)


        # 初期化(そんなことはしないとは思うけど,続けて異なるシミュレートをする際に前回の結果が残っていたら困るので)
        self.E_history = [] # 各kBTの下で1スピンあたりエネルギー値を格納するためのリストを定義
        self.M_history = [] # 各kBTの下で1スピンあたり磁化を格納するためのリストを定義
        self.C_history = [] # 各kBTの下で1スピンあたり比熱を格納するためのリストを定義
        self.X_history = [] # 各kBTの下で1スピンあたり磁化率を格納するためのリストを定義
        self.S_history = [] # 各kBTの下で全スピン状態を格納するためのリストを定義

        start_time = datetime.datetime.now()
        print('')
        print('開始時刻',start_time.strftime('%Y-%m-%d %H:%M:%S'))
        print('')
        print('-----Monte Carlo Simulation('+self.MCmode+') Start-----')
        print('Progress Bar:')
        for k in range(len(self.kBT_list)):
            kBT = self.kBT_list[k]

            if k == 1:
                dt = datetime.datetime.now() - start_time
                print('終了予定時刻',start_time+dt*len(self.kBT_list))
            pro_bar = ('='*int(20*k/len(self.kBT_list))) + (' '*int(20*(1-k/len(self.kBT_list))))
            print('\r[{0}] {1}/{2} '.format(pro_bar, k,len(self.kBT_list)), end='')
            time.sleep(0.3)
            if k == 0:
                # スピン状態の初期化
                if self.cold:
                    s = coldstart() # kBT~0のときの平衡状態でスピンは揃っているはずなので, それに近い(というか同じ)状態からスタート
                else:
                    s = hotstart() # スピンがほぼ完全にランダムになっている高い温度kBTから低い温度に向けて冷やして行く
                
            else: # 収束を早めるために前回の続きからスタート
                s = self.S_history[-1].copy()

            E = self.Hamiltonian(s) # (初期)エネルギーを計算。
            M = np.sum(s) # 磁化の計算

            E_kBT = [] # kBTにおけるエネルギーの平均値のリスト     
            M_kBT = [] # kBTにおける磁化のリスト
            
            # 初期値を追加
            E_kBT.append(E/self.Ntot)
            M_kBT.append(M/self.Ntot) 
            
            # self.H == 0のときのみ使える(?)高速化(Wolff,SwendsenWang用)
            # expの計算は重いので前計算しておく
            e_2JkBT = np.exp(-2*self.J/kBT)

            # メイン
            for _ in range(self.N_MCS):
                if self.MCmode == 'Wolff':
                    # Wolffアルゴリズムによる状態更新
                    s, E = Wolff_algorithm(s, e_2JkBT, E_kBT)
                elif self.MCmode == 'SwendsenWang':
                    # SwendsenWangアルゴリズムによる状態更新
                    s, E = SwendsenWang_algorithm(s, e_2JkBT, E_kBT)
                else:
                    r = randrange(self.Ntot-1) # 0からNy*Nx-1まででランダムな数が選ばれる
                    # つまり,スピンの位置(i,j)が一様な確率で選ばれる
                    i = r // self.Nx # 第i行
                    j = r % self.Nx # 第j列

                    s_trial = s.copy()
                    delta_E = self.delta_Hamiltonian(s,i,j)
                    s_trial[i,j] = -1*s[i,j]
                    E_trial = E + delta_E

                    if self.MCmode == 'GibbsSampler':
                        # 熱浴法による状態更新
                        if random() < 1/(1+np.exp(delta_E/kBT)): # 遷移確率W=1/(1+exp(βΔE))
                            s = s_trial
                            E = E_trial
                        else:
                            pass
                    elif self.MCmode == 'Metropolis':
                        # メトロポリス法による状態更新
                        if E_trial < E : # エネルギーが下がっていたら採択(np.exp(-delta_E/kBT)の計算が重いのでif文をわざわざ増やした)
                            s = s_trial
                            E = E_trial
                        else : # エネルギーが上がっていたら，
                            if random() < np.exp(-delta_E/kBT): # 確率Boltzmann因子exp(-βΔE)で採択
                                s = s_trial
                                E = E_trial
                            else:
                                pass
                    else:
                        print('self.MCmodeを正しく入力してください')
                        return 

                E_kBT.append(E/self.Ntot)
                M = np.sum(s)
                M_kBT.append(M/self.Ntot)
            
            self.S_history.append(s)
            avsteps = int(self.N_MCS/self.equilibrium_num) # 熱平衡状態に達している(と思うことにする)のは,stepsのうち後ろから1/equilibriumのところ
            # 各kBTにおいて熱平衡状態に達していると考えられる,最後からavsteps個のデータのうちself.Psごとに状態を持ってきて平均する
            used_E = np.array(E_kBT[-avsteps::self.Ps]) # E_kBTの末尾からavsteps個のデータからself.Psごとに取得
            used_M = np.array(M_kBT[-avsteps::self.Ps]) # M_kBTの末尾からavsteps個のデータからself.Psごとに取得

            used_M = np.abs(used_M)

            # これが熱平衡状態における各物理量の期待値(すべて1スピンあたりの物理量)
            self.E_history.append(np.average(used_E))
            # 磁化は絶対値をとる
            self.M_history.append(np.average(np.abs(used_M)))
            self.C_history.append(np.var(used_E)*self.Ntot/(kBT**2))  # 比熱の計算
            # 厳密にはC_1 == N*(E^2-<E>^2)/kBT^2で, 分母は(kBT)^2ではない．この"C"は厳密にはC/kB
            self.X_history.append(np.var(used_M)*self.Ntot/kBT) # 磁化率の計算
            # X_1 == N*(M^2-<M>^2)/kBT
            # これらのゆらぎと応答の関係は，Kirkwoodの関係式という
        
        if not self.cold: # hotstartの場合，描画のために全てのデータを昇順に直しておく
            self.kBT_list = self.kBT_list[::-1]
            self.E_history = self.E_history[::-1]
            self.M_history = self.M_history[::-1]
            self.C_history = self.C_history[::-1]
            self.X_history = self.X_history[::-1]
            self.S_history = self.S_history[::-1]

        print('')
        print('-----Monte Carlo Simulation('+self.MCmode+') Finish-----')

        if is_save: # データの保存
            prefix = 'Ising,'+str(self.MCmode)+','+str(self.Nx)+','+str(self.Ny)+','+str(self.J)+','+str(self.H)+','+str(self.max_kBT)+','+str(self.steps_kBT)+','+str(self.N_MCS)+','+str(self.equilibrium_num)+','+str(self.Q)+','+self.start
            prefix = self.dir_path+self.MCmode+'/'+prefix
            np.savez(prefix,E=self.E_history,M=self.M_history,C=self.C_history,S=self.S_history,X=self.X_history)






    ### self.MonteCarlo()を実行していることが必要
    def visualize_temperature_development(self):
        if len(self.S_history) == 0:
            print("self.MonteCarlo()を実行してください")
            return 0

        def make_title(titlename): # あまりにも横長なので便宜上関数化した
            return r"Ising Model({}):{}".format(self.MCmode,titlename)+'\n'+r"$N_x$={}, $N_y$={}, $J$={}, $H$={},".format(self.Nx,self.Ny,self.J,self.H)+'\n'+r"$N_{{MCS}}$={}, $P_s$={}, {}".format(self.N_MCS,self.Ps,self.start)
        def make_filename(filename):
            return self.dir_path+self.MCmode+'/'+self.MCmode[0]+filename

        # 厳密解の計算
        theoretical_kBT_list = np.linspace(0.0001,self.max_kBT,1000)
        K = self.J/theoretical_kBT_list
        k = 2*np.sinh(2*K)/(np.cosh(2*K)**2)
        k2 = 2*np.tanh(2*K)**2-1
        # scipy.specialのimportに関してエラーが出たが，実行は問題なくできる
        E_theoretical = -self.J*(1+2/np.pi*k2*sps.ellipk(k**2))/np.tanh(2*K)
        C_theoretical = 2*K**2/np.pi/(np.tanh(2*K)**2)*(2*sps.ellipk(k**2)-2*sps.ellipe(k**2)-(1-k2)*(np.pi/2+k2*sps.ellipk(k**2)))
        M_theoretical = np.array([(1-(1/np.sinh(2*self.J/theoretical_kBT_list[i]))**4)**(1/8) if theoretical_kBT_list[i]<self.kBTc else 0 for i in range(len(theoretical_kBT_list))])   
        
        figsize = (8,7)
        Xlabel = r"$k_BT/k_BT_c$"

        title = r"エネルギー $E$"
        Ylabel = r"1スピンあたりの"+title
        make_figure(theoretical_kBT_list/self.kBTc,E_theoretical,plot_type='plot',lcolor='black',lwidth=0.5,label='厳密解',consecutive=1,fig_size=figsize)
        make_figure(self.kBT_list/self.kBTc,self.E_history,Xlabel=Xlabel,Ylabel=Ylabel,label=r"実験データ",titlename=make_title(title),filename=make_filename('1'),plot_type='scatter',Yerr=np.sqrt(self.C_history)*self.kBT_list,consecutive=-1)
        
        title = r"磁化の絶対値 $|M|$"
        Ylabel = r"1スピンあたりの"+title
        make_figure(theoretical_kBT_list/self.kBTc,M_theoretical,plot_type='plot',lcolor='black',lwidth=0.5,label='厳密解',consecutive=1,fig_size=figsize)
        make_figure(self.kBT_list/self.kBTc,self.M_history,Xlabel=Xlabel,Ylabel=Ylabel,label=r"実験データ",titlename=make_title(title),filename=make_filename('2'),plot_type='scatter',Yerr=np.sqrt(self.X_history*self.kBT_list),consecutive=-1)
        
        title = r"磁化の2乗 $M^2$"
        Ylabel = r"1スピンあたりの"+title
        make_figure(theoretical_kBT_list/self.kBTc,M_theoretical**2,plot_type='plot',lcolor='black',lwidth=0.5,label='厳密解',consecutive=1,fig_size=figsize)
        make_figure(self.kBT_list/self.kBTc,np.array(self.M_history)**2,Xlabel=Xlabel,Ylabel=Ylabel,label=r"実験データ",titlename=make_title(title),filename=make_filename('3'),consecutive=-1)
        
        
        title = r"磁化率 $\chi$"
        Ylabel = r"1スピンあたりの"+title
        make_figure(self.kBT_list/self.kBTc,np.array(self.X_history),Xlabel=Xlabel,Ylabel=Ylabel,label=r"実験データ",titlename=make_title(title),filename=make_filename('4'),fig_size=figsize)
        
        title = r"比熱 $C/k_B$"
        Ylabel = r"1スピンあたりの"+title
        make_figure(theoretical_kBT_list/self.kBTc,C_theoretical,plot_type='plot',lcolor='black',lwidth=0.5,label='厳密解',consecutive=1,fig_size=figsize)
        make_figure(self.kBT_list/self.kBTc,self.C_history,Xlabel=Xlabel,Ylabel=Ylabel,label=r"実験データ",titlename=make_title(title),filename=make_filename('5'),consecutive=-1)
        


    # npzデータ化した前回のデータから,各クラス変数を復元する
    def reconstructor(self,filename):
        self.MCmode,self.Nx,self.Ny,self.J,self.H,self.max_kBT,self.steps_kBT,self.N_MCS,self.equilibrium_num,self.Q,self.start = filename.split(',')[1:]
        file_path = self.dir_path+self.MCmode+'/'+filename
        npz_data = np.load(file_path)
        self.Nx, self.Ny, self.steps_kBT, self.N_MCS, self.Q = map(int,[self.Nx, self.Ny,self.steps_kBT,self.N_MCS,self.Q])
        self.J, self.H, self.max_kBT, self.steps_kBT, self.equilibrium_num = map(float,[self.J,self.H,self.max_kBT,self.steps_kBT,self.equilibrium_num])
        self.start = self.start[:-4]

        self.E_history = npz_data['E'] # kBT=0~max_kBTにおけるエネルギーの変遷
        self.M_history = npz_data['M'] # kBT=0~max_kBTにおける磁化の変遷
        self.C_history = npz_data['C'] # kBT=0~max_kBTにおける比熱の変遷
        self.X_history = npz_data['X'] # kBT=0~max_kBTにおける磁化率の変遷
        self.S_history = npz_data['S'] # kBT=0~max_kBTにおけるスピン全体の変遷




        

    ### self.MonteCarlo()を実行していることが必要
    def calculate_critical_index(self): # 臨界指数の計算
        # Xは定義域(0,∞)で，相転移点からの距離とする
        
        """
        # self.MonteCarlo()で計算した物理量の臨界指数を計算する
        # kBTc := 2*self.J/np.log(1+np.sqrt(2))
        # t := |T-Tc|/Tc
        # alpha:ゼロ磁場での比熱 C ~ t^(-alpha)    (T>Tc,T<Tcでalphaは不変)
        # beta:T<Tcでの磁化 M ~ t^beta           (T>Tcで自発磁化0)
        # gamma:T>Tcでの磁化率 X ~ t^(-gamma)

        # 2次元Ising模型の平均場近似解における臨界指数はそれぞれ，
        # alpha = 0 (不連続変化)
        # beta = 1/2
        # gamma = 1

        # 2次元Ising模型の厳密解における臨界指数はそれぞれ，
        # alpha = 0 (C ~ -ln(t))
        # beta = 1/8 = 0.125
        # gamma = 7/4 = 1.75

        # なお，ハイパー・スケーリング則によれば，alpha+2beta+gamma = 2が成り立つ
        """
        # 転移点をT=Tcと仮定した上で臨界指数を計算する

        def make_filename(filename):
            return self.dir_path+self.MCmode+'/'+self.MCmode[0]+filename
            #return '' # 保存せずに図だけ表示する
            
        figsize = (8,7)
        
        # Tc近傍の点のうちどこからどこまでを近似に使用するか(alphaとgammaについては有限系であることが効いてくる)
        # lowlim <= log(|T-Tc|/Tc) <= uplim を近似に使用する
        lowlim = -4
        uplim = -1
        
        # Tc近傍の点のうちいくつを近似に使用するか(betaのみ)
        num_effective_points = 10

        # αの対数近似の場合
        X_alpha = np.array([np.log(np.abs(self.kBT_list[i]-self.kBTc)/self.kBTc) for i in range(len(self.kBT_list)) if self.kBT_list[i]>self.kBTc])
        Y_alpha = np.array([self.C_history[i] for i in range(len(self.kBT_list)) if self.kBT_list[i]>self.kBTc])
        usedX,usedY = X_alpha[(lowlim<=X_alpha) & (X_alpha<=uplim)],Y_alpha[(lowlim<=X_alpha) & (X_alpha<=uplim)]
        a, b = np.polyfit(usedX,usedY,1)
        make_figure(X_alpha,a*X_alpha+b,plot_type='plot',lcolor='black',label=r"$T_c$近傍({}-{})点による近似直線".format(np.where(X_alpha<=uplim)[0][-1],np.where(X_alpha<lowlim)[0][-1])+'\n'+r"($C/k_B$={:.3f}$\log(|T-T_c|/T_c)+${:.3f})".format(a,b),consecutive=1,fig_size=figsize)
        # このプロットで直線上に点が並べばT~Tcで対数発散している(ぱっと見ではよくわからないが，近似の値は厳密解の値にかなり近い)
        make_figure(X_alpha,Y_alpha,filename=make_filename('8-1'),titlename=r"臨界指数$\alpha$({})".format(self.MCmode)+'\n'+r"(対数発散($\alpha=0$)なら$T_c$近傍の点は直線上に並ぶ)",Xlabel=r"$\log(|T-T_c|/T_c)$",Ylabel=r"比熱 $C/k_B$",label=r"実験データ",consecutive=-1)
        
        # 一応決定係数も見ておく
        R2_log = np.var(a*usedX+b-np.average(usedY)) / np.var(usedY-np.average(usedY))
        # AIC(赤池情報量)を用いてモデル比較
        AIC1 = len(usedX)*(1+np.log(2*np.pi*np.var(usedY-(a*usedX+b)))) + 6
        print('対数近似の決定係数：',R2_log)
        print('AIC1=',AIC1)
        

        # αの累乗近似の場合
        X_alpha = np.array([np.log(np.abs(self.kBT_list[i]-self.kBTc)/self.kBTc) for i in range(len(self.kBT_list)) if self.kBT_list[i]>self.kBTc and self.C_history[i]>0])
        Y_alpha = np.array([np.log(self.C_history[i]) for i in range(len(self.kBT_list)) if self.kBT_list[i]>self.kBTc and self.C_history[i]>0])
        usedX,usedY = X_alpha[(lowlim<=X_alpha) & (X_alpha<=uplim)],Y_alpha[(lowlim<=X_alpha) & (X_alpha<=uplim)]
        alpha, b = np.polyfit(usedX,usedY,1)
        make_figure(X_alpha,alpha*X_alpha+b,plot_type='plot',lcolor='black',label=r"$T_c$近傍({}-{})点による近似直線($\alpha={:.3f}$)".format(np.where(X_alpha<=uplim)[0][-1],np.where(X_alpha<lowlim)[0][-1],-alpha),consecutive=1,fig_size=figsize)
        make_figure(X_alpha,Y_alpha,filename=make_filename('8-2'),titlename=r"臨界指数$\alpha$({})".format(self.MCmode)+'\n'+r"(累乗近似)",Xlabel=r"$\log(|T-T_c|/T_c)$",Ylabel=r"比熱の対数 $\log(C/k_B)$",label=r"実験データ",consecutive=-1)
        
        # 一応決定係数も見ておく
        R2_exp = np.var(alpha*usedX+b-np.average(usedY)) / np.var(usedY-np.average(usedY))
        # AIC(赤池情報量)を用いてモデル比較
        AIC2 = len(usedX)*(1+np.log(2*np.pi*np.var(np.exp(usedY)*(usedY-(alpha*usedX+b))))) - 2*np.sum(usedY) + 6
        print('冪乗近似の決定係数：',R2_exp)
        print('AIC2=',AIC2)
        


        # βの累乗近似
        X_beta = np.array([np.log(np.abs(self.kBT_list[i]-self.kBTc)/self.kBTc) for i in range(len(self.kBT_list)) if self.kBT_list[i]<self.kBTc and self.M_history[i]>0])
        Y_beta = np.array([np.log(self.M_history[i]) for i in range(len(self.kBT_list)) if self.kBT_list[i]<self.kBTc and self.M_history[i]>0])
        beta, b = np.polyfit(X_beta[-num_effective_points:],Y_beta[-num_effective_points:],1)
        make_figure(X_beta,beta*X_beta+b,plot_type='plot',lcolor='black',label=r"$T_c$近傍{}点による近似直線($\beta={:.3f}$)".format(num_effective_points,beta),consecutive=1,fig_size=figsize)
        make_figure(X_beta,Y_beta,filename=make_filename('9'),titlename=r"臨界指数$\beta$({}):".format(self.MCmode)+"\n"+r"$T<T_c$で$M\sim(|T-T_c|/T_c)^\beta$",Xlabel=r"$\log(|T-T_c|/T_c)$",Ylabel=r"磁化の対数 $\log(M)$",label=r"実験データ",consecutive=-1)

        # γの累乗近似
        X_gamma = np.array([np.log(np.abs(self.kBT_list[i]-self.kBTc)/self.kBTc) for i in range(len(self.kBT_list)) if self.kBT_list[i]>self.kBTc and self.X_history[i]>0])
        Y_gamma = np.array([np.log(self.X_history[i]) for i in range(len(self.kBT_list)) if self.kBT_list[i]>self.kBTc and self.X_history[i]>0])
        gamma, b = np.polyfit(X_gamma[(lowlim<=X_gamma) & (X_gamma<=uplim)], Y_gamma[(lowlim<=X_gamma) & (X_gamma<=uplim)],1)
        make_figure(X_gamma,gamma*X_gamma+b,plot_type='plot',lcolor='black',label=r"$T_c$近傍({}-{})点による近似直線($\gamma={:.3f}$)".format(np.where(X_gamma<=uplim)[0][-1],np.where(X_gamma<lowlim)[0][-1],-gamma),consecutive=1,fig_size=figsize)
        make_figure(X_gamma,Y_gamma,filename=make_filename('10'),titlename=r"臨界指数$\gamma$({}):".format(self.MCmode)+"\n"+r"$T>T_c$で$\chi\sim(|T-T_c|/T_c)^{{-\gamma}}$",Xlabel=r"$\log(|T-T_c|/T_c)$",Ylabel=r"磁化率の対数 $\log(\chi)$",label=r"実験データ",consecutive=-1)

        print('臨界指数')
        #print('alpha:'+str(-alpha))
        print('beta :'+str(beta))
        print('gamma:'+str(-gamma))


        

def main():
    # 理論計算による相転移温度Tcは，H=0で，kBTc=2J/ln(1+√2)=2.2691853J
    
    
    
    # single-spin-flip
    Nx = 50  # x方向にNx個のスピン
    Ny = 50  # y方向にNy個のスピン
    J = 1     # 交換相互作用定数
    H = 0.0   # 外部磁場
    max_kBT = 7 # 最大温度
    steps_kBT = 40 # 温度刻み数
    cold = True # 低温側からスタート

    equilibrium_num = 1.1 # 平衡状態開始割合の逆数
    sampling_period = 7500 # サンプリング周期
    Q = 1000 # サンプル状態数
    N_MCS = int(equilibrium_num*sampling_period*Q) # MCstep回数
    
    """
    # cluster-spin-flip
    Nx = 50  # x方向にNx個のスピン
    Ny = 50  # y方向にNy個のスピン
    J = 1     # 交換相互作用定数
    H = 0.0   # 外部磁場
    max_kBT = 7 # 最大温度
    steps_kBT = 40 # 温度刻み数
    cold = True # 低温側からスタート

    equilibrium_num = 1.1 # 平衡状態開始割合の逆数
    sampling_period = 20 # サンプリング周期
    Q = 1000 # サンプル状態数
    N_MCS = int(equilibrium_num*sampling_period*Q) # MCstep回数
    """
    
    dir_name = 'Desktop' # 保存先/読込元のディレクトリ名
    is_save = False # 保存

    I = Ising2(Ny,Nx,J,H,dir_name=dir_name,max_kBT=max_kBT,steps_kBT=steps_kBT,N_MCS=N_MCS,sampling_period=sampling_period,equilibrium_num=equilibrium_num,Q=Q,cold=cold) # coldstartがデフォルト

    ### 不要な部分をコメントアウトして使う
    start_time = time.process_time()
    I.MonteCarlo(MCmode='GibbsSampler',is_save=is_save)
    #I.MonteCarlo(MCmode='Metropolis',is_save=is_save)
    #I.MonteCarlo(MCmode='Wolff',is_save=is_save)
    #I.MonteCarlo(MCmode='SwendsenWang',is_save=is_save)
    finish_time = time.process_time()
    print("Time: "+"%g" % (finish_time - start_time)) # 実行時間を計測

    #I.reconstructor('Ising,GibbsSampler,50,50,1,0.0,7,40,8250000,1.1,1000,coldstart.npz')
    #I.reconstructor('Ising,Metropolis,50,50,1,0.0,7,40,8250000,1.1,1000,coldstart.npz')
    #I.reconstructor('Ising,SwendsenWang,50,50,1,0.0,7,40,22000,1.1,1000,coldstart.npz')
    #I.reconstructor('Ising,Wolff,50,50,1,0.0,7,40,22000,1.1,1000,coldstart.npz')
    
    I.visualize_temperature_development() # kBTを変化させた時の各物理量の変化
    I.calculate_critical_index() # 臨界指数の算出
    

if __name__ == "__main__":
    main()
