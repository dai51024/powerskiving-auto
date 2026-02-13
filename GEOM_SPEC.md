**GEOM\_SPEC.md（ 幾何核 完全凍結）**

GEOM\_SPECバージョン: 1-1-1

対応SPECバージョン: 1-1-1

⸻

0\. この文書の役割（凍結対象）

この GEOM\_SPEC.md は、以下を 数式と手順で完全凍結する：

	•	共役点群生成（tool\_conjugate\_grid の生成）

	•	work\_target\_surface（ワーク目標歯面の具体式）

	•	ゴールデン（Σ=0での s\_rot 自動決定の具体式）

	•	envelope\_light（ターゲット点群生成と包絡評価の具体式）

	•	relief 評価（axis\_taper の min\_effective\_relief\_angle の具体式）

禁止：

	•	実装側で「別の式」「別の解釈」「別のパラメータ化」「暗黙の既定値」を入れること。

	•	本文に無い式を“それっぽく”補うこと。

重要：

	•	spec\_version と geom\_spec\_version は config / report / manifest に必ず刻印し、一致しなければ ConfigError（即停止）。

⸻


0.1 入力/出力の決定性（SPEC側契約）

この GEOM\_SPEC.md は「幾何核（数式と手順）」を凍結するが、I/O の決定性は SPEC.md 側で凍結する。

  • config のファイル形式と config\_sha256 の定義は SPEC.md の 4.0.1 / 4.0.2 に従う（本幾何核も前提とする）。

  • report.json / cad\_report\_stepXX.json / manifest.json の JSON文字列化規約は SPEC.md の 8.4 に従う（幾何核の数値も同じ規約で刻印される）。

  • manifest.json の自己参照禁止・created\_utc 固定・files[] 列挙順は SPEC.md の 8.3.1 に従う。


1\. 前提スコープ（この幾何核が成立する条件）

	•	ワーク：内歯（internal gear）、平歯（helix=0）

	•	工具：ピニオン型（pinion）、外周刃、工具軸は Ry(Σ) 傾き（SPEC参照）

	•	送り（feed）：0固定（軸方向送りやねじれ補正はこの版では扱わない）

⸻

2\. 共通定義（定数・関数：完全固定）

2.1 定数（固定）

	•	PI  \= 3.141592653589793

	•	TWO\_PI \= 6.283185307179586

2.2 回転（固定）

	•	Rz(θ) は \+Z 回り右手回転（SPECと同じ）

	•	Ry(Σ) は \+Y 回り右手回転（SPECと同じ）

2.3 wrap（固定：値域 \[-π,+π)）

	•	wrap\_rad(x) \= x \- TWO\_PI \* floor((x \+ PI)/TWO\_PI)

	•	よって wrap\_rad(PI) \= \-PI

2.4 q（固定：量子化、built-in round禁止）

	•	sign(0) \= \+1 とする（-0禁止のため）

	•	q(x, d) \= sign(x) \* floor(|x|\*10^d \+ 0.5) / 10^d

2.5 percentile（固定：higher\_order\_stat、補間禁止）

N個の値 x を昇順ソートして x\_sorted とし、p∈(0,1\]：

	•	k \= ceil(p\*N) \- 1

	•	percentile(p) \= x\_sorted\[ clamp(k, 0, N-1) \]  
2.6 基本関数（固定：実装差を殺す）

(1) 角度変換（固定）

  • deg2rad(x\_deg) \= x\_deg \* PI / 180

  • rad2deg(x\_rad) \= x\_rad \* 180 / PI

(2) hypot（固定）

  • hypot2(x,y) \= sqrt(x\*x \+ y\*y)

(3) dot / cross（右手系、固定）

  • dot(a,b) \= a.x\*b.x \+ a.y\*b.y \+ a.z\*b.z

  • cross(a,b) \= (a.y\*b.z \- a.z\*b.y,

                  a.z\*b.x \- a.x\*b.z,

                  a.x\*b.y \- a.y\*b.x)

(4) norm / normalize（固定）

  • eps\_norm \= 1e-12

  • norm(v) \= sqrt(dot(v,v))

  • normalize(v):

      if norm(v) \< eps\_norm \-\> INVALID（その場の呼び出し側で reason\_code を立てる）

      else \-\> v / norm(v)

(5) clamped zero（固定：fのゼロ扱い）

  • f\_clamp(f) \= 0 if |f| \<= f\_zero\_eps else f

⸻

3\. config に要求する幾何キー（GEOM\_SPEC側の契約）

3.1 必須キー（ここに無いと幾何核が動かない）

以下は config 必須（欠落は ConfigError）：

(A) ワーク基本

	•	module\_mm

	•	z2

	•	pressure\_angle\_deg

	•	face\_width\_mm

	•	helix\_beta\_deg（この版は 0固定。0以外は ConfigError）

	•	work\_target\_model（この版は “ideal\_involute” 固定。違えば ConfigError）

(B) 工具・機構

	•	z1

	•	center\_distance\_a\_mm

	•	sigma\_rad

	•	theta\_tooth\_center\_rad（Tool座標の歯中心角：SPEC側の定義に従う）

	•	s\_rot\_selected は 幾何核が決める（入力ではない）。reportへ保存する。

　　• dtheta\_deadband\_rad（必須：rootのside整合チェックで使用。SPEC 4.9と同一キー）

(C) 共役格子（tool\_conjugate\_grid の“サンプリング範囲”）

この版では、共役格子の u（=半径）範囲を 明示入力にする（暗黙禁止）：

	•	grid\_u\_min\_mm

	•	grid\_u\_max\_mm

	•	Nu（int\>=2）

	•	Nv（int\>=2）

制約（満たさない場合は ConfigError ではなく、その範囲の点が OUTSIDE\_DOMAIN になり得るが、通常は設計ミス）：

	•	grid\_u\_min\_mm \< grid\_u\_max\_mm

(D) ゴールデン（Σ=0で s\_rot 選定）

	•	golden\_tol\_p95\_mm

	•	golden\_tol\_max\_mm

	•	golden\_dz\_mm

	•	golden\_dz\_max\_mm

	•	golden\_min\_points

	•	golden\_pitch\_band\_dr\_mm

	•	golden\_ref\_n（int\>=20）

(E) envelope\_light（包絡評価）

	•	theta1\_range\_deg

	•	theta1\_step\_deg

	•	multi\_rev\_k\_min（int）

	•	multi\_rev\_k\_max（int, k\_max\>=k\_min）

	•	edge\_stride（int\>=1）

	•	theta2\_offset\_deg（位相オフセット）

	•	n\_target\_u（int\>=1）

	•	n\_target\_v（int\>=1）

	•	target\_u\_min\_mm

	•	target\_u\_max\_mm

	•	target\_r\_filter\_mode（“off” | “pitch\_band”）

	•	pitch\_band\_dr\_mm（pitch\_band時のみ使用）

(F) relief（axis\_taper評価）

	•	relief\_mode（この版の評価対象は “axis\_taper” のみ。それ以外は skipped\_due\_to\_missing\_od ではなく failed\_validation 推奨）

	•	relief\_angle\_deg

	•	land\_width\_mm（この版の relief “角度評価”では寸法としてログに残すだけ。形状生成はしない）

⸻

4\. 派生幾何（ワーク/工具の基本円：固定式）

以下は計算式を固定し、report に保存してよい（推奨）。

4.1 ワーク（internal gear）の派生量

	•	m \= module\_mm

	•	alpha \= deg2rad(pressure\_angle\_deg)

	•	r\_pitch\_work \= 0.5 \* m \* z2

	•	r\_base\_work  \= r\_pitch\_work \* cos(alpha)

注意：この版の work\_target\_surface は「理想インボリュート」を扱う。

インボリュートの定義域は r \>= r\_base\_work。

u \< r\_base\_work は OUTSIDE\_DOMAIN。

4.2 工具（pinion）の派生量（主にゴールデン用）

	•	r\_pitch\_tool \= 0.5 \* m \* z1

	•	r\_base\_tool  \= r\_pitch\_tool \* cos(alpha)

⸻

5\. work\_target\_surface（“ideal\_involute” / 完全凍結）

5.1 入出力（固定）

入力：

	•	side ∈ {“plus”,“minus”}（W0座標における歯の左右）

	•	u\_mm：半径 r（mm）。この幾何核では u は 半径を意味する（他の意味は禁止）

	•	v\_mm：z座標（mm）。範囲は \[-face\_width\_mm/2, \+face\_width\_mm/2\] を推奨だが、範囲外でも式は定義される

出力（W0座標）：

	•	p\_W0 \= (x,y,z)（mm）

	•	n\_W0 \= (nx,ny,nz)（unit）

5.2 インボリュートの基礎（固定）

u\_mm からインボリュートパラメータ t を求める（固定）：

	•	r \= u\_mm

	•	r\_b \= r\_base\_work

	•	if r \< r\_b: OUTSIDE\_DOMAIN

	•	t \= sqrt( max(0, (r/r\_b)^2 \- 1\) )

基礎インボリュート（2D、base orientation、固定）：

	•	x0(t) \= r\_b \* (cos t \+ t\*sin t)

	•	y0(t) \= r\_b \* (sin t \- t\*cos t)

接線（2D、固定）：

	•	tx0(t) \= cos t

	•	ty0(t) \= sin t

（※上は方向ベクトル。スケールは不要）

法線候補（2D、固定）：

	•	nx0(t) \= \-sin t

	•	ny0(t) \=  cos t

（これは tx0,ty0 を \+90°回転したもの）

5.3 歯厚中心合わせの回転オフセット（固定：標準歯厚仮定）

圧力角：

	•	alpha \= deg2rad(pressure\_angle\_deg)

ピッチ円での t：

	•	t\_p \= tan(alpha)（固定）

ピッチ点角度（固定）：

	•	phi\_p \= wrap\_rad(atan2( y0(t\_p), x0(t\_p) ))

半歯厚角（固定：標準歯厚、プロファイルシフト無し）：

	•	theta\_half \= PI / (2\*z2)

回転オフセット（固定）：

	•	delta \= wrap\_rad(theta\_half \- phi\_p)

5.4 plus/minus の生成（固定）

plus（2D）：

	•	p2 \= Rz(+delta) \* \[x0(t), y0(t)\]

	•	t2 \= Rz(+delta) \* \[tx0(t), ty0(t)\]

	•	n2 \= Rz(+delta) \* \[nx0(t), ny0(t)\]

minus（2D、鏡像+回転、固定）：

	•	まず鏡像：\[x0, y0\] \-\> \[x0, \-y0\]

	•	同様にベクトルも vy \-\> \-vy

	•	p2 \= Rz(-delta) \* \[x0(t), \-y0(t)\]

	•	t2 \= Rz(-delta) \* \[tx0(t), \-ty0(t)\]

	•	n2 \= Rz(-delta) \* \[nx0(t), \-ny0(t)\]

5.5 法線の向き（internal gearとして固定）

この版は gear\_type=internal を前提とし、ワーク法線は「歯溝側（工具が入る側）」へ向ける。

判定（固定）：

	•	r\_hat \= normalize(\[p2.x, p2.y\])

	•	if dot(n2, r\_hat) \> 0: n2 \= \-n2

（＝半径外向き成分を持つなら反転。内向きへ揃える）

出力（3D、固定）：

	•	p\_W0 \= (p2.x, p2.y, v\_mm)

	•	n\_W0 \= normalize( (n2.x, n2.y, 0\) )

⸻

6\. 共役点群生成（tool\_conjugate\_grid / 完全凍結）

6.1 何を解くか（凍結）

各ワーク点 p\_W0(u,v) に対し、工具回転角 θ1 を求め、接触点を Tool座標へ写した点列を 工具フランク点として採用する。

接触条件（固定：基本かみあい方程式 / meshing equation）：

	•	f(θ1) \= dot( v\_rel(θ1), n\_W(θ1) ) \= 0

ここで、feed=0、回転比固定：

	•	θ2(θ1) \= s\_rot \* (z1/z2) \* θ1 \+ θ2\_0

	•	共役点群生成では θ2\_0 \= 0 固定（この版で凍結）

	•	s\_rot は Step1（ゴールデン）で決まる

ワーク回転（固定）：

	•	p\_W(θ1) \= Rz(θ2) \* p\_W0

	•	n\_W(θ1) \= Rz(θ2) \* n\_W0

工具中心（固定）：

	•	c1\_W \= (0, a, 0)（a=center\_distance\_a\_mm）

工具軸（World表現、固定）：

	•	k1 \= (sinΣ, 0, cosΣ)（Σ=sigma\_rad）

	•	ワーク軸（固定）：

	•	k2 \= (0, 0, 1\)

相対速度（ω1で割った形、固定）：

	•	v\_rel \= cross(k1, (p\_W \- c1\_W)) \- (s\_rot\*(z1/z2)) \* cross(k2, p\_W)

よって：

	•	f(θ1) \= dot(v\_rel, n\_W)

6.2 グリッドサンプリング（iu,iv → work(u,v) / 固定）

共役格子は work\_target\_surface のパラメータでサンプルする。

	•	u\_min \= grid\_u\_min\_mm

	•	u\_max \= grid\_u\_max\_mm

	•	z\_min \= \-0.5 \* face\_width\_mm

	•	z\_max \= \+0.5 \* face\_width\_mm

サンプル（固定：端点含む線形）：

	•	u(iu) \= u\_min \+ iu\*(u\_max-u\_min)/(Nu-1)  for iu=0..Nu-1

	•	v(iv) \= z\_min \+ iv\*(z\_max-z\_min)/(Nv-1)  for iv=0..Nv-1

注意：Nu=2やNv=2も許可。ゼロ除算回避は実装側で条件分岐（固定の対応）：

	•	Nu==1 は禁止（ConfigError）

	•	Nv==1 は禁止（ConfigError）

6.3 1点の生成（固定）

入力：

	•	side（“plus”/“minus”：work側）

	•	iu, iv

手順（固定）：

	1\.	u\_mm \= u(iu), v\_mm \= v(iv)

	2\.	work\_target\_surface(side, u\_mm, v\_mm) を計算

	•	OUTSIDE\_DOMAIN なら grid点 valid=0 reason\_code=OUTSIDE\_DOMAIN

	3\.	θ1 を f(θ1)=0 で解く（6.4で凍結）

	4\.	得られた θ1\* と θ2\* で p\_W, n\_W を計算

	5\.	Tool座標へ写像（SPECの変換と一致する形で固定）：

	•	R\_TW \= Rz(-θ1\*) \* Ry(-Σ)

	•	p\_T \= R\_TW \* (p\_W \- c1\_W)

	•	n\_T \= R\_TW \* n\_W

            •           n\_T は normalize を使う（2.6の eps\_norm に従う）

            •           norm(n\_T) \< eps\_norm の場合：valid=0, reason\_code=INVALID\_NORMAL

	6\.	出力：

	•	gridに p\_T, n\_T, valid=1, reason\_code=OK

	•	併せて内部ログ用に θ1\*, θ2\*, residual=|f(θ1\*)| を保持してよい（report統計に使う）

6.4 ルートソルバ（θ1の解き方 / 完全凍結）

この版の solver は「ブラケット探索 \+ 二分法」で固定（Newton禁止）。

ただし従来方式の弱点（符号変化が見えないと取り逃がす）を仕様で潰す。

固定パラメータ（configではなく仕様固定）：

  • theta1\_seed\_init \= 0.0

  • theta1\_step\_scan\_rad \= deg2rad(1.0)   （粗走査刻み）

  • theta1\_scan\_max\_steps \= 360           （±360°＝±2π）

  • theta1\_bisect\_max\_iter \= 80

  • theta1\_tol\_rad \= 1e-12

  • f\_zero\_eps \= 1e-12                    （fのゼロ扱い）

  • theta1\_refine\_div \= 10                （失敗時の局所リファイン分割）

  • theta1\_refine\_max\_steps \= 10          （±(1.0°) までを 0.1°刻みで再走査）

探索順（固定）：

  • gridは sideごとに iv昇順→iu昇順（v-major）

  • 各 side は theta1\_seed を theta1\_seed\_init で開始

  • 1点で root を採用したら、その side の次点の seed は wrap\_rad(root) に更新（値域\[-π,+π)で拘束）

f の取り扱い（固定）：

  • f\_clamp(f) \= 0 if |f| \<= f\_zero\_eps else f

  • “==0” 判定は禁止。常に |f|\<=f\_zero\_eps でゼロ扱いする。

1点（u,v）に対する solver（固定）：

(0) best tracking（固定：失敗時の局所リファイン中心）

  • best\_abs \= \+INF

  • best\_x \= theta1\_seed

(1) 候補root評価関数（固定）

  • try\_accept(x):

      \- θ2 \= s\_rot\*(z1/z2)\*x \+ θ2\_0（共役生成では θ2\_0=0 固定）

      \- p\_W \= Rz(θ2)\*p\_W0, n\_W \= Rz(θ2)\*n\_W0

      \- R\_TW \= Rz(-x) \* Ry(-Σ)

      \- p\_T \= R\_TW \* (p\_W \- c1\_W)

      \- theta\_tool \= wrap\_rad(atan2(p\_T.y, p\_T.x))

      \- dtheta \= wrap\_rad(theta\_tool \- theta\_tooth\_center\_rad)

      \- if |dtheta| \<= dtheta\_deadband\_rad \-\> reject（deadband）

      \- side整合（固定）：

          side=="plus"  \-\> dtheta \> 0 が合格

          side=="minus" \-\> dtheta \< 0 が合格

      \- 合格なら accept（root=x）

      \- 不合格なら reject

(2) 粗走査（固定：±を交互に伸ばす。各iで \+ → \- の順）

  • x0 \= theta1\_seed

  • f0 \= f(x0)

  • best\_abs/best\_x 更新（固定：初回は必ずx0）

  • if |f0| \<= f\_zero\_eps:

      if try\_accept(x0) \== accept: return root

  • prev\_plus \= (x0,f0)

  • prev\_minus \= (x0,f0)

  • for i=1..theta1\_scan\_max\_steps:

      \--- plus側（先）---

      x1 \= x0 \+ i\*theta1\_step\_scan\_rad

      f1 \= f(x1)

      if |f1| \< best\_abs: best\_abs=|f1|, best\_x=x1   （同値tieは更新しない＝先勝ち固定）

      if |f1| \<= f\_zero\_eps:

          if try\_accept(x1) \== accept: return root

      else:

          a=prev\_plus.x, fa=prev\_plus.f

          b=x1, fb=f1

          fca=f\_clamp(fa), fcb=f\_clamp(fb)

          if fca==0:

              if try\_accept(a) \== accept: return root

          elif fcb==0:

              if try\_accept(b) \== accept: return root

          elif fca\*fcb \< 0:

              root \= bisect(a,b)（下記(4)）

              if try\_accept(root) \== accept: return root

      prev\_plus=(x1,f1)

      \--- minus側（後）---

      x2 \= x0 \- i\*theta1\_step\_scan\_rad

      f2 \= f(x2)

      if |f2| \< best\_abs: best\_abs=|f2|, best\_x=x2

      if |f2| \<= f\_zero\_eps:

          if try\_accept(x2) \== accept: return root

      else:

          a=x2, fa=f2

          b=prev\_minus.x, fb=prev\_minus.f

          （a\<bに正規化してから同様に判定）

          if a\>b: swap(a,b), swap(fa,fb)

          fca=f\_clamp(fa), fcb=f\_clamp(fb)

          if fca==0:

              if try\_accept(a) \== accept: return root

          elif fcb==0:

              if try\_accept(b) \== accept: return root

          elif fca\*fcb \< 0:

              root \= bisect(a,b)

              if try\_accept(root) \== accept: return root

      prev\_minus=(x2,f2)

(3) 局所リファイン（固定：粗走査で失敗した場合のみ）

  • x0r \= best\_x

  • step\_r \= theta1\_step\_scan\_rad / theta1\_refine\_div   （=0.1°）

  • max\_r \= theta1\_refine\_max\_steps                      （=±1.0°）

  • f0r \= f(x0r)

  • if |f0r| \<= f\_zero\_eps:

      if try\_accept(x0r) \== accept: return root

  • prev\_plus=(x0r,f0r), prev\_minus=(x0r,f0r)

  • for i=1..max\_r:

      （(2)と同じ手順で x0r±i\*step\_r を評価し、

       近傍の |f|\<=eps と ブラケット→二分法 を試す。

       ループ内の評価順は \+ → \- 固定）

  • ここまで来て見つからないなら SOLVER\_FAIL

(4) 二分法 bisect(a,b)（固定）

  • 入力は a\<b を保証

  • fa=f(a), fb=f(b)

  • if |fa|\<=f\_zero\_eps: return a

  • if |fb|\<=f\_zero\_eps: return b

  • fca=f\_clamp(fa), fcb=f\_clamp(fb)

  • if fca\*fcb \> 0: ここに来た時点で仕様違反（bracketではない）→ fail

  • for iter=1..theta1\_bisect\_max\_iter:

      m \= 0.5\*(a+b)

      fm \= f(m)

      if |b-a| \<= theta1\_tol\_rad OR |fm| \<= f\_zero\_eps: return m

      fcm \= f\_clamp(fm)

      if fca\*fcm \<= 0:

          b=m, fb=fm, fcb=fcm

      else:

          a=m, fa=fm, fca=fcm

  • 反復上限なら return 0.5\*(a+b)（決定性のため）

⸻

7\. ゴールデン（Σ=0で s\_rot を決める / 完全凍結）

目的：s\_rot ∈ {+1,-1} のうち正しい符号を Σ=0 で選ぶ。

この版では「参照インボリュート折れ線との 2D距離」で選ぶ。

7.1 ゴールデンの入力条件（固定）

	•	Σ=0 固定

	•	θ2\_0=0 固定

	•	それ以外の config は通常通り（m,z1,z2,α,Nu,Nv,grid\_u範囲など）

7.2 参照インボリュート（工具側 / 固定）

工具ピッチ円：

	•	r\_p \= r\_pitch\_tool \= 0.5\*m\*z1

	•	r\_b \= r\_base\_tool \= r\_p\*cos(alpha)

alpha \= deg2rad(pressure\_angle\_deg)

t\_p \= tan(alpha)

phi\_p \= wrap\_rad(atan2(y0(t\_p), x0(t\_p)))（x0,y0は5章の式だが r\_b を tool のものにする）

半歯厚角（標準歯厚仮定）：

	•	theta\_half \= PI/(2\*z1)

回転オフセット：

	•	delta \= wrap\_rad(theta\_half \- phi\_p)

参照折れ線の t範囲（固定：ピッチ帯から決める）：

	•	r\_min \= r\_p \- golden\_pitch\_band\_dr\_mm

	•	r\_max \= r\_p \+ golden\_pitch\_band\_dr\_mm

	•	r\_min \= max(r\_min, r\_b)（r\_b未満は切り上げ）

	•	t(r) \= sqrt(max(0,(r/r\_b)^2 \- 1))

	•	t\_min=t(r\_min), t\_max=t(r\_max)

midpoint サンプル（固定）：

	•	N \= golden\_ref\_n

	•	t\_j \= t\_min \+ (j+0.5)\*(t\_max-t\_min)/N, j=0..N-1

参照折れ線（固定）：

	•	plus：Pj \= Rz(+delta)\*\[x0(t\_j), y0(t\_j)\]

	•	minus：Pj \= Rz(-delta)\*\[x0(t\_j), \-y0(t\_j)\]

7.3 ゴールデン評価点の抽出（固定）

候補 s\_rot ごとに、Σ=0で共役格子（raw）を生成する（6章の方法）。

z帯域抽出（固定）：

	•	valid点の z を集め z\_mid \= percentile(0.5)（higher\_order\_stat）

	•	dz \= golden\_dz\_mm から開始

	•	抽出集合 S(dz) \= { valid==1 かつ |z \- z\_mid| \<= dz }

	•	点数が golden\_min\_points 未満なら dz \*= 2 を繰り返し、dz \<= golden\_dz\_max\_mm まで拡張

	•	それでも未満なら candidate\_failed=true（この候補は失格扱い）

半径帯抽出（固定）：

	•	r\_p \= r\_pitch\_tool

	•	r ∈ \[r\_p \- golden\_pitch\_band\_dr\_mm, r\_p \+ golden\_pitch\_band\_dr\_mm\]

	•	r は Tool座標で hypot(x\_mm, y\_mm)

side分類（固定）：

	•	theta\_tool \= wrap\_rad(atan2(y,x))

	•	dtheta \= wrap\_rad(theta\_tool \- theta\_tooth\_center\_rad)

	•	dtheta\>0 → plus、dtheta\<0 → minus、|dtheta|\<=deadband →除外

7.4 距離と集計（固定）

距離（固定）：

	•	2D点 Q=(x,y) から参照折れ線（線分列）への点→線分最短距離の最小値（mm）

	•	zは使わない

集計（固定）：

	•	plus/minus別に距離配列を作る

	•	p95 は higher\_order\_stat

	•	max は最大値

	•	worst:

	•	golden\_p95 \= max(plus\_p95, minus\_p95)

	•	golden\_max \= max(plus\_max, minus\_max)

候補選択（固定）：

	•	primary：golden\_p95 が小さい方

	•	tie：golden\_max が小さい方

	•	それでも同値：s\_rot=+1 を採用（固定）

合否（固定）：

	•	golden\_p95 \<= golden\_tol\_p95\_mm かつ golden\_max \<= golden\_tol\_max\_mm

	•	不合格なら 即停止（golden\_failed）

⸻

8\. envelope\_light（完全凍結）

8.1 目的

生成した cutting\_edge をスイープした点群（envelope点群）を W0 に戻し、work\_target\_cloud と最近傍距離で誤差を評価する。

8.2 work\_target\_cloud の生成（W0座標 / 固定）

u,vサンプル（固定：midpoint）：

	•	u\_i \= target\_u\_min\_mm \+ (i+0.5)\*(target\_u\_max\_mm-target\_u\_min\_mm)/n\_target\_u

	•	v\_j \= (-0.5\*face\_width\_mm) \+ (j+0.5)\*(face\_width\_mm)/n\_target\_v

各点：

	•	p\_W0 \= work\_target\_surface(side, u\_i, v\_j)

pitch\_band フィルタ（固定）：

	•	r\_pitch\_work \= 0.5\*m\*z2

	•	mode が “pitch\_band” のときだけ適用

	•	r \= hypot(p\_W0.x, p\_W0.y)

	•	keep if r ∈ \[r\_pitch\_work \- pitch\_band\_dr\_mm, r\_pitch\_work \+ pitch\_band\_dr\_mm\]

失敗条件（固定）：

	•	フィルタ後点数が 0 の side があれば failed\_validation で停止

8.3 envelope点群生成（固定）

前提：

	•	edges\_ok==true のときのみ実行（SPEC側）

(1) edgeの間引き（固定）

	•	selected edge の点列（point\_id昇順）から

	•	keep if point\_id % edge\_stride \== 0

	•	最後の点（point\_id=n-1）は必ず含める（重複は除外）

(2) theta1 サンプリング（固定）

	•	theta1\_range\_rad \= deg2rad(theta1\_range\_deg)

	•	theta1\_step\_rad  \= deg2rad(theta1\_step\_deg)

	•	n\_theta1 \= floor(theta1\_range\_rad/theta1\_step\_rad) \+ 1

	•	theta1\_base\[i\] \= \-0.5\*theta1\_range\_rad \+ i\*theta1\_step\_rad

(3) multi\_rev（固定）

	•	k \= multi\_rev\_k\_min .. multi\_rev\_k\_max（両端含む、昇順）

(4) theta2（固定）

	•	theta2\_0 \= deg2rad(theta2\_offset\_deg)

	•	theta2 \= s\_rot\_selected\*(z1/z2)\*theta1 \+ theta2\_0

(5) 変換（固定）

Tool→World：

	•	p\_W \= c1\_W \+ (Ry(sigma\_rad)\*Rz(theta1)) \* p\_T

World→W0：

	•	p\_W0 \= Rz(-theta2) \* p\_W

点列の列挙順（固定）：

	•	side（plus→minus）

	•	edge点（昇順）

	•	theta1\_base（i昇順）

	•	k（昇順）

8.4 距離と集計（固定）

距離（固定）：

	•	target\_cloud\_side に対する KD-tree 最近傍距離（点→点、ユークリッド）

集計（固定）：

	•	plus/minus別 p95 と max

	•	worst：

	•	envelope\_p95\_mm \= max(plus\_p95, minus\_p95)

	•	envelope\_max\_mm \= max(plus\_max, minus\_max)

cross-eval（固定・ログのみ）：

	•	plus envelope → work\_minus cloud の p95/max

	•	minus envelope → work\_plus cloud の p95/max

⸻

9\. relief 評価（axis\_taper / 完全凍結）

9.1 目的

「軸テーパで逃げを付けた場合の有効逃げ角」が負になっていないかを点列だけで評価する。

（B-Rep形状生成はしない）

9.2 定義（固定）

入力：

	•	selected cutting edge 点列（side別）

	•	各点の tool座標 p=(x,y,z) と edge tangent t（unit）

	•	β \= deg2rad(relief\_angle\_deg)

	•	e\_z \= (0,0,1)

axis\_taper の “逃げ方向” ベクトル（固定モデル）：

	•	r \= hypot(x,y)

	•	r\_hat \= (x/r, y/r, 0)（r=0ならその点は評価スキップ）

	•	逃げは「-Z方向へ進むほど半径が減る」モデルで固定：

	•	d \= \-( e\_z \+ tan(β) \* r\_hat )

edge の法線平面（tに垂直）への射影（固定）：

	•	d\_perp  \= d  \- dot(d,t)\*t

	•	ez\_perp \= e\_z \- dot(e\_z,t)\*t

	•	rh\_perp \= r\_hat \- dot(r\_hat,t)\*t

退避成分（固定）：

	•	if ||ez\_perp|| \< 1e-12 → effective \= β（tがZに平行で射影が壊れる場合）

	•	else if ||rh\_perp|| \< 1e-12 → その点は評価スキップ（幾何退化）

	•	else

	•	u\_ez \= normalize(ez\_perp)

	•	u\_rh \= normalize(rh\_perp)

	•	a \= abs(dot(d\_perp, u\_rh))（法線平面内の半径方向退避）

	•	b \= abs(dot(d\_perp, u\_ez))（法線平面内の軸方向成分）

	•	effective \= atan2(a, b)（rad）

出力：

	•	min\_effective\_relief\_angle\_deg \= rad2deg( min(effective over all evaluated edge points, both sides) )

評価対象点（固定）：

	•	selected edge の valid==1 の点のみ

	•	両side（plus/minus）両方を評価し、全体の min を取る

⸻

10\. geom\_kernel\_stats（reportに要求する統計 / 固定）

共役格子生成（6章）の結果から、report に以下を保存する（SPEC側が要求しているため、この版で定義を固定する）：

	•	n\_total\_points \= 2 \* Nu \* Nv

	•	n\_valid\_points \= count(valid==1 over both sides)

	•	valid\_ratio\_total \= n\_valid\_points / n\_total\_points

	•	valid\_ratio\_plus  \= n\_valid\_plus / (Nu\*Nv)

	•	valid\_ratio\_minus \= n\_valid\_minus / (Nu\*Nv)

残差（固定）：

	•	各 valid 点で residual\_abs \= abs(f(θ1\*))（mm相当）

	•	residual\_abs\_min/p50/p95/max を higher\_order\_stat で集計（p50=median）

theta1\_jump\_count（固定）：

	•	近傍は 4近傍（iu±1,iv) と (iu,iv±1)

	•	両点 valid のとき

	•	d \= abs( wrap\_rad(theta1\[i\]-theta1\[j\]) )

	•	if d \> (PI/2) : jump++（閾値は固定で PI/2）

	•	side別に数えて合算してよい（推奨：plus/minus別もログ）

⸻

11\. 追加（任意だが強く推奨：rawデバッグCSV）

SPEC側に「raw推奨」があるので、幾何核として形式を推奨定義する（DoD外）。

ファイル：

	•	tool\_conjugate\_grid\_plus\_raw.csv

	•	tool\_conjugate\_grid\_minus\_raw.csv

ヘッダ（推奨固定）：

	•	iu,iv,u\_mm,v\_mm,x\_mm,y\_mm,z\_mm,nx,ny,nz,theta1\_rad,theta2\_rad,residual\_abs,valid,reason\_code

※この raw は比較・原因究明のため。Master Truth は正規化後CSV（SPEC側）であることは変わらない。

⸻



⸻

12. テスト用固定サンプル（参照）

目的：

  • 幾何核（共役生成・ゴールデン・包絡・逃げ評価）が仕様通りに動いているかを、最小の固定入力で自動判定できるようにする。

参照：

  • 固定サンプル config と期待出力（ゴールデン）の置き場所・比較方法は SPEC.md の Appendix C を参照する。

推奨（幾何核のテストで最低限見るもの）：

  • golden（7章）の s\_rot\_selected, golden\_p95\_mm, golden\_max\_mm が再現されること。

  • tool\_conjugate\_grid（6章）で valid\_ratio\_total と residual 統計（10章）が再現されること。

  • envelope\_light（8章）で envelope\_p95\_mm / envelope\_max\_mm（plus/minus/worst）が再現されること（edges\_ok==true のケース）。

  • relief（9章）で min\_effective\_relief\_angle\_deg が再現されること（axis\_taper）。

※上記の「再現」は、数値比較ではなく sha256（バイト列一致）で判定するのが本仕様の思想（Master Truth 固定）に合う。

