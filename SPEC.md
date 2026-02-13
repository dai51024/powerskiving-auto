**SPEC.md（ 完全凍結 I/O \+ 決定性 \+ 検証 \+ Visual Pack）**  
SPECバージョン: 1-1-1

対応GEOM\_SPECバージョン: 1-1-1

⸻

0\. 目的

目標ワーク歯面（内歯・平歯）から逆算したパワースカイビング工具（ピニオン型・外周刃固定）について、

	•	共役フランク：flank\_plus / flank\_minus

	•	切れ刃：cutting\_edge\_plus / cutting\_edge\_minus

を生成し、以下を満たす。

	1\.	再現性優先（Master Truth を数値に固定）

	•	CADトポロジ（面×面交線、トリム、ブーリアン）に依存せず、数値点列を真実データとする。

	2\.	見た目で判断できる（Visual Pack 必須）

	•	「それっぽく見えない」問題を仕様で潰す。Visual Pack をDoDに入れる。

	3\.	品質保証（定量検証）

	•	残差・包絡（envelope\_light）・負逃げ・gash露出を定量化し、report.jsonに残す。

	•	干渉は Lean では「診断ログのみ」（DoD外）。

⸻

0.1 非目標（本仕様ではやらない）

	•	“CAM直行”の閉じたB-Repソリッド（imprint/boolean/pack）自動生成（SOLID=1必須はやらない）

	•	full工具（全歯配列）の厳密B-Rep化、隣接overlapの厳密保証

	•	gash/land/relief の完全B-Rep形状生成（本仕様では検証が正）

	•	研削盤向け NC（Gコード/XML）生成

	•	干渉の厳密合否判定（DoDに入れない。diagnostic logのみ）

⸻

0.2 設計原則（ブレたら負け）

	•	Master Truth＝数値点列CSV（grid/edge/uv境界/断面DXF）。これ以外を真実にしない

	•	B-Rep交線は禁止（rake∩flank の交線をCADで取らない）

	•	2-flank必須＝「両側の面/刃/検証が成立」（ソリッド境界化は要求しない）

	•	edges\_ok をゲートにして 暗黙フォールバック禁止（別Zレンジ等へ勝手に逃げない）

	•	plus/minus の定義固定（暗黙禁止）

	•	side正規化（swap）は早期（Step2.1）に確定。後段swapは禁止

	•	決定性の凍結：component\_id / loop\_id / polyline向き / 三角化分割 / percentile / 丸め / 出力桁 / ファイル名

	•	marching squares の曖昧ケース（5/10）は bilinear Asymptotic Decider \+ tie-break 固定

	•	Visual Pack は“派生”だが DoD 必須（見た目で破綻を潰す）

	•	幾何核（共役生成 / work\_target\_surface / relief評価 / envelope\_sampling / golden定義）は GEOM\_SPEC.md に凍結し、geom\_spec\_version を刻印して実装依存を排除する

	•	\-0禁止 / NaN禁止 / Inf禁止 / 指数表記禁止

	•	float直ソート禁止（量子化 q() した値でキー化）

⸻

1\. スコープ

対象：

	•	ワーク：内歯・平歯（helix=0）

	•	工具：ピニオン型（外周刃）固定（tool\_type=“pinion”）

必須成果物（DoD対象）は：

	•	Master Truth（CSV群＋断面DXF）

	•	Visual Pack（STL/STEP：人間が見て判断できる）

	•	report.json / cad\_report\_stepXX.json / manifest.json

trimmed face STEP は best-effort（DoD外）。

⸻

2\. 用語

	•	Work：被切削歯車（内歯）

	•	Tool：スカイビングカッター（ピニオン型）

	•	W0座標：ワーク固定基準（θ2=0 の基準歯面が定義される座標）

	•	W座標（World）：ワーク中心固定の世界座標（W0と同一原点・同一軸、ただしワーク回転θ2を適用した結果がW）

	•	T座標：工具座標（工具軸 \+Z\_tool、工具回転角 θ1）

	•	flank\_plus / flank\_minus：工具側共役フランク（左右）

	•	rake\_face：すくい面（本仕様では平面）

	•	cutting\_edge\_plus/minus：rake\_face と flank\_side の交線に相当する曲線（数値抽出点列が真実）

	•	envelope\_light：edge sweep近似で評価する包絡誤差

	•	Visual Pack：表示メッシュ＋刃チューブ＋参照（OD\_REF/rake/軸）を合成したSTL/STEP群（見た目で破綻を検出）

⸻

3\. 座標系・符号規約（最重要）

3.1 W0/W座標

	•	原点：ワーク中心

	•	\+Z：ワーク軸

	•	\+X/+Y：右手系

	•	θ2：ワーク回転（+Z回り、右ねじ）

回転行列（固定）：

	•	Rz(θ)=

\[ cosθ \-sinθ 0

  sinθ  cosθ 0

  0      0   1 \]

W0→W（固定、feedなし）：

	•	p\_W \= Rz(θ2) \* p\_W0

	•	n\_W \= Rz(θ2) \* n\_W0

3.2 工具中心と中心距離

	•	中心距離 a \[mm\]

	•	工具中心（W座標）：

	•	c1\_W \= (0, a, 0\) を唯一の正（他は禁止）

3.3 工具軸の傾き（Ry固定）

	•	交差角 Σ（tilt）

	•	工具軸は Ry(Σ) を採用（Rxは禁止）

Ry（固定定義：右手系、+Σで \+Zが+X側へ倒れる）：

	•	Ry(Σ)=

\[ cosΣ 0 sinΣ

  0    1 0

 \-sinΣ 0 cosΣ \]

3.4 Tool角度 θ の定義（固定）

	•	Tool座標Tで theta\_raw \= atan2(y, x)

	•	theta \= wrap\_rad(theta\_raw)

	•	theta=0 は \+X\_tool

	•	theta正方向：+Z\_tool から見て反時計回り

3.4.0 wrap規約（凍結：実装割れ禁止）

定数（固定）：

	•	PI \= 3.141592653589793

	•	TWO\_PI \= 6.283185307179586

wrap\_rad（固定：値域 \[-π, \+π)）：

	•	wrap\_rad(x) \= x \- TWO\_PI \* floor((x \+ PI)/TWO\_PI)

	•	よって wrap\_rad(PI) \= \-PI（+PIを出さない）

wrap\_deg（固定：値域 \[-180, \+180)）：

	•	wrap\_deg(x\_deg) \= rad2deg( wrap\_rad(deg2rad(x\_deg)) )

禁止：

	•	(-π,+π\] など別仕様のwrap

	•	言語組み込みの余り演算や round に依存したwrap

3.4.1 flank\_plus / flank\_minus の定義（暗黙禁止）

	•	歯中心角：theta\_tooth\_center\_rad（Tool座標）

	•	任意点で dtheta \= wrap\_rad(theta \- theta\_tooth\_center\_rad)

命名規約（固定）：

	•	plus：dtheta \> 0 側（+theta側）

	•	minus：dtheta \< 0 側（-theta側）

	•	dtheta==0 は deadband 扱い（side判定に使わない）

report必須：

	•	flank\_side\_definition \= “plus\_is\_positive\_dtheta\_about\_theta\_tooth\_center”

	•	theta\_tooth\_center\_rad / deg

3.5 Tool↔World 変換（凍結：実装割れ防止の中核）

工具の回転は「工具の局所Z軸回り（Tool座標でのRz）」に定義する（固定）。

Tool→World（固定）：

	•	R\_WT(Σ,θ1) \= Ry(Σ) \* Rz(θ1)

	•	p\_W \= c1\_W \+ R\_WT \* p\_T

	•	n\_W \= R\_WT \* n\_T

World→Tool（固定）：

	•	R\_TW(Σ,θ1) \= Rz(-θ1) \* Ry(-Σ)

	•	p\_T \= R\_TW \* (p\_W \- c1\_W)

	•	n\_T \= R\_TW \* n\_W

W0→Tool 合成（固定、feedなし）：

	•	p\_T \= Rz(-θ1) \* Ry(-Σ) \* ( Rz(θ2)\*p\_W0 \- c1\_W )

	•	n\_T \= Rz(-θ1) \* Ry(-Σ) \* ( Rz(θ2)\*n\_W0 )

3.5.1 サニティテスト（必須）

以下を満たさない実装は 座標規約違反で即停止（DoD不合格）：

	•	Σ=0, θ1=0 のとき：R\_WT \= I（tool軸はWの+Zと平行）

	•	任意 p\_T=(0,0,0) に対して：p\_W \= c1\_W

	•	Σ=0, θ1=0, θ2=0 のとき：p\_T \= p\_W \- c1\_W

	•	Σ\>0, θ1=0 のとき：Toolの \+Z軸（World表現）は (sinΣ, 0, cosΣ)

3.6 回転比（符号は自動決定→以後固定）

	•	theta2 \= s\_rot \* (z1/z2) \* theta1 \+ theta2\_0

	•	s\_rot ∈ {+1, \-1}

s\_rotの扱い（固定）：

	•	s\_rot は Σ=0ゴールデンで自動決定し report に保存

	•	以後、その run では固定（手変更禁止）

	•	envelope\_light を含む全工程で同じ s\_rot を使用する

⸻

4\. 入力（config契約：必須キーの完全リスト）

4.0 版管理（必須）

  • spec\_version（必須）

  • 固定値："1-1-1"

  • geom\_spec\_version（必須）

  • 固定値："1-1-1"

  • 欠落または不一致：ConfigError（即停止）


4.0.1 config ファイル形式（必須：凍結）

目的：入力の揺れ（JSON/YAML差、BOM、改行、コメント等）で config\_sha256 や再現性が割れないようにする。

  • config は **JSONファイル** とする（YAML禁止、TOML禁止）。

  • 文字コード：UTF-8（BOM無し）

  • 改行：LF固定（CRLF禁止）

  • JSON仕様：RFC 8259 相当の「厳密JSON」

    • コメント禁止（//, /* */ など）

    • trailing comma 禁止

    • 重複キー禁止（同一キーが複数回出現したら ConfigError）

  • 数値表記（入力側も凍結）：

    • NaN/Inf 禁止（拡張パーサで受けない）

    • 指数表記禁止（例：1e-3 禁止）

    • -0 禁止（例：-0, -0.0 は ConfigError）

    • 0 は +0 として扱う

禁止：

  • YAML/JSON5/JSONC など「厳密JSON以外」を config として受けること

  • パーサの「重複キーは後勝ち」などの暗黙挙動に依存すること（必ず ConfigError）

注意：

  • 入力値そのものの量子化（q）は「出力・ソートの決定性」のための規約であり、入力値を暗黙に丸めない（暗黙丸め禁止）。

4.0.2 config\_sha256 の定義（必須：凍結）

  • config\_sha256 は、読み込んだ config ファイルの **生バイト列**（UTF-8）に対して sha256 を計算した 16進小文字文字列とする。

    • 同一ファイル（同一バイト列）なら OS や言語が違っても必ず一致する。

  • sha256 の対象は「ファイル内容のみ」。パス名や OS メタデータは含めない。

  • 解析後のJSON（キー順正規化・空白除去など）から hash を作るのは禁止（実装差が増えるため）。

  • config を標準入力など「ファイル以外」で受け取る場合は、受け取ったバイト列そのものを対象に sha256 を計算する。


4.1 ワーク

  • module\_mm

  • z2（内歯）

  • pressure\_angle\_deg

  • face\_width\_mm（mm）

  • helix\_beta\_deg（固定：0。0以外はConfigError）

  • work\_target\_model（固定："ideal\_involute"。それ以外はConfigError）

派生量（固定：本文側で決める。GEOM\_SPECでも同定義を使う）

	•	z\_w\_min\_mm \= \-0.5 \* face\_width\_mm

	•	z\_w\_max\_mm \= \+0.5 \* face\_width\_mm

4.2 工具

	•	tool\_type=“pinion”（固定）

	•	z1

	•	center\_distance\_a\_mm

	•	sigma\_rad（=Σ）

	•	rake\_angle\_deg

	•	relief\_mode（“axis\_taper” をDoD対象の評価モードとして推奨）

	•	relief\_angle\_deg

	•	land\_width\_mm

4.3 rake参照点（※二重定義禁止）

	•	ref\_point\_mode \= (“grid\_uv” | “manual\_xyz”)

grid\_uv：

	•	iu\_ref, iv\_ref（正規化後の grid\_plus を参照する）

manual\_xyz：

	•	x\_ref\_mm, y\_ref\_mm, z\_ref\_mm（Tool座標）

派生量（入力ではない・report必須）：

	•	rake\_ref\_point\_T

	•	theta\_ref \= wrap\_rad(atan2(y\_ref, x\_ref))

4.4 歯中心角（Tool座標）

	•	tooth\_center\_theta\_mode \= (“zero” | “pitch\_snap”)

	•	theta\_tooth\_center\_rad（mode=zeroなら0、pitch\_snapなら k\_tooth\*(2π/z1)）

4.5 歯セクタ角（候補フィルタ）

	•	theta\_sector0\_rad \= theta\_tooth\_center\_rad \- π/z1

	•	theta\_sector1\_rad \= theta\_tooth\_center\_rad \+ π/z1

	•	theta\_sector\_margin\_deg

4.6 OD（外径）ルール

	•	r\_od\_margin\_mm

	•	r\_od\_pad\_mm

	•	used\_z\_pad\_mm（この名前に統一。別名は禁止）

	•	edge\_min\_length\_mm

4.7 gash露出検証（深さ成立のみ）

本仕様のDoD対象は「深さ成立」だけとし、位相・本数は DoD外（ログ用途）に降格する。

必須（DoDで使用）：

  • gash\_depth\_from\_od\_mm（必須）

  • gash\_depth\_margin\_mm（必須）

任意（DoD外：将来拡張/可視化用。欠落しても ConfigError にしない）：

  • gash\_count（推奨：z1。未指定なら report に null）

  • gash\_phase\_mode（例："tooth\_center\_snap"。未指定なら report に null）

  • gash\_phase\_offset\_deg（未指定なら 0 とみなすのは禁止。report に null）

4.8 サンプリング・許容値・閾値（必須）

共役格子：

	•	Nu, Nv（int）

       • grid\_u\_min\_mm（必須：mm半径。GEOM\_SPECでは work\_target\_surface の u は「半径(mm)」）

        • grid\_u\_max\_mm（必須：mm半径。grid\_u\_min\_mm \< grid\_u\_max\_mm）

刃点列（DoD 7.4）：

	•	edge\_chord\_tol\_mm（等弧長リサンプル規約に使用：Step5.5）

	•	edge\_on\_face\_tol\_mm

	•	edge\_on\_rake\_tol\_mm

tangent規約（Step5.3）：

	•	z\_span\_min\_mm

	•	t\_cross\_min

断面DXF（Step7）：

	•	z\_sections\_mm\[\]（空は禁止）

逃げ評価（DoD 7.7）：

	•	relief\_margin\_deg

ゴールデン（Step1 / DoD 7.1）：

	•	golden\_tol\_p95\_mm

	•	golden\_tol\_max\_mm

	•	golden\_dz\_mm

	•	golden\_dz\_max\_mm

	•	golden\_min\_points

　  •     golden\_pitch\_band\_dr\_mm（必須）

　  •     golden\_ref\_n（必須：int\>=20）

包絡（Step9 / DoD 7.8）：

	•	tol\_p95\_mm

	•	tol\_max\_mm

     • theta1\_range\_deg（必須）

     • theta1\_step\_deg（必須）

　  • multi\_rev\_k\_min（必須 int）

      • multi\_rev\_k\_max（必須 int, k\_max\>=k\_min）

      • edge\_stride（必須 int\>=1）

      • theta2\_offset\_deg（必須）

      • n\_target\_u（必須 int\>=1）

      • n\_target\_v（必須 int\>=1）

      • target\_u\_min\_mm（必須）

      • target\_u\_max\_mm（必須）

      • target\_r\_filter\_mode（必須："off" | "pitch\_band"）

      • pitch\_band\_dr\_mm（pitch\_band時に使用。キー自体は必須）

invalid（DoD 7.5）：

	•	invalid\_tol\_total（例：0.005）

	•	edge\_band\_width\_cells（0以上整数）

4.9 side正規化の堅牢化（必須）

	•	dtheta\_deadband\_rad

	•	n\_deadband\_eval\_min

4.10 marching squares 共通パラメータ（必須）

	•	ms\_iso\_eps

	•	ms\_fc\_eps

	•	ms\_eps\_d

	•	ms\_tie\_break \= “pair\_B”（固定）

4.11 uv境界ループ分類（必須）

	•	uv\_on\_edge\_eps

	•	hole\_test\_grid\_n（奇数。推奨21）

	•	hole\_test\_grid\_min\_inside\_points（推奨1）

4.12 決定性の量子化（必須）

float直ソート禁止。ソートキーは量子化する。

	•	sort\_round\_rad\_digits

	•	sort\_round\_mm\_digits

	•	sort\_round\_uv\_digits

	•	sort\_round\_area\_digits

量子化関数（固定。言語組み込みroundは禁止）：

	•	sign(0)=+1 とする（-0禁止のため）

	•	q(x, d) \= sign(x) \* floor(|x|\*10^d \+ 0.5) / 10^d

（0.5は0から遠い方向へ丸める）

4.13 percentile 定義（必須）

	•	percentile\_method \= “higher\_order\_stat”（固定）

定義：

N個の値 x を昇順ソートし、p∈(0,1\] に対して

	•	k \= ceil(p\*N) \- 1

	•	percentile(p) \= x\[ clamp(k, 0, N-1) \]

p95 は p=0.95、p01 は p=0.01、p50(median)は p=0.5。

4.14 CSV出力の丸め・表記（必須：言語依存を殺す）

	•	csv\_float\_digits\_mm

	•	csv\_float\_digits\_rad

	•	csv\_float\_digits\_unitless（追加：nx/tx等の桁）

規約（固定）：

	•	CSVに出す float は必ず事前に q() を適用してから文字列化する（printf任せ禁止）

	•	文字列は固定小数（指数表記禁止）

	•	\-0 は禁止。q後に \-0.0 が出るなら \+0.0 に正規化してから出す

	•	NaN/Inf 禁止（出すくらいなら valid=0 & 値は0で埋めて reason\_code を立てる）

⸻

5\. 出力（必須成果物 \+ optional）

5.0 出力ディレクトリ

	•	output\_dir/configで指定

	•	Step0で stale output を全削除（混入禁止）

5.1 Master Truth（数値：必須）

	•	tool\_conjugate\_grid\_plus.csv / tool\_conjugate\_grid\_minus.csv（正規化後）

	•	cutting\_edge\_points.csv（候補含む。selectedは仕様ルールで固定）

	•	flank\_uv\_boundary\_plus.csv / flank\_uv\_boundary\_minus.csv（uv境界ループ）

	•	sections\_{ZTAG}.dxf（z\_sections\_mm\[\]ぶん必須）

	•	report.json

	•	cad\_report\_stepXX.json（例外でも必須）

	•	manifest.json（必須：全成果物sha256）

（任意・推奨：デバッグ用）

	•	tool\_conjugate\_grid\_plus\_raw.csv / tool\_conjugate\_grid\_minus\_raw.csv（swap前）

	•	trimmed face STEP（DoD外）

5.2 Visual Pack（必須：DoD対象）

「見た目で壊れてない」を判断するための派生物。Master Truth からのみ生成する。

必須（DoD）：

	•	flank\_boundary\_wire\_plus.step / flank\_boundary\_wire\_minus.step

	•	uv境界→3D写像ワイヤ（B-Repトリム禁止。grid双一次補間で写像）

	•	flank\_view\_mesh\_plus.stl / flank\_view\_mesh\_minus.stl

	•	validセルのみ三角化した“クリップ済み表示メッシュ”

	•	cutting\_edge\_tube\_plus.stl / cutting\_edge\_tube\_minus.stl（edges\_ok==true のとき必須）

	•	cutting\_edge の太線化（チューブメッシュ）

	•	tool\_visual\_one\_tooth.stl（必須）

	•	flank\_view\_mesh± \+ edge\_tube± \+ OD\_REFリング \+ rake\_patch \+ refs を合成

	•	tool\_visual\_preview\_ring.stl（必須）

	•	tool\_visual\_one\_tooth を歯ピッチで回転複製したプレビュー

best-effort（DoD外：ただし出せるなら強く推奨）：

	•	tool\_one\_tooth\_view.step（ワイヤのみで構成。面無しデフォルト）

	•	rake\_patch境界（wire）

	•	cutting\_edge±（wire）

	•	boundary\_wire±（wire）

	•	OD\_REF（wire：円はポリライン近似で可）

	•	refs（軸・原点）

禁止：

	•	r\_od円筒を“ボディ”として入れる（刃が内側に見える誤解を作る）

	•	STL→STEP変換で工具形状を作る

	•	STEPにソリッド／面（B-Rep）を入れる（この版の意図に反する）

5.2.6 STL/STEP 出力の決定性

目的：

  • 同一の Master Truth（CSV）から生成される Visual Pack が、実装差で形状や三角形列挙順が割れないようにする。

  • 少なくとも「同一実装・同一config」でのバイト列一致を保証できる仕様に寄せる。

共通（STL/STEP）：

  • 文字コード：ASCII（7bit）を使用（UTF-8でも同じだが、実装ぶれを減らす）

  • 改行：LF固定

  • 指数表記禁止（例：1e-3 禁止）

  • NaN/Inf 禁止

  • \-0 禁止（q()後に \-0.000... が出たら \+0.000... に正規化）

  • 数値は必ず q() を適用してから固定小数で出力する（printf任せ禁止）

    \- mm座標：q(x, csv\_float\_digits\_mm)

    \- unitless（法線など）：q(x, csv\_float\_digits\_unitless)

5.2.7 ASCII STL ライタ凍結

(1) STL形式：

  • ASCII STL 固定（Binary STLは禁止）

(2) facet normal：

  • facet normal は必ず出す

  • normal は三角形の幾何から計算する（頂点順で決まる）

  • 出力 normal は q(., csv\_float\_digits\_unitless)

(3) ヘッダ/フッタ（固定）：

  • 1行目：solid \<basename\>

  • 最終行：endsolid \<basename\>

  • \<basename\> はファイル名から拡張子を除いた文字列（例：tool\_visual\_one\_tooth）

(4) vertex 出力順：

  • 各三角形は (p0,p1,p2) の順で出力

  • p0/p1/p2 の座標は q() 後の固定小数

(5) 三角形列挙順：

  • 仕様で規定された生成アルゴリズムのループ順をそのまま出力順とする（後段で再ソート禁止）

5.2.8 STEP（wire-only）ライタ凍結

対象：

  • flank\_boundary\_wire\_plus/minus.step

  • （best-effortの one\_tooth\_view.step を実装する場合も同じ規約）

(1) STEP形式：

  • ISO-10303-21 の Part21 形式（テキスト）固定

  • FILE\_SCHEMA は 'CONFIG\_CONTROL\_DESIGN'（AP203）固定

  • FILE\_NAME の timestamp は固定文字列 '1970-01-01T00:00:00'（決定性のため）

  • 単位は mm（MILLI, METRE）を unit context として定義する（テンプレ固定）

(2) 幾何表現：

  • CARTESIAN\_POINT と POLYLINE のみを使用

  • 円やスプラインは使わない（必要ならポリライン近似）

(3) 閉ループの表現（固定）：

  • CSV側は closed点（終点=始点）を持たないが、STEPの POLYLINE は「最後に先頭点を1回だけ繰り返して閉じる」

  • 例：P0..P(n-1) を持つ場合、POLYLINE点列は (P0,P1,...,P(n-1),P0)

(4) entity 番号（固定）：

  • \#1 から連番で、出力順＝生成順

  • コンテキスト/プロダクト定義ブロック（固定テンプレ）→ 幾何（点→線）→ curve set → shape repr の順

  • 途中で番号を飛ばさない

(5) polyline/point の出力順（固定）：

  • loop\_id 昇順

  • loop内 point\_id 昇順

  • plusファイルも minusファイルも「そのsideのloop\_id順」を守る（勝手に外周優先など禁止）

⸻

5.3 CSVスキーマ（凍結）

5.3.0 CSV共通（凍結）

	•	UTF-8, LF

	•	指数表記禁止

	•	NaN/Inf禁止

	•	\-0禁止（出力直前に \-0.xxx を 0.xxx に正規化）

	•	小数桁固定（列ごとに指定）

	•	ファイル生成順は必ず plus → minus

5.3.1 tool\_conjugate\_grid\_{side}.csv（正規化後）

side \= plus / minus

ヘッダ順固定（これ以外禁止）：

	•	iu,iv,u\_idx,v\_idx,x\_mm,y\_mm,z\_mm,nx,ny,nz,valid,reason\_code

列定義：

	•	iu, iv : int（0..Nu-1, 0..Nv-1）

	•	u\_idx, v\_idx : float（grid index座標。必ず u\_idx=iu, v\_idx=iv を出す。出力は q(., sort\_round\_uv\_digits)・小数 sort\_round\_uv\_digits 桁固定）

	•	x\_mm,y\_mm,z\_mm : float（mm、出力は q(., csv\_float\_digits\_mm)・小数 csv\_float\_digits\_mm 桁固定）

	•	nx,ny,nz : float（unitless、出力は q(., csv\_float\_digits\_unitless)・小数 csv\_float\_digits\_unitless 桁固定）

	•	valid : int（0/1）

	•	reason\_code : string（enum：Appendix A）

行順（固定）：

	•	iv昇順（0→Nv-1）、同iv内で iu昇順（0→Nu-1）

invalid点の数値（固定）：

	•	計算不能の場合は x=y=z=0, nx=ny=nz=0 を出す（NaN禁止）

	•	reason\_code で理由を必ず立てる

5.3.2 cutting\_edge\_points.csv（Master Truth）

ヘッダ順固定（これ以外禁止）：

	•	edge\_side,edge\_component\_id,selected,point\_id,u\_idx,v\_idx,

x\_mm,y\_mm,z\_mm,nx,ny,nz,tx,ty,tz,

s\_mm,plane\_dist\_mm,valid,reason\_code

列定義：

	•	edge\_side : “plus” | “minus”

	•	edge\_component\_id : int（Step5.4の決定的付番）

	•	selected : int（0/1）

	•	point\_id : int（component内連番。向き規約後の先頭が0。resample後は resample後点列の連番）

	•	u\_idx,v\_idx : float（格子index空間。出力は q(., sort\_round\_uv\_digits)・小数 sort\_round\_uv\_digits 桁固定）

	•	x\_mm,y\_mm,z\_mm : float（Tool座標。出力は q(., csv\_float\_digits\_mm)）

	•	nx,ny,nz : float（Tool座標フランク法線。出力は q(., csv\_float\_digits\_unitless)）

	•	tx,ty,tz : float（Tool座標接線。出力は q(., csv\_float\_digits\_unitless)）

	•	s\_mm : float（弧長。出力は q(., csv\_float\_digits\_mm)）

	•	plane\_dist\_mm : float（rake平面符号距離。出力は q(., csv\_float\_digits\_mm)）

	•	valid : int（0/1）

	•	reason\_code : string（Appendix A）

行順（固定）：

	•	edge\_side の順：plus → minus

	•	各side内：edge\_component\_id 昇順

	•	各component内：point\_id 昇順

5.3.3 flank\_uv\_boundary\_{side}.csv（uv境界ループ）

	•	closed点（終点=始点）を配列に含めない（閉じは論理で扱う）

ヘッダ順固定：

	•	loop\_id,point\_id,is\_hole,parent\_loop\_id,

u\_idx,v\_idx,u\_idx\_q,v\_idx\_q,

area\_uv,area\_uv\_q,

centroid\_u,centroid\_v,centroid\_u\_q,centroid\_v\_q,

length\_uv,length\_uv\_q,

loop\_flipped

列定義（丸め）：

	•	u\_idx,v\_idx : q(., sort\_round\_uv\_digits) を出し、小数 sort\_round\_uv\_digits 桁固定

	•	u\_idx\_q,v\_idx\_q : u\_idx,v\_idx と同値でよい（必須列）

	•	area\_uv : q(., sort\_round\_area\_digits) を出し、小数 sort\_round\_area\_digits 桁固定

	•	centroid\_\* : q(., sort\_round\_uv\_digits)

	•	length\_uv : q(., sort\_round\_uv\_digits)

	•	\*\_q : 同値でよい（必須列として残す）

行順（固定）：

	•	loop\_id 昇順 → point\_id 昇順

⸻

6\. パイプライン（工程契約）

6.0 共通アルゴリズム規約（決定性のための固定）

6.0.1 marching squares（全用途共通：仕様の核）

用途：

	•	Step3A（valid領域境界）

	•	Step5（edge抽出：plane\_dist=0）

	•	Step7（flank断面：z-z\_sec=0）

セル座標（固定）：

	•	u\_idx は \+X、v\_idx は \+Y

	•	corners:

	•	p00=(i,j), p10=(i+1,j), p11=(i+1,j+1), p01=(i,j+1)

	•	edges（固定）：

	•	e0: bottom(p00-p10), e1:right(p10-p11), e2:top(p01-p11), e3:left(p00-p01)

(1) クランプ（固定）：

	•	f’ \= 0 if |f| \<= ms\_iso\_eps else f

(2) ビット化（固定。0は負側扱い）：

	•	b \= 1 if f’ \> 0 else 0

	•	case\_id \= (b00\<\<0) | (b10\<\<1) | (b11\<\<2) | (b01\<\<3)

※ f’==0 は b=0（負側）

(3) 交点算出（固定）：

各辺(a,b)で f\_a’, f\_b’ を使い交点tを決める

	•	if f\_a’==0 and f\_b’==0: t=0.5

	•	elif f\_a’==0: t=0

	•	elif f\_b’==0: t=1

	•	elif f\_a’\*f\_b’ \< 0: t \= f\_a’ / (f\_a’ \- f\_b’)

	•	else: 交点無し

交点座標は (u,v) を線形補間（固定）。

(4) 接続テーブル（固定：corner順/edge順に依存）

非曖昧ケースのセグメント（edgeペア）：

	•	0: \-

	•	1: (e3,e0)

	•	2: (e0,e1)

	•	3: (e3,e1)

	•	4: (e1,e2)

	•	6: (e0,e2)

	•	7: (e3,e2)

	•	8: (e2,e3)

	•	9: (e0,e2)

	•	11:(e1,e2)

	•	12:(e1,e3)

	•	13:(e0,e1)

	•	14:(e0,e3)

	•	15:-

※(a,b)は常に edge\_id の昇順で保存。複数セグメントは辞書順で並べる。

(5) 曖昧ケース（5/10）：bilinear Asymptotic Decider（必須）

対象：case\_id==5 または 10

	•	d   \= f00’ \- f10’ \- f01’ \+ f11’

	•	det \= f00’\*f11’ \- f10’\*f01’

	•	if |d| \> ms\_eps\_d: f\_s \= det / d

	•	else: f\_s 未定義（tie-breakへ）

	•	pair\_A \= {(e0,e3),(e1,e2)}

	•	pair\_B \= {(e0,e1),(e2,e3)}  ※ms\_tie\_break=“pair\_B” を固定

tie-break条件（固定）：

	•	|d| \<= ms\_eps\_d  OR  |f\_s| \<= ms\_fc\_eps → 常に pair\_B

符号での選択（固定）：

	•	case 5:  f\_s \> 0 → pair\_B、f\_s \< 0 → pair\_A

	•	case 10: f\_s \> 0 → pair\_A、f\_s \< 0 → pair\_B

	•	tie-breakは常に pair\_B

6.0.2 polyline/loop 組み立て（決定性：凍結）

端点キー（固定）：

	•	key \= ( q(u\_idx, sort\_round\_uv\_digits), q(v\_idx, sort\_round\_uv\_digits) )

セグメント正規化（固定）：

	•	seg\_key \= (min(key0,key1), max(key0,key1)) の辞書順

	•	これでセグメント配列を辞書順ソートしてから処理する（決定性）

連結（固定）：

	•	adjacency: endpoint\_key \-\> 接続セグメント一覧（seg\_key順を保持）

	•	component抽出は「未訪問セグメントのうち最小seg\_keyから開始」

	•	open/closed判定：

	•	次数1端点が存在するなら open（startは次数1端点のkey最小）

	•	次数1端点が無いなら closed（startはendpoint\_key最小）

	•	次のセグメント選択：

	•	現在端点に接続し、かつ未訪問の seg のうち seg\_key 最小を選ぶ

	•	分岐（次数\>2）が一度でも出た場合：marching\_squares\_error で即停止（割れ防止）

6.0.3 特徴量の凍結（component/loop のソートに使う値は式で固定）

(1) length

	•	uv境界ループの length は 2D（length\_uv）を使う

	•	edge component の length は 3D（length\_3d\_mm）を使う

	•	DXF断面 component の length は 2D（length\_xy\_mm）を使う

(2) area\_uv（shoelace固定）

	•	area\_uv \= 0.5 \* Σ (u\_i\*v\_{i+1} \- u\_{i+1}\*v\_i)（循環）

(3) centroid\_uv（多角形重心固定）

	•	A \= area\_uv

	•	|A| \< eps\_area\_uv の場合：centroid \= 頂点平均

	•	それ以外：

	•	Cx \= (1/(6A)) \* Σ ( (u\_i+u\_{i+1})(u\_iv\_{i+1}-u\_{i+1}\*v\_i) )

	•	Cy \= (1/(6A)) \* Σ ( (v\_i+v\_{i+1})(u\_iv\_{i+1}-u\_{i+1}\*v\_i) )

eps固定：

	•	eps\_area\_uv \= 1e-12

	•	eps\_theta\_mean \= 1e-12

(4) theta\_comp（circular mean固定）

	•	theta\_i \= wrap\_rad(atan2(y\_i, x\_i))

	•	mx \= mean(cos(theta\_i)), my \= mean(sin(theta\_i))

	•	if mx^2+my^2 \< eps\_theta\_mean: theta\_comp \= theta\_0（先頭点）

	•	else: theta\_comp \= wrap\_rad(atan2(my, mx))

6.0.4 percentile（共通）

percentile\_method=“higher\_order\_stat” を必ず使う（補間禁止）。

6.0.5 出力ファイル名の決定性（断面DXF）

ZTAG（固定）：

	•	z\_q \= q(z\_sec, sort\_round\_mm\_digits)

	•	ZTAG \= “z” \+ sign \+ (整数部5桁ゼロ埋め) \+ “.” \+ (小数部 sort\_round\_mm\_digits 桁)

	•	例（digits=3）：+12.5 → z+00012.500、-3 → z-00003.000

ファイル名（固定）：

	•	sections\_{ZTAG}.dxf

	•	例：sections\_z+00012.500.dxf

6.0.6 view mesh（三角化・法線・表裏：決定性）

三角化（固定）：

	•	validセル（4隅valid==1）のみ対象

	•	対角線 p00→p11

	•	tri：

	•	(p00,p10,p11)

	•	(p00,p11,p01)

	•	tri出力順：iv昇順→iu昇順→上記2tri順

表裏（固定）：

	•	各セルで平均法線 n\_cell \= normalize(n00+n10+n11+n01)（ゼロなら(0,0,0)）

	•	tri法線 n\_tri \= normalize((p1-p0)×(p2-p0))

	•	dot(n\_tri, n\_cell) \< 0 の場合、p1とp2をswapして反転（外向きを揃える）

（これでビューアの“真っ黒”事故を減らす）

6.0.7 tube mesh（ポリライン太線化）の凍結

用途：

  • Step8.1 cutting\_edge\_tube

  • Step8.3 OD\_REFリング（※リングは専用式も使うが、基本は同じ思想）

  • Step8.4 refs（軸・原点マーカー）

入力：

  • 中心線 polyline：P\[k\], k=0..N-1（N\>=2、隣接重複点は事前に除去）

  • 半径 r\_mm

  • sides S（整数\>=3）

基本方針：

  • “parallel transport” で断面フレームを伝播し、ねじれ割れを抑えつつ決定性を確保する。

(0) 事前処理（固定）：

  • 連続点で ||P\[k+1\]-P\[k\]|| \< 1e-12 の点は削除する（削除後も N\>=2 が必要）

  • 削除後 N\<2 なら component\_invalid として、その要素の生成は failed（statusに残す）

(1) tangent の定義（固定）：

  • seg\[k\] \= normalize(P\[k+1\]-P\[k\]) for k=0..N-2

  • t\[0\]     \= seg\[0\]

  • t\[N-1\]   \= seg\[N-2\]

  • t\[k\] (1\<=k\<=N-2)：

      a=seg\[k-1\], b=seg\[k\]

      t\_raw \= a \+ b

      ||t\_raw|| \< 1e-12 の場合：t\[k\]=b（折返し退化の固定対応）

      それ以外：t\[k\]=normalize(t\_raw)

(2) 初期フレーム（固定）：

  • up \= (0,0,1)

  • if |dot(t\[0\], up)| \>= 0.99 then up \= (0,1,0)

  • n\[0\] \= normalize(cross(up, t\[0\]))

  • b\[0\] \= cross(t\[0\], n\[0\])

  • ここで cross の結果が 1e-12 未満なら（最悪ケース）：

      up=(1,0,0) に固定で切替えて同様に計算し、それでもダメなら failed

(3) フレーム伝播（parallel transport / 固定）：

  • for k=1..N-1:

      v \= cross(t\[k-1\], t\[k\])

      s \= ||v||

      c \= dot(t\[k-1\], t\[k\])

      if s \< 1e-12:

         n\[k\]=n\[k-1\], b\[k\]=b\[k-1\]

      else:

         axis \= v / s

         angle \= atan2(s, c)

         Rodrigues回転で n\[k-1\], b\[k-1\] を axis, angle で回し n\[k\], b\[k\] を得る

      （Rodrigues式そのものは標準定義。実装依存にしない）

(4) 断面頂点（固定）：

  • phi\[j\] \= TWO\_PI \* j / S, j=0..S-1

  • V\[k,j\] \= P\[k\] \+ r\_mm\*( cos(phi\[j\])\*n\[k\] \+ sin(phi\[j\])\*b\[k\] )

(5) 側面三角形（固定：cap無し）：

  • for k=0..N-2:

      for j=0..S-1:

        j2=(j+1)%S

        tri1 \= (V\[k,j\],   V\[k+1,j\], V\[k+1,j2\])

        tri2 \= (V\[k,j\],   V\[k+1,j2\],V\[k,j2\])

  • tri列挙順は k昇順→j昇順→(tri1,tri2) の順に固定

(6) 法線と表裏（固定）：

  • tri法線 n\_tri \= normalize((p1-p0)×(p2-p0))

  • STLに出す normal は n\_tri をそのまま使う（再推定・平滑化禁止）

⸻

Step0: I/O準備

	•	stale output削除（混入禁止）

	•	失敗時も cad\_report\_step00.json を必ず出す（io\_missing\_or\_corrupt など）

Step1: Σ=0 ゴールデン & s\_rot決定（必須ゲート）

幾何定義（参照インボリュート等）と比較手順は GEOM\_SPEC.md に凍結し、本SPECでは I/O と合否条件のみ固定する。

	•	s\_rot=+1/-1 両方で Σ=0 評価し、誤差が小さい方を採用

	•	z\_mid\_tool と帯域抽出は golden\_\* パラメータに従う（GEOM\_SPEC定義）

	•	reportへ保存：s\_rot\_selected、golden\_p95\_mm、golden\_max\_mm、z\_mid\_tool、golden\_dz\_used\_mm

不合格なら即停止（固定）：

	•	golden\_p95\_mm \> golden\_tol\_p95\_mm OR golden\_max\_mm \> golden\_tol\_max\_mm

Step2: 共役点群生成（raw）

	•	s\_rot\_selected を使って raw を生成（幾何核はGEOM\_SPEC）

	•	rawは invalid を落とさず reason\_code を立てる

Step2.1: side正規化（swap）※必須・ここで確定

評価点定義（凍結）：

	•	eval\_points \= { valid=1 かつ |dtheta| \> dtheta\_deadband\_rad }

	•	eval\_count \= count(eval\_points)

	•	eval\_count \< n\_deadband\_eval\_min → side\_normalization\_ambiguous で即停止

pos\_fraction（凍結）：

	•	pos\_fraction \= count(dtheta\>0 in eval\_points) / eval\_count

swap判定（固定）：

	•	(raw\_plus\_pos\_fraction\>=0.5 AND raw\_minus\_pos\_fraction\<=0.5) → no swap

	•	(raw\_plus\_pos\_fraction\<=0.5 AND raw\_minus\_pos\_fraction\>=0.5) → swap

	•	それ以外 → side\_normalization\_ambiguous（即停止）

出力：

	•	tool\_conjugate\_grid\_plus/minus.csv（正規化後）

	•	report必須：

	•	side\_swapped

	•	raw\_to\_normalized\_mapping

	•	raw\_plus\_pos\_fraction / raw\_minus\_pos\_fraction

	•	dtheta\_deadband\_rad

	•	eval\_count

Step2.2: grid法線の向き正規化（必須：見た目/接線の揺れ防止）

正規化後の grid\_plus/minus（valid==1）について：

	•	r\_hat \= normalize(\[x,y,0\])（r=0ならスキップ）

	•	dot(n, r\_hat) \< 0 なら n=-n（外向きに揃える）

	•	その後 n を正規化

report必須：

	•	normal\_flip\_count\_plus/minus

	•	normal\_flip\_rate\_plus/minus

Step3: uv境界抽出・分類（DoD対象）

Step3A: uv境界抽出（決定性固定）

	•	境界抽出スカラー場（凍結）：s \= \+1（valid==1） / \-1（valid==0）

	•	外側取り逃し防止（凍結）：

	•	valid\_mask を 1セル外側にパディングし、パディング領域は valid=0（s=-1）

	•	marching squares はパディング後格子で実行し、得た (u\_idx,v\_idx) から (1,1) を引いて元格子へ戻す

	•	s=0 等高線を marching squares（6.0.1）で抽出

	•	component組立は 6.0.2

Step3A-1: ループ分類（決定性：必須）

対象：各component（closed loop を想定、openが混ざったら failed\_validation）

(1) strictly-inside 判定

	•	point-in-polygon：ray casting

	•	on-edge判定 eps：uv\_on\_edge\_eps

	•	on-edge は inside とみなさない（strict）

(2) p\_test の決定（凹形状対策：必須）

	•	centroid\_uv を 6.0.3 で計算

	•	centroid が strictly-inside なら p\_test=centroid

	•	そうでなければ bbox を hole\_test\_grid\_n×hole\_test\_grid\_n の固定格子で走査し、

lexicographic（u→v）順で最初に見つかった strictly-inside 点を p\_test とする（決定的）

	•	strictly-inside 点が hole\_test\_grid\_min\_inside\_points 未満なら hole\_classification\_ambiguous で即停止

(3) 親子関係と is\_hole（必須）

	•	ループ i の p\_test がループ j に strictly-inside なら j は i の container

	•	parent は containers のうち |area\_uv| が最小のもの（最内）

	•	depth \= 0 if parent=-1 else parent.depth+1

	•	is\_hole \= (depth % 2 \== 1\)

(4) 向き規約（必須）

	•	outer（is\_hole=0）は CCW（area\_uv\>0）

	•	hole（is\_hole=1）は CW（area\_uv\<0）

逆なら反転し loop\_flipped=1

(5) start点規約（必須）

	•	(u\_idx\_q, v\_idx\_q) の辞書順最小点を start とし rotate

	•	closed点（終点=始点）を配列に含めない

(6) loop\_id 付番（探索順禁止：必須）

特徴量（量子化使用）：

	•	is\_hole

	•	area\_abs\_q \= |q(area\_uv, sort\_round\_area\_digits)|

	•	centroid\_u\_q, centroid\_v\_q

	•	length\_uv\_q

	•	n\_points

ソートキー（固定）：

	•	(is\_hole ASC, area\_abs\_q DESC, centroid\_u\_q ASC, centroid\_v\_q ASC, length\_uv\_q DESC, n\_points DESC)

この順で loop\_id=0.. を付与

(7) 自己交差チェック（DoD）

	•	非隣接辺同士の交差があれば不合格（collinear重なりもNG）

	•	eps は uv\_on\_edge\_eps

	•	交差が出たら failed\_validation で即停止し、交差ペアを report に保存

Step3A-2: uv→3D boundary wire（必須）

	•	uv境界点を grid の双一次補間（u\_idx,v\_idx）で 3Dへ写像（B-Rep禁止）

	•	出力：flank\_boundary\_wire\_plus/minus.step

※STEPの表現は「ポリライン（CARTESIAN\_POINT \+ POLYLINE）」で良い（円やスプライン不要）。

（実装者がCADカーネル無しで出せることが目的）

Step3B: view mesh（STL：DoD対象）

	•	6.0.6 に従い三角化、表裏、出力順を固定

	•	出力：flank\_view\_mesh\_plus/minus.stl

Step3C: trimmed STEP（best-effort：DoD外）

	•	uv境界ループを使って trimmed face を試作し STEPへ

	•	失敗してもDoDにしない。trim\_status を report に必須で残す

Step4: rake\_face（平面）確定（n\_rake凍結）

theta\_ref \= wrap\_rad(atan2(y\_ref, x\_ref))

局所基底（Tool座標、凍結）：

	•	e\_z \= (0,0,1)

	•	e\_r \= (cos(theta\_ref), sin(theta\_ref), 0\)

	•	e\_t \= (-sin(theta\_ref), cos(theta\_ref), 0\)   \# \+theta方向

rake角 γ \= deg2rad(rake\_angle\_deg)

n\_rake（凍結）：

	•	n\_rake \= normalize( cos(γ)\*e\_t \- sin(γ)\*e\_r )

平面：

	•	n\_rake · (p \- rake\_ref\_point\_T) \= 0

report必須：

	•	rake\_ref\_point\_T, theta\_ref, n\_rake, rake\_angle\_deg

Step5: cutting\_edge 抽出（数値法固定：B-Rep交線禁止）

Step5.0 plane\_dist の評価

	•	plane\_dist\_mm \= n\_rake·(p-rake\_ref\_point\_T) を grid点で評価

invalid corner を含むセルの扱い（凍結）：

	•	セル4隅のどれかが valid=0 の場合、そのセルは marching squares 対象外（セグメント生成しない）

Step5.1 marching squares で edge component 抽出

	•	marching squares（6.0.1）で d=0 等高線 → component群

	•	component組立は 6.0.2

	•	(u\_idx,v\_idx) から双一次補間で3D復元（p, n\_flank）

Step5.2 n\_flank の向き規約（凍結）

	•	n\_flank は bilinear 補間→正規化

	•	radial\_unit \= normalize((x,y,0))（r=0ならスキップ）

	•	dot(n\_flank, radial\_unit) \< 0 の場合、n\_flank \= \-n\_flank に反転（外向きを揃える）

Step5.3 tangent 規約（必須：凍結）

各点で：

	•	t\_raw\_i \= normalize(cross(n\_rake, n\_flank\_i))

	•	||cross|| \< t\_cross\_min の点は valid=0、reason\_code=DEGENERATE\_CROSS

component向き決定（固定）：

	•	端点差分から dz と dtheta を算出

	•	|dz| \>= z\_span\_min\_mm なら dz\>=0 となる向きに反転

	•	そうでなければ dtheta\>=0 となる向きに反転

score一括反転（固定）：

	•	t\_poly\_i \= normalize(p\_{i+1}-p\_{i-1})（中央差分、端点は片側）

	•	score \= Σ dot(t\_raw\_i, t\_poly\_i) over valid points

	•	score \< 0 → component全点で t \= \-t\_raw

	•	score \>=0 → t \= t\_raw

弧長 s\_mm：

	•	上記向きに沿って累積、s=0は先頭点

Step5.4 component\_id の決定性（探索順禁止：必須）

特徴量（量子化してソートに使う）：

	•	in\_sector（歯セクタ内：theta\_sector0..1 \+ margin）

	•	theta\_comp\_q（6.0.3のtheta\_comp → q(., sort\_round\_rad\_digits)）

	•	abs\_delta\_q \= q(|wrap(theta\_comp-theta\_ref)|, sort\_round\_rad\_digits)

	•	length\_q \= q(length\_3d\_mm, sort\_round\_mm\_digits)

	•	n\_points

付番ソート（固定）：

	•	(in\_sector DESC, abs\_delta\_q ASC, length\_q DESC, theta\_comp\_q ASC, n\_points DESC)

Step5.5 selected=1 の選択（固定）

	•	歯セクタ外は除外

	•	残りで abs\_delta\_q 最小

	•	tie-break：length\_q 最大

	•	選択理由と全候補特徴量を report に残す

Step5.6 edgeリサンプル（凍結：edge\_chord\_tol\_mmを使う）

目的：製造・比較のため点間距離を揃える（実装割れ禁止）。

	•	selected component（sideごとに1本）についてのみリサンプルする

	•	total\_length \= length\_3d\_mm（valid==1の点のみで算出）

	•	N \= max(2, ceil(total\_length / edge\_chord\_tol\_mm) \+ 1\)

	•	s\_target\_k \= k \* total\_length / (N-1), k=0..N-1

	•	線分上の線形補間で p(s\_target) を作る

	•	uv も同様に s パラメータで線形補間（u\_idx,v\_idxを出すため）

	•	n\_flank は uv から再補間して求める（Step5.2規約を適用）

	•	plane\_dist\_mm と t はリサンプル後に再計算する（Step5.3規約）

Step5.7 edges\_ok ゲート（必須）

	•	selected\_plus\_length\_mm / selected\_minus\_length\_mm を算出（リサンプル後点列でOK）

	•	edge\_plus\_ok \= status ok AND length\>=edge\_min\_length\_mm

	•	edge\_minus\_ok \= status ok AND length\>=edge\_min\_length\_mm

	•	edges\_ok \= edge\_plus\_ok AND edge\_minus\_ok

	•	edges\_ok=false のとき tool\_z\_min/max \= null

	•	以後のOD/gash/DXF/envelope/干渉diag 等の後工程はスキップ（暗黙フォールバック禁止）

Step5.8 edge\_band（DoD用セル集合）の定義（凍結）

	•	selected edge（リサンプル後）の各点 (u\_idx,v\_idx) について cell=(floor(u\_idx), floor(v\_idx))

	•	bandセル集合 \= Manhattan距離 ≤ edge\_band\_width\_cells のセル全体

	•	band内の各セルについて「4隅格子点 valid=1」が必須

	•	1つでも満たさない場合：DoD不合格

Step5.9: OD決定（edges\_ok前提）

	•	used\_z\_range=\[tool\_z\_min-used\_z\_pad\_mm, tool\_z\_max+used\_z\_pad\_mm\]

	•	r\_grid\_max\_used：plus/minus 両側×used\_z\_range 内でmax

	•	r\_edge\_max：edge±でmax

	•	r\_od=max(r\_grid\_max\_used, r\_edge\_max)+r\_od\_margin\_mm+r\_od\_pad\_mm

	•	reportへ内訳必須

Step6: relief/land（検証）

	•	DoD対象：axis\_taper の min\_effective\_relief\_angle\_deg

	•	reportへ：min\_effective\_relief\_angle\_deg

※評価の幾何核は GEOM\_SPEC.md（geom\_spec\_version）に従う。

Step6.5: gash露出（深さ成立）

	•	req\_depth\_i \= r\_od \- r\_i（edge点）

	•	req\_depth\_max \= max(req\_depth\_i)

	•	合格：gash\_depth\_from\_od\_mm \>= req\_depth\_max \+ gash\_depth\_margin\_mm

	•	reportへ必須

Step7: 断面DXF（Master Truth由来固定 / B-Rep section禁止）

	•	z\_sections\_mm\[\] の各 z\_sec について出力（空は禁止）

	•	ファイル名は 6.0.5 のZTAG規約に従う：sections\_{ZTAG}.dxf

7.1 flank断面（grid由来）

	•	g(u,v)=z(u,v)-z\_sec を格子点で評価

	•	marching squares（6.0.1）で g=0 等高線 → component群（セル4隅validのみ）

	•	component組立は 6.0.2

	•	双一次補間で (x,y,z\_sec) → 2D polyline

7.1.1 複数componentの扱い（決定性：必須）

特徴量（量子化を使う）：

	•	in\_sector

	•	theta\_comp\_q（6.0.3）

	•	dist\_center\_q \= q(|wrap(theta\_comp-theta\_tooth\_center)|, sort\_round\_rad\_digits)

	•	length\_xy\_q \= q(length\_xy\_mm, sort\_round\_mm\_digits)

	•	n\_points

component\_id ソート（固定）：

	•	(in\_sector DESC, dist\_center\_q ASC, length\_xy\_q DESC, theta\_comp\_q ASC, n\_points DESC)

採用（固定）：

	•	in\_sector外を除外

	•	dist\_center\_q 最小、tie-break length\_xy\_q 最大

	•	採用1本のみを DXF の FLANK\_\* に出す

	•	除外は report に記録（z\_sec/理由）

7.2 edge断面（edge点列由来）

	•	selected edge polyline と z=z\_sec の交点を線形補間で求める

	•	DXFは POINT（固定推奨）

7.3 OD\_REF / RAKE

	•	OD\_REF：半径r\_odの円（CIRCLE）

	•	RAKE：rake平面と z=z\_sec の交線（LINE）

DXF仕様（固定）：

	•	DXFバージョン：R2000

	•	FLANK\_\*：LWPOLYLINE（頂点順は component点順）

	•	EDGE\_\*：POINT

	•	RAKE：LINE

	•	OD\_REF：CIRCLE

	•	単位：mm

Step8: Visual Pack 生成（DoD必須）

Visual Pack は Master Truth から決定的に生成する。

Step8.1 cutting\_edge チューブ（edges\_ok==true 前提）

固定パラメータ（凍結：config不要）：

	•	edge\_tube\_radius\_mm \= 0.05 \* module\_mm

	•	edge\_tube\_sides \= 12

	•	selected edge polyline を中心線としてチューブメッシュを生成

	•	出力：

	•	cutting\_edge\_tube\_plus.stl

	•	cutting\_edge\_tube\_minus.stl

Step8.1 追記：

  • tube mesh の生成は 6.0.7 に従う（cap無し）

  • 中心線 polyline は resample後の selected edge 点列（point\_id昇順）をそのまま使う

  • 三角形列挙順は 6.0.7 の k昇順→j昇順→(tri1,tri2) を保持（後段ソート禁止）

Step8.2 rake\_patch（参照板 / 完全凍結）

目的：

  • rake平面がどこにあるかを視覚で確認するための薄板。

  • “edgeのbbox投影”等の曖昧表現は禁止。寸法・生成法を完全凍結する。

固定パラメータ（凍結）：

  • rake\_patch\_scale \= 1.2

  • rake\_patch\_thickness\_mm \= max(0.01, 0.02 \* module\_mm)

  • rake\_patch\_pad\_s\_mm \= 0.5 \* module\_mm

  • rake\_patch\_pad\_z\_mm \= 0.5 \* module\_mm

  • rake\_patch\_min\_half\_s\_mm \= 1.0 \* module\_mm

  • rake\_patch\_min\_half\_z\_mm \= 1.0 \* module\_mm

パッチ座標系（Tool座標、凍結）：

  • e\_z \= (0,0,1)

  • n\_rake は Step4 で確定したもの

  • e\_s \= normalize(cross(n\_rake, e\_z))

    \- ||cross|| \< 1e-12 の場合（理論上ほぼ無いが）：

      e\_s \= e\_r（Step4の e\_r）に固定し、以後の計算を続行

  • これで rake平面内の2軸が (e\_s, e\_z) に固定される

対象点集合 S\_source（凍結）：

  • edges\_ok==true のとき：

      S\_source \= selected edge（resample後）の全点（plusとminus両方）

  • edges\_ok==false のとき：

      S\_source \= valid==1 の grid点（plus/minus両方、全点）

スカラー範囲（凍結）：

  • s\_ref \= dot(rake\_ref\_point\_T, e\_s)

  • z\_ref \= rake\_ref\_point\_T.z

  • s\_min \= min(dot(p, e\_s) for p in S\_source)

  • s\_max \= max(dot(p, e\_s) for p in S\_source)

  • z\_min \= min(p.z for p in S\_source)

  • z\_max \= max(p.z for p in S\_source)

半寸法（凍結：rake\_ref\_point中心で必ず覆う）：

  • half\_s \= rake\_patch\_scale \* max(|s\_min \- s\_ref|, |s\_max \- s\_ref|) \+ rake\_patch\_pad\_s\_mm

  • half\_z \= rake\_patch\_scale \* max(|z\_min \- z\_ref|, |z\_max \- z\_ref|) \+ rake\_patch\_pad\_z\_mm

  • half\_s \= max(half\_s, rake\_patch\_min\_half\_s\_mm)

  • half\_z \= max(half\_z, rake\_patch\_min\_half\_z\_mm)

矩形（rake平面上、中心=rake\_ref\_point\_T、凍結）：

  • C \= rake\_ref\_point\_T

  • corners（平面中央面）：

      Q0 \= C \+ (+half\_s)\*e\_s \+ (+half\_z)\*e\_z

      Q1 \= C \+ (-half\_s)\*e\_s \+ (+half\_z)\*e\_z

      Q2 \= C \+ (-half\_s)\*e\_s \+ (-half\_z)\*e\_z

      Q3 \= C \+ (+half\_s)\*e\_s \+ (-half\_z)\*e\_z

厚み付け（凍結）：

  • t \= rake\_patch\_thickness\_mm

  • 上面（+n\_rake側）：Qi\_top \= Qi \+ 0.5\*t\*n\_rake

  • 下面（-n\_rake側）：Qi\_bot \= Qi \- 0.5\*t\*n\_rake

三角化（凍結：対角線は 0→2）：

  • 上面（外向き normal ≈ \+n\_rake）：

      (Q0\_top, Q1\_top, Q2\_top), (Q0\_top, Q2\_top, Q3\_top)

  • 下面（外向き normal ≈ \-n\_rake）：

      (Q0\_bot, Q2\_bot, Q1\_bot), (Q0\_bot, Q3\_bot, Q2\_bot)

  • 側面（各辺を2三角形、辺順は 0-1,1-2,2-3,3-0 固定）：

      例：辺0-1

        (Q0\_top, Q0\_bot, Q1\_bot), (Q0\_top, Q1\_bot, Q1\_top)

      同様に残り3辺

出力：

  • rake\_patch は tool\_visual\_one\_tooth に合成される（単体STLは作らない仕様でも可）

report必須（数値は q() 後の値を保存）：

  • rake\_patch: half\_s\_mm, half\_z\_mm, thickness\_mm, pad\_s\_mm, pad\_z\_mm

  • rake\_patch: source\_mode ("edges" | "grid")

  • rake\_patch: s\_min,s\_max,z\_min,z\_max

Step8.3 OD\_REFリング（参照）

Step8.3 OD\_REFリング（参照 / r2で生成法を凍結）

固定パラメータ（凍結）：

  • od\_ref\_ring\_radius\_mm \= r\_od

  • od\_ref\_ring\_tube\_radius\_mm \= 0.02 \* module\_mm

  • od\_ref\_ring\_sides \= 12

  • od\_ref\_ring\_major\_segments \= 180（固定）

  • z\_ring \= z\_mid\_tool（Step1で確定した値）

中心線（凍結：閉曲線、角度は \[-pi, \+pi) の等分割）：

  • for i=0..M-1（M=od\_ref\_ring\_major\_segments）:

      ang \= \-PI \+ TWO\_PI \* i / M

      P\[i\] \= (r\_od\*cos(ang), r\_od\*sin(ang), z\_ring)

リングメッシュ（凍結）：

  • 中心線 polyline は \[P\[0\],P\[1\],...,P\[M-1\],P\[0\]\] として閉じる（最後に先頭点を1回だけ追加）

  • tube mesh は 6.0.7 を使用（cap無し）

report必須：

  • od\_ref\_ring: major\_segments, tube\_radius\_mm, sides, z\_ring

Step8.4 refs（軸・原点マーカー / 完全凍結）

目的：

  • Tool座標の \+Z軸方向と原点を見失わないための参照形状。

  • 「決定式」の曖昧表現は禁止。長さも太さも凍結する。

固定パラメータ（凍結）：

  • refs\_rod\_radius\_mm \= max(0.005, 0.01 \* module\_mm)

  • refs\_z\_pad\_mm \= 0.5 \* module\_mm

  • refs\_origin\_arm\_len\_mm \= 0.6 \* module\_mm

  • refs\_origin\_arm\_count \= 2（X/Y の2本。Zは軸棒で兼ねる）

  • refs\_rod\_sides \= 12（tube mesh と同じ）

Z範囲（凍結）：

  • edges\_ok==true のとき：

      z0 \= tool\_z\_min

      z1 \= tool\_z\_max

  • edges\_ok==false のとき：

      z0 \= min(z of valid grid points over both sides)

      z1 \= max(z of valid grid points over both sides)

  • z0/z1 が取れない（validがゼロ）場合は refs 生成 failed とし status に残す

軸棒中心線（凍結）：

  • zc \= 0.5\*(z0+z1)

  • Lz \= (z1 \- z0) \+ 2\*refs\_z\_pad\_mm

  • P\_axis0 \= (0,0, zc \- 0.5\*Lz)

  • P\_axis1 \= (0,0, zc \+ 0.5\*Lz)

軸棒メッシュ（凍結）：

  • 中心線 polyline \= \[P\_axis0, P\_axis1\]

  • tube mesh は 6.0.7 を使用（cap無し）

原点アーム（凍結）：

  • Xアーム：中心線 \[(-refs\_origin\_arm\_len\_mm,0,0),(+refs\_origin\_arm\_len\_mm,0,0)\]

  • Yアーム：中心線 \[(0,-refs\_origin\_arm\_len\_mm,0),(0,+refs\_origin\_arm\_len\_mm,0)\]

  • それぞれ 6.0.7 の tube mesh を使用（cap無し）

合成（凍結）：

  • refs メッシュの三角形列挙順：

      1\) 軸棒（Z）

      2\) Xアーム

      3\) Yアーム

report必須：

  • refs: rod\_radius\_mm, rod\_sides

  • refs: z0,z1, zc, Lz, z\_pad\_mm

  • refs: origin\_arm\_len\_mm

Step8.5 tool\_visual\_one\_tooth.stl（合成）

	•	flank\_view\_mesh\_plus/minus

	•	cutting\_edge\_tube\_plus/minus（edges\_ok==trueなら必須、falseなら省略してstatusに残す）

	•	rake\_patch

	•	OD\_REFリング（r\_odがあれば）

	•	refs

を合成して出力。  
Step8.5 追記（合成順の凍結）

tool\_visual\_one\_tooth.stl の三角形列挙順は以下の“固定順”で連結する：

  1\) flank\_view\_mesh\_plus

  2\) flank\_view\_mesh\_minus

  3\) cutting\_edge\_tube\_plus（edges\_ok==true のときのみ）

  4\) cutting\_edge\_tube\_minus（edges\_ok==true のときのみ）

  5\) OD\_REFリング（r\_od が定義されるときのみ）

  6\) rake\_patch

  7\) refs

禁止：

  • 連結後に三角形を面積や座標で再ソートすること

  • ビューア都合で法線再推定して頂点順を入れ替えること

Step8.6 tool\_visual\_preview\_ring.stl（回転複製）

固定パラメータ（凍結）：

	•	preview\_tooth\_count\_odd \= 7（奇数固定）

	•	Δθ \= 2π / z1

	•	k \= \-(n-1)/2 .. \+(n-1)/2 の順で複製（k昇順固定）

report必須：

	•	tri\_count\_one\_tooth

	•	tri\_count\_preview\_ring

	•	status\_edge\_tube\_plus/minus

	•	status\_od\_ref\_ring

	•	status\_rake\_patch

	•	status\_refs

Step9: envelope\_light（固定）

※サンプリング仕様・target cloud仕様は GEOM\_SPEC.md（geom\_spec\_version）に従う。

本SPECでは I/O と必須ログを固定する。

前提：

	•	edges\_ok==true のときのみ実行（falseなら skipped\_due\_to\_edges\_not\_ok）

report必須：

	•	envelope\_plus\_p95\_mm, envelope\_plus\_max\_mm

	•	envelope\_minus\_p95\_mm, envelope\_minus\_max\_mm

	•	envelope\_p95\_mm \= max(plus\_p95, minus\_p95)

	•	envelope\_max\_mm \= max(plus\_max, minus\_max)

	•	cross\_eval\_plus\_to\_minus\_p95/max

	•	cross\_eval\_minus\_to\_plus\_p95/max

percentileは 4.13 の定義に従う。

Step9.5 interference\_diag（診断ログのみ：DoD外）

	•	出すなら report に keepout\_radius\_mm / clearance\_min\_mm / clearance\_p01\_mm / negative\_count を保存

	•	percentileは 4.13 に従う

Step10: pack（必須）

	•	report.json と cad\_report\_stepXX.json を揃える

	•	manifest.json を生成（sha256、spec\_version、geom\_spec\_version、config hash、ファイル一覧）

	•	stale混入禁止

⸻

7\. DoD（Definition of Done）

7.0 成果物の完全性（必須）

必須成果物がすべて存在し、読み取りに成功すること。

	•	CSV：ヘッダ一致、行パース可能、NaN/Inf無し、指数表記無し、-0無し

	•	DXF：必須レイヤが存在（FLANK\_, EDGE\_, RAKE, OD\_REF）

	•	STL：読み込み可能、三角形数\>0

	•	STEP（boundary wire）：読み込み可能（空でないwireがある）

	•	Visual Pack：tool\_visual\_one\_tooth.stl / tool\_visual\_preview\_ring.stl が読み込み可能で tri\_count\>0

	•	manifest.json：全必須ファイルを列挙しsha256が入っている

	•	失敗した場合：DoD不合格として停止し、欠落/破損ファイル名を report に残す

7.1 ゴールデン（必須）

	•	Σ=0 ゴールデン合格（golden\_p95\_mm\<=golden\_tol\_p95\_mm かつ golden\_max\_mm\<=golden\_tol\_max\_mm）

	•	s\_rot\_selected が決定され report に保存されている

7.2 side正規化（必須）

	•	Step2.1 が ambiguous でない（eval\_count\>=n\_deadband\_eval\_min）

	•	side\_swapped / mapping / eval\_count / pos\_fraction が report に存在

7.3 面（トリム済み面をDoDにしない）

必須（DoD）：

	•	uv境界ループが存在（plus/minus両方）

	•	uv境界の自己交差なし（Step3A-1(7)で判定）

	•	loop\_id / start点 / 向き / outer-hole / parent が決定され report に残る

	•	boundary\_wire が生成できる（plus/minus）

	•	view mesh STL が生成できる（plus/minus）

DoD外（best-effort）：

	•	trimmed STEP の成否（trim\_status をreportに残すだけ）

7.4 切れ刃（必須）

	•	edge\_plus/minus が空でない（候補が出る）

	•	edges\_ok==true

	•	edgeがrake平面上：max|plane\_dist|\<=edge\_on\_rake\_tol\_mm

	•	edgeが対応flank上：u\_idx,v\_idx補間で評価し dist\<=edge\_on\_face\_tol\_mm（最近点投影は禁止）

	•	tangent規約（dz/dtheta/score）が report に残る

	•	component\_id 決定性ログが存在

	•	edges\_ok==true の場合：cutting\_edge\_tube\_plus/minus.stl が生成されている（tri\_count\>0）

7.5 invalid（必須）

	•	invalid\_rate\_total \<= invalid\_tol\_total

	•	used\_z\_range×edge\_band（width=edge\_band\_width\_cells）に invalid=0

	•	selected抽出セルに invalid 混入が無い（混入したら不合格）

7.6 OD / gash露出（必須）

	•	r\_od が定義される（edges\_ok前提）

	•	gash\_depth\_from\_od\_mm が必要深さ \+ margin を満たす

7.7 逃げ（必須）

	•	axis\_taper の min\_effective\_relief\_angle\_deg \>= relief\_margin\_deg

7.8 包絡（必須：envelope\_light）

	•	envelope\_p95(worst) \<= tol\_p95\_mm

	•	envelope\_max(worst) \<= tol\_max\_mm

	•	plus/minus別の値が report に保存されている

※干渉は診断ログのみ（DoD外）

7.9 Visual Pack（必須）

	•	tool\_visual\_one\_tooth.stl が読み込み可能で tri\_count\>0

	•	tool\_visual\_preview\_ring.stl が読み込み可能で tri\_count\>0

	•	report に Visual Pack の主要ログ（tri\_count、各要素status、寸法）が存在

⸻

8\. report / cad\_report / manifest（必須キー）

8.1 report.json（必須）

最低限必須キー（欠落はDoD不合格）：

	•	spec\_version

	•	geom\_spec\_version

	•	golden: s\_rot\_selected, golden\_p95\_mm, golden\_max\_mm, z\_mid\_tool, golden\_dz\_used\_mm

	•	side: theta\_tooth\_center\_rad, side\_swapped, raw\_to\_normalized\_mapping,

raw\_plus\_pos\_fraction, raw\_minus\_pos\_fraction, dtheta\_deadband\_rad, eval\_count

	•	transforms: a\_mm, sigma\_rad, (sanity\_test\_passed=true)

	•	marching\_squares: ms\_iso\_eps, ms\_fc\_eps, ms\_eps\_d, ms\_tie\_break

	•	normals: normal\_flip\_count\_plus/minus, normal\_flip\_rate\_plus/minus

	•	surface\_uv\_boundary: loop統計（loop\_id,is\_hole,parent,area\_uv,centroid,length\_uv,flipped, self\_intersection\_ok）

	•	rake: rake\_ref\_point\_T, theta\_ref, n\_rake, rake\_angle\_deg

	•	edge:

	•	候補特徴量一覧（side, component\_id, in\_sector, theta\_comp, abs\_delta, length\_mm, selected理由）

	•	tangent（dz/dtheta/score）

	•	plane\_dist\_max\_abs\_mm（selected side別）

	•	edges\_ok: edge\_plus\_ok, edge\_minus\_ok, edges\_ok, selected\_plus\_length\_mm, selected\_minus\_length\_mm,

tool\_z\_min, tool\_z\_max

	•	OD: used\_z\_range, r\_od, 内訳（r\_grid\_max\_used, r\_edge\_max）

	•	gash: req\_depth\_max, pass

	•	envelope: plus/minus/worst, cross-eval（診断）

	•	visual\_pack:

	•	tri\_count\_one\_tooth, tri\_count\_preview\_ring

	•	status\_edge\_tube\_plus/minus, status\_od\_ref\_ring, status\_rake\_patch, status\_refs

	•	主要寸法（edge\_tube\_radius\_mm 等）

	•	io\_check: 必須ファイルの存在/bytes/sha256（manifestと一致）

	•	runtime: 実行時間や環境（任意キー可）

8.2 cad\_report\_stepXX.json（必須）

	•	step\_id（例：“step5\_edge\_extract” など固定文字列）

	•	status（enum）

	•	error\_code（enum。statusと同一でもよい）

	•	message（文字列）

	•	exception\_stacktrace（任意）

8.3 manifest.json（必須）

必須キー：

	•	spec\_version

	•	geom\_spec\_version

	•	config\_sha256

	•	created\_utc

	•	files\[\]（各ファイルの path, sha256, bytes）

	•	（任意）git\_commit

⸻


8.3.1 manifest.json の自己参照・列挙順・created\_utc（必須：凍結）

自己参照（必須）：

  • files[] には **manifest.json 自身を含めない**（自己参照で決定性が壊れるため）。

    • 禁止：path=="manifest.json"（output\_dir 配下でも同名なら禁止）

  • 同一 path の重複は許可しない（重複は failed\_validation）。

列挙順（必須：決定性）：

  • files[] の列挙順は path の辞書順（昇順）で固定する。

    • “生成した順” や OS のディレクトリ列挙順に依存しないこと。

created\_utc（必須：決定性）：

  • manifest.json の created\_utc は、決定性のため **固定値** を使用する：

    • created\_utc = "1970-01-01T00:00:00Z"

  • 実際の実行時刻が必要なら、report.json の runtime（任意キー）に記録する（manifest に書かない）。

config ファイルの扱い（固定）：

  • 入力 config ファイル自体は原則 files[] に含めない（入力であり出力成果物ではない）。

  • ただし、入力 config を output\_dir にコピーして同梱する運用を取る場合は、そのコピーされたファイルは files[] に含めてよい（その場合 path を明示し、sha256/bytes はコピー後ファイルに対して計算する）。

8.4 JSON 出力の文字列化規約（report.json / cad\_report\_stepXX.json / manifest.json）（必須：凍結）

目的：同一データから、言語・JSONライブラリ差で JSON のバイト列が割れないようにする。

共通（3ファイルすべて）：

  • 文字コード：UTF-8（BOM無し）

  • 改行：LF固定（CRLF禁止）

  • ファイル末尾：末尾に LF を 1 つ付与する（空ファイル禁止）

  • JSON最外層：object（配列を最外層にするのは禁止）

  • インデント：2スペース固定（タブ禁止）

  • オブジェクトのキー順：辞書順（Unicode code point 順）で昇順固定（sort\_keys=true 相当）

  • 配列要素順：仕様で規定された順序をそのまま保持する（後段ソート禁止）

  • 文字列：JSON標準のエスケープのみ（制御文字は必ずエスケープ）。\uXXXX の使用は任意だが、同一実装内で揺らさないこと。

数値（共通）：

  • 整数：10進表記のみ（先頭ゼロ禁止）

  • 浮動小数：

    • 必ず事前に q(x,d) を適用してから文字列化する（printf任せ禁止）

    • 固定小数で出す（指数表記禁止）

    • -0 禁止（出力直前に -0.xxx は +0.xxx に正規化）

    • NaN/Inf 禁止（出すくらいなら status=failed / valid=0 を立てる）

浮動小数の d（桁）の決定（凍結）：

  • 値を保持する「キー名の末尾」で決める（実装差を殺す）。

    • *_mm  → d = csv\_float\_digits\_mm

    • *_rad → d = csv\_float\_digits\_rad

    • *_deg → d = csv\_float\_digits\_unitless

    • *_ratio / *_fraction / *_rate → d = csv\_float\_digits\_unitless

    • *_tol / *_eps → d = csv\_float\_digits\_unitless

    • それ以外の float → d = csv\_float\_digits\_unitless

  • 配列・ネストの場合：

    • 直接のキー名が存在する場合はそのキー名の規約を使う。

    • 配列要素が裸の数値で直接のキー名が無い場合は、配列を保持するキー名の規約を使う。

例外（cad\_report）：

  • cad\_report\_stepXX.json の exception\_stacktrace は決定性を壊しやすいので、標準出力では null（またはキー省略）を推奨する。

    • デバッグ用途で入れる場合も、改行は LF に正規化し、パス等の環境依存情報をそのまま入れないこと（DoD外）。


Appendix A. enum（固定）

A.1 reason\_code（grid/edge共通）

	•	OK

	•	SOLVER\_FAIL

	•	OUTSIDE\_DOMAIN

	•	INVALID\_NORMAL

	•	DEGENERATE\_CROSS

	•	INVALID\_CELL\_CORNER

	•	NAN\_FORBIDDEN\_REPLACED\_WITH\_ZERO

A.2 status（step/cad\_report）

	•	ok

	•	failed

	•	skipped\_due\_to\_edges\_not\_ok

	•	skipped\_due\_to\_missing\_od

	•	side\_normalization\_ambiguous

	•	hole\_classification\_ambiguous

	•	io\_missing\_or\_corrupt

	•	golden\_failed

	•	failed\_validation

	•	marching\_squares\_error

⸻

Appendix B. 幾何核（GEOM\_SPEC.md）

	•	本SPECは「I/O＋決定性＋検証＋Visual Pack」を凍結する。

	•	共役生成 / work\_target\_surface / relief評価 / envelope\_sampling / golden比較法 の幾何核は

GEOM\_SPEC.md（geom\_spec\_version="1-1-1"）に凍結し、

config/report/manifest で一致を保証する。



⸻

Appendix C. テスト強制のための固定サンプル（推奨：仕様書固定・自動ループ用）

目的：

  • 「正しい」を人間の解釈に依存させず、最小の固定入力と固定期待出力（ゴールデン）で自動判定できるようにする。

  • 本仕様は Master Truth を数値点列に固定しているため、ゴールデンの比較は sha256（バイト列一致）で行うのが最も強い。

C.1 固定パス（推奨）

  • configs/smoke\_v1-1-1.json

  • fixtures/spec\_1-1-1\_geom\_1-1-1/smoke\_expected/

    • このディレクトリは「期待出力の正」を格納する（コミットされる想定）。

    • manifest.json の files[] は manifest.json 自身を含めない（8.3.1）。

C.2 smoke config（最小例：厳密JSON / 指数表記なし）

以下は「最小で回る」ことを目的に Nu/Nv やサンプル数を小さくした例である。  
（値は例。実際の製造条件は別途調整すること。）

```json
{
  "spec_version": "1-1-1",
  "geom_spec_version": "1-1-1",

  "output_dir": "out/smoke",

  "module_mm": 2.0,
  "z2": 60,
  "pressure_angle_deg": 20.0,
  "face_width_mm": 10.0,
  "helix_beta_deg": 0.0,
  "work_target_model": "ideal_involute",

  "tool_type": "pinion",
  "z1": 20,
  "center_distance_a_mm": 40.0,
  "sigma_rad": 0.2617993877991494,

  "rake_angle_deg": 5.0,

  "relief_mode": "axis_taper",
  "relief_angle_deg": 6.0,
  "land_width_mm": 1.0,

  "ref_point_mode": "manual_xyz",
  "x_ref_mm": 20.0,
  "y_ref_mm": 0.0,
  "z_ref_mm": 0.0,

  "tooth_center_theta_mode": "zero",
  "theta_tooth_center_rad": 0.0,

  "theta_sector_margin_deg": 5.0,

  "r_od_margin_mm": 0.5,
  "r_od_pad_mm": 0.5,
  "used_z_pad_mm": 0.5,
  "edge_min_length_mm": 3.0,

  "gash_depth_from_od_mm": 5.0,
  "gash_depth_margin_mm": 0.2,
  "gash_count": null,
  "gash_phase_mode": null,
  "gash_phase_offset_deg": null,

  "Nu": 21,
  "Nv": 11,
  "grid_u_min_mm": 56.5,
  "grid_u_max_mm": 65.0,

  "edge_chord_tol_mm": 0.5,
  "edge_on_face_tol_mm": 0.05,
  "edge_on_rake_tol_mm": 0.05,

  "z_span_min_mm": 1.0,
  "t_cross_min": 0.000001,

  "z_sections_mm": [0.0],

  "relief_margin_deg": 1.0,

  "golden_tol_p95_mm": 0.02,
  "golden_tol_max_mm": 0.05,
  "golden_dz_mm": 0.2,
  "golden_dz_max_mm": 3.0,
  "golden_min_points": 50,
  "golden_pitch_band_dr_mm": 1.0,
  "golden_ref_n": 40,

  "tol_p95_mm": 0.05,
  "tol_max_mm": 0.1,
  "theta1_range_deg": 10.0,
  "theta1_step_deg": 2.0,
  "multi_rev_k_min": 0,
  "multi_rev_k_max": 0,
  "edge_stride": 1,
  "theta2_offset_deg": 0.0,
  "n_target_u": 10,
  "n_target_v": 5,
  "target_u_min_mm": 56.5,
  "target_u_max_mm": 65.0,
  "target_r_filter_mode": "off",
  "pitch_band_dr_mm": 2.0,

  "invalid_tol_total": 0.005,
  "edge_band_width_cells": 1,

  "dtheta_deadband_rad": 0.03490658503988659,
  "n_deadband_eval_min": 30,

  "ms_iso_eps": 0.000000000001,
  "ms_fc_eps": 0.000000000001,
  "ms_eps_d": 0.000000000001,
  "ms_tie_break": "pair_B",

  "uv_on_edge_eps": 0.000000001,
  "hole_test_grid_n": 21,
  "hole_test_grid_min_inside_points": 1,

  "sort_round_rad_digits": 10,
  "sort_round_mm_digits": 6,
  "sort_round_uv_digits": 6,
  "sort_round_area_digits": 6,

  "percentile_method": "higher_order_stat",

  "csv_float_digits_mm": 6,
  "csv_float_digits_rad": 12,
  "csv_float_digits_unitless": 8
}
```

C.3 smoke 期待出力（ファイル一覧：最小）

smoke\_v1-1-1.json を入力し、output\_dir に生成される成果物は少なくとも以下を含む（必須成果物の最小集合）。

Master Truth（数値）：

  • tool\_conjugate\_grid\_plus.csv

  • tool\_conjugate\_grid\_minus.csv

  • cutting\_edge\_points.csv

  • flank\_uv\_boundary\_plus.csv

  • flank\_uv\_boundary\_minus.csv

  • sections\_{ZTAG}.dxf（z\_sections\_mm[] の各要素ぶん）

    • 上の例（z\_sections\_mm=[0.0], sort\_round\_mm\_digits=6）では：

      • sections\_z+00000.000000.dxf

JSON：

  • report.json

  • cad\_report\_step00.json ～ cad\_report\_step10.json（Stepを実装する場合）

  • manifest.json

Visual Pack（DoD対象）：

  • flank\_boundary\_wire\_plus.step / flank\_boundary\_wire\_minus.step

  • flank\_view\_mesh\_plus.stl / flank\_view\_mesh\_minus.stl

  • cutting\_edge\_tube\_plus.stl / cutting\_edge\_tube\_minus.stl（edges\_ok==true のとき）

  • tool\_visual\_one\_tooth.stl

  • tool\_visual\_preview\_ring.stl

C.4 期待出力の判定方法（凍結）

  • fixtures/.../smoke\_expected/ にある期待出力と、今回の output\_dir の出力を比較する。

  • 比較は「ファイルの sha256（生バイト列）一致」で行う（改行コードや桁違いも検出する）。

  • manifest.json の files[] は比較の起点にもなる（path, sha256, bytes を照合）。

  • created\_utc は 8.3.1 の固定値なので、ゴールデン比較で揺れない。

  • report.json の runtime（任意キー）は、ゴールデン比較では省略（推奨）または固定値を使用する（暗黙で現在時刻を書かない）。

