### 3. 使用蒙特卡洛方法计算域的面积
**任务描述。** 三个域 $(D_1, D_2, D_3)$ 由它们的边界方程所定义。需要计算的域D 的面积是由 $(D_1, D_2, D_3)$ 的并集定义的，即 $(D = D_1 ∪ D_2 ∪ D_3)$。

需要计算域 $D$ 的面积，精确到2-3%。每个子域 $D_i$ 的边界是由下面的方程定义的超越曲线：
$$ |x - x_i|^{p_i} + |y - y_i|^{p_i} = c_i, $$
其中 $p_i > 1, c_i > 0$。

$i = 1,2,3$

（因此，一个域的方程是
$ |x - x_i|^{p_i} + |y - y_i|^{p_i} \leq c_i ）$。

很明显，每个域都属于这样一个矩形（实际上是正方形），形式为

$ [x_i - c_i^{1/p_i}, x_i + c_i^{1/p_i}] \times [y_i - c_i^{1/p_i}, y_i + c_i^{1/p_i}] $：

$$ D_i \subseteq [x_i - c_i^{1/p_i}, x_i + c_i^{1/p_i}] \times [y_i - c_i^{1/p_i}, y_i + c_i^{1/p_i}] $$
$$(i = 1,2,3)$$

所以，对于域 $D$ 有：

$$ D \subseteq R = [\min\{x_1 - c_1^{1/p_1}, x_2 - c_2^{1/p_2}, x_3 - c_3^{1/p_3}\}, \max{x_1 + c_1^{1/p_1}, x_2+ c_2^{1/p_2}, x_3 + c_3^{1/p_3}}] \times [\min{y_1- c_1^{1/p_1}, y_2 - c_2^{1/p_2}, y_3 - c_3^{1/p_3}}, \max{y_1 + c_1^{1/p_1}, y_2 + c_2^{1/p_2}, y_3 + c_3^{1/p_3}}] $$

让 $S$) 表示矩形 $R$ 的面积，$S_0$ 为域 $D$ 的面积。设 $\xi = (\xi_1, \xi_2)$ 是一个在 $R$ 上均匀分布的随机向量变量。定义随机变量 $\eta$ 如下：

$$
\eta = 
\begin{cases}
1, &  \xi \in D \\
0, &  \xi \notin D
\end{cases}
$$

然后 $E[\eta] = S_0 / S$ 并且

$\overline{\eta}_N = $

$
 \frac{1}{N} \sum_{k=1}^{N} \eta_k \to E[\eta] = \frac{S_0}{S} \text{（in probability）}
$

其中 $ \{\eta_k\} $ 是具有分布 $\eta $ 的独立同分布随机变量。实际上，实现样本是获得的，并且样本值 $ \hat{\eta}_N = m / N $ 是计算的，其中 $m$ 是单位实现的数量 $ (\eta_k = 1) $，$ N $ 是样本大小。因此，

$$
m / N \approx S_0 / S, \text{ or } S_0 \approx S \cdot m / N
$$

精度可以用中心极限定理来估计。随机变量 $\eta$ 有伯努利分布，设 $ E[\eta] = p  ( 0 < p < 1 )$，所以

$$
\Pr \left( \left| p - \hat{\eta}_N \right| < \frac{x}{\sqrt{N}} \cdot D[\eta] \right) \approx \frac{2}{\sqrt{2\pi}} \int_{0}^{x} e^{-t^2/2} dt
$$


$
\left( \left| p - \hat{\eta}_N \right| < \frac{x}{\sqrt{N}} \cdot D[\eta] \right)
$

$
\approx \frac{2}{\sqrt{2\pi}} \int_{0}^{x} e^{-t^2/2} dt
$

