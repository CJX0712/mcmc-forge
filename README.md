# mcmc-forge · 马尔可夫链蒙特卡洛实验室

零依赖单文件 HTML。手写实现三个 MCMC 采样器，每条正确性都由**可独立复算的硬不变量**钉死——
包括拿 **Gray-code 枚举全部 2¹⁶ 个 Ising 构型**算出的精确配分函数去对拍 Gibbs 采样。

打开 `index.html` 即用，无需构建、无需联网。

## 采样器

| 采样器 | 关键实现 |
|---|---|
| **Random-Walk Metropolis** | 三种提议核：`rw`（对称高斯）、`log`（正变量，对数空间）、`logit`（(0,1) 变量）。后两者**非对称**，接受率里必须带 Jacobian 项 `log q(x\|y) − log q(y\|x)` |
| **Hamiltonian MC** | leapfrog 辛积分（半步动量 → 全步位置 → 半步动量），动量重采样，Metropolis 校正 `min(1, e^{−ΔH})` |
| **Gibbs（Ising）** | 随机扫描，条件概率 `P(sᵢ=+1) = σ(2β Σ_nn sⱼ)`；能量/磁化强度增量 O(1) 更新 |

## 8 条不变量

页面底部「运行自检」即时执行，无头环境 `node _smoke.js` 复跑。

| # | 不变量 | 实测 |
|---|---|---|
| ① | 非对称提议的 Jacobian 修正 == 独立提议密度函数算出的比值；细致平衡 `π(x)q(y\|x)α(x,y) == π(y)q(x\|y)α(y,x)` | max Δ = 0（三种提议核） |
| ② | MC 均值 == 解析均值（ESS 校正后 \|z\| < 4） | normal 0.79 / bimodal 1.01 / gamma 0.40 / logistic 0.01 |
| ③ | leapfrog 可逆（动量取负往返复原）+ 能量误差 ΔH ∝ ε² | 复原误差 5.55e-17；ε 减半 ΔH 降 4.1× |
| ④ | Ising Gibbs == 2¹⁶ 构型精确枚举（β=0.4） | ⟨\|m\|⟩ 0.76992 vs 0.76471（0.68%）；⟨E⟩/N −1.3887 vs −1.3791（0.69%） |
| ⑤ | KS 统计量 × √ESS < 2.0（99% 临界 1.63） | 0.70 / 0.63 / 1.17 / 0.47 |
| ⑥ | RWM 最优尺度处接受率 ≈ 0.234（d=10，ESS 峰值） | σ\*=1.0 → 0.149（理论 2.38/√10 = 0.752） |
| ⑦ | HMC 有效样本量 >> RWM（banana，同 N） | ESS 1563 vs 69（**22.5×**） |
| ⑧ | Gelman-Rubin R̂：早期 > 1.05 且收敛后 < 1.02 | 1.1399 → 1.0055 |

## 探针实测（`node _probe.js` → `_probe.txt`）

断言全绿不等于结果对，所以探针把内部状态 dump 出来人眼复核：

- **特殊函数**：`lgamma`、`normCdf`、`gammap` 全部对照解析值，误差 **1e-16 ~ 1e-12**
- **两条独立路径对拍**：同一目标分别用 RWM 与 HMC 采样，均值互相印证且都逼近解析值
- **Ising 自洽**：β=0 时 Z == 65536，⟨\|m\|⟩ == 二项分布解析值 **0.19638062**（8 位全同）
- **对照实验**：故意去掉 log 提议的 Jacobian → 平稳分布从 `π(x)` 变成 `π(x)/x`，
  即 Gamma(3,2) ∝ x²e^(−x/2) 退化成 Gamma(2,2) ∝ x·e^(−x/2)，解析均值 4，**实测 3.9566**

## 用法

```bash
# 浏览器
open index.html

# 无头验证（211 条断言）
node _smoke.js

# 探针 dump
node _probe.js && cat _probe.txt
```

引擎挂在 `globalThis.MCMC`，无 DOM 依赖，可在 Node 里直接调用：

```js
const M = require('./engine');            // 或按 _smoke.js 的方式从 HTML 抽取
const s = M.rwm(M.TARGETS.gamma, 20000, { rng: M.mulberry32(1), step: 0.9 });
M.mean(s.samples.map(v => v[0]));          // ≈ 6
M.isingExact(4, 0.4).absM / 16;            // 精确 ⟨|m|⟩，非采样
```

## 已知限制

- **RWM 在 banana 上混合极慢**：x₁ 是沿弯月的慢方向，ESS ≈ 150 / N=8000。
  横向组合 `x₂ − x₁²` ~ N(0,1) 才是快方向，实测始终稳在 1.00。要估 `E[x₁²]` 请用 HMC。
- **HMC 在 Gamma 上优势不大**：`U = −2ln x + x/2` 在 x→0 处梯度爆炸，ε 必须很小，
  实测 HMC/RWM 的 ESS 比只有 1.2×。
- **ε > 2 会超出 leapfrog 稳定域**（谐振子情形）：轨迹能量爆炸，发散轨迹被判定为 ΔH = +∞ 全部拒绝，
  数值上不产生 NaN。
- 1D 标准正态上 ε=1、L=3 时 `E|ΔH|` 精确为 0 —— leapfrog 总相位 `L·2·asin(ε/2) = π`，
  恰好半周期，能量守恒到机器精度。**这是正确的物理，不是 bug**。

## 文件

```
index.html    单文件应用（内联 CSS + engine + UI 三段，零外部依赖）
_smoke.js     无头断言（Node vm 抽取 engine）
_probe.js     内部状态 ASCII dump
```

## License

MIT © 晨星
