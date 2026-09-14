// 无头验证：从 index.html 抽出 <script id="engine">，在 Node vm 里跑断言
const fs = require('fs'), vm = require('vm'), path = require('path');
const dir = __dirname;
const html = fs.readFileSync(path.join(dir, 'index.html'), 'utf8');
const m = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
if (!m) { console.error('engine script not found'); process.exit(1); }
const ctx = { console };
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(m[1], ctx, { filename: 'engine.js' });
const M = ctx.MCMC;

let pass = 0, fail = 0; const fails = [];
function ok(cond, name, detail) { if (cond) pass++; else { fail++; fails.push(name + ' | ' + detail); } }
const close = (a, b, tol) => Math.abs(a - b) <= tol;

// ══════════ 1) 内置 8 条不变量 ══════════
M.selfTest().forEach((r, i) => ok(r.pass, 'selfTest#' + (i + 1) + ' ' + r.name, r.detail));

// ══════════ 2) 特殊函数（对照解析值） ══════════
{
  const cases = [
    ['lgamma(0.5)', M.lgamma(0.5), 0.5723649429247001],
    ['lgamma(1)', M.lgamma(1), 0],
    ['lgamma(5)', M.lgamma(5), 3.1780538303479458],
    ['lgamma(10)', M.lgamma(10), 12.801827480081469],
    ['lgamma(0.1)', M.lgamma(0.1), 2.252712651734206],
    ['normCdf(0)', M.normCdf(0), 0.5],
    ['normCdf(1)', M.normCdf(1), 0.8413447460685429],
    ['normCdf(1.96)', M.normCdf(1.96), 0.9750021048517796],
    ['normCdf(-3)', M.normCdf(-3), 0.001349898031630095],
    ['normCdf(4.5)', M.normCdf(4.5), 0.9999966023268753],
    ['gammap(3,3)', M.gammap(3, 3), 0.5768099188721559],
    ['gammap(1,2)', M.gammap(1, 2), 0.8646647167633873]
  ];
  cases.forEach(([nm, got, exp]) => ok(close(got, exp, 1e-11), '特殊函数 ' + nm, `${got} vs ${exp}`));
  // 恒等式 P + Q = 1
  let worst = 0;
  [[0.5, 0.3], [3, 1], [3, 8], [7.5, 6], [0.2, 0.05]].forEach(([a, x]) => {
    worst = Math.max(worst, Math.abs(M.gammap(a, x) + M.gammaq(a, x) - 1));
  });
  ok(worst < 1e-14, 'gammap + gammaq == 1', 'worst=' + worst);
  // Gamma(3,2) 的 CDF 在均值处，以及 CDF 单调不减
  let mono = true, prev = -1;
  for (let x = 0; x < 30; x += 0.25) { const v = M.gammap(3, x / 2); if (v < prev - 1e-15) mono = false; prev = v; }
  ok(mono, 'Gamma CDF 单调不减', '');
}

// ══════════ 3) 提议核 Jacobian（逐模式数值对照） ══════════
{
  // 'log' 模式：q(x|y)/q(y|x) 应 == y/x
  let w = 0;
  for (let t = 0; t < 200; t++) {
    const x = [0.1 + Math.random() * 20], y = [0.1 + Math.random() * 20];
    w = Math.max(w, Math.abs(M.logQratio(x, y, 'log', 1) - (Math.log(y[0]) - Math.log(x[0]))));
  }
  ok(w < 1e-12, 'log 提议 Jacobian == ln y − ln x', 'w=' + w);
  // 'logit' 模式：应 == ln[y(1−y)] − ln[x(1−x)]
  w = 0;
  for (let t = 0; t < 200; t++) {
    const x = [0.01 + Math.random() * 0.98], y = [0.01 + Math.random() * 0.98];
    const exp = Math.log(y[0]) + Math.log(1 - y[0]) - Math.log(x[0]) - Math.log(1 - x[0]);
    w = Math.max(w, Math.abs(M.logQratio(x, y, 'logit', 1) - exp));
  }
  ok(w < 1e-12, 'logit 提议 Jacobian 对照', 'w=' + w);
  // 'rw' 模式恒为 0
  ok(M.logQratio([1, 2], [3, 4], 'rw', 2) === 0, 'rw 提议对称（比值 0）', '');
  // 独立密度函数 vs 采样器比值（三维 log 模式）
  w = 0;
  for (let t = 0; t < 100; t++) {
    const x = [0.5 + Math.random() * 5, 1 + Math.random() * 5], y = [0.5 + Math.random() * 5, 1 + Math.random() * 5];
    const st = 0.7;
    const used = M.logQratio(x, y, 'log', 2);
    const truth = M.logProp(y, x, st, 'log', 2) - M.logProp(x, y, st, 'log', 2);
    w = Math.max(w, Math.abs(used - truth));
  }
  ok(w < 1e-12, '二维 log 提议：采样器比值 == 独立密度比', 'w=' + w);
}

// ══════════ 4) 随机压力：6 目标 × 5 seed ══════════
{
  const keys = ['normal', 'bimodal', 'gamma', 'logistic', 'corr', 'banana'];
  let nFinite = 0;
  for (const key of keys) {
    const tg = M.TARGETS[key];
    for (let s = 0; s < 5; s++) {
      const rng = M.mulberry32(1000 + s * 37 + key.length);
      const res = M.rwm(tg, 8000, { rng, step: tg.step });
      const xs = res.samples.map(v => v[0]);
      ok(res.samples.length === 8000, `${key}#${s} 样本数`, '' + res.samples.length);
      const allFin = res.samples.every(v => v.every(Number.isFinite));
      if (allFin) nFinite++;
      ok(allFin, `${key}#${s} 样本全部有限`, '');
      ok(res.acceptRate > 0 && res.acceptRate < 1, `${key}#${s} 接受率 ∈ (0,1)`, '' + res.acceptRate.toFixed(4));
      if (tg.kind === '1d') {
        const mu = M.mean(xs), e = M.ess(xs), se = tg.sd / Math.sqrt(Math.max(e, 1));
        const z = Math.abs(mu - tg.mean) / se;
        ok(z < 5, `${key}#${s} 均值 z 检验`, `z=${z.toFixed(2)} mu=${mu.toFixed(4)}`);
        const D = M.ksStat(xs, tg.cdf) * Math.sqrt(Math.max(e, 1));
        ok(D < 3.0, `${key}#${s} KS·√ESS`, '' + D.toFixed(2));
      } else if (key === 'banana') {
        // v = x₂ − x₁² 是横向「快方向」，RWM 在这一维混合良好 → 可严格断言
        const ev2 = M.mean(res.samples.map(v => { const t = v[1] - v[0] * v[0]; return t * t; }));
        ok(Math.abs(ev2 - 1) < 0.25, `${key}#${s} E[(x₂−x₁²)²]≈1（快方向）`, '' + ev2.toFixed(3));
        // x₁ 是沿弯月的「慢方向」，RWM 混合极慢（ESS≈150 / N=8000）→ 必须 ESS 校正后做 z 检验
        const q = xs.map(v => v * v), ex2 = M.mean(q), e = M.ess(xs);
        let vq = 0; for (const u of q) vq += (u - ex2) * (u - ex2);
        const sdq = Math.sqrt(vq / q.length);
        const z = Math.abs(ex2 - 1) / (sdq / Math.sqrt(Math.max(e, 1)));
        ok(z < 6, `${key}#${s} E[x₁²] z 检验（慢方向）`, `z=${z.toFixed(2)} v=${ex2.toFixed(3)} ess=${Math.round(e)}`);
      } else {
        const exy = M.mean(res.samples.map(v => v[0] * v[1]));
        ok(Math.abs(exy - 0.95) < 0.2, `${key}#${s} E[x₁x₂]≈0.95`, '' + exy.toFixed(3));
      }
    }
  }
  ok(nFinite === 30, '全部 30 组样本有限', '' + nFinite);
}

// ══════════ 5) HMC 与 RWM 交叉验证（两条独立路径，同一目标） ══════════
{
  ['normal', 'logistic', 'gamma'].forEach(key => {
    const tg = M.TARGETS[key];
    const a = M.rwm(tg, 20000, { rng: M.mulberry32(31), step: tg.step }).samples.map(v => v[0]);
    const b = M.hmc(tg, 20000, { rng: M.mulberry32(32), eps: key === 'gamma' ? 0.3 : 0.35, L: 12 }).samples.map(v => v[0]);
    const ea = M.ess(a), eb = M.ess(b);
    const ma = M.mean(a), mb = M.mean(b);
    const se = Math.sqrt(tg.sd * tg.sd / Math.max(ea, 1) + tg.sd * tg.sd / Math.max(eb, 1));
    const z = Math.abs(ma - mb) / se;
    ok(z < 4, `RWM vs HMC 均值一致 #${key}`, `z=${z.toFixed(2)} ${ma.toFixed(4)} vs ${mb.toFixed(4)}`);
    ok(Math.abs(ma - tg.mean) < 0.2 * tg.sd, `HMC 均值接近解析 #${key}`, '' + ma.toFixed(4));
    const D = M.ksStat(b, tg.cdf) * Math.sqrt(Math.max(eb, 1));
    ok(D < 3.0, `HMC KS·√ESS #${key}`, '' + D.toFixed(2));
  });
}

// ══════════ 6) 边界条件 ══════════
{
  const tg = M.TARGETS.normal;
  const r1 = M.rwm(tg, 1, { rng: M.mulberry32(1), step: 2.4 });
  ok(r1.samples.length === 1 && Number.isFinite(r1.samples[0][0]), 'N=1 可运行', '');
  const r2 = M.rwm(tg, 3000, { rng: M.mulberry32(2), step: 1e-7 });
  ok(r2.acceptRate > 0.99, '步长→0：几乎全接受', '' + r2.acceptRate.toFixed(4));
  const r3 = M.rwm(tg, 3000, { rng: M.mulberry32(3), step: 500 });
  ok(r3.acceptRate < 0.02, '步长→∞：几乎全拒绝', '' + r3.acceptRate.toFixed(4));
  // Gamma 从极小值起步（log 提议不会越界）
  const g = M.TARGETS.gamma;
  const r4 = M.rwm(g, 6000, { rng: M.mulberry32(4), step: 0.9, x0: [1e-4] });
  ok(r4.samples.every(v => v[0] > 0), 'Gamma 采样恒为正（log 提议）', '');
  const mu = M.mean(r4.samples.map(v => v[0])), e = M.ess(r4.samples.map(v => v[0]));
  ok(Math.abs(mu - 6) < 5 * g.sd / Math.sqrt(Math.max(e, 1)), 'Gamma 从 1e-4 起步仍能收敛到均值 6', '' + mu.toFixed(3));
  // HMC 极端参数
  const h1 = M.hmc(tg, 2000, { rng: M.mulberry32(5), eps: 1.2, L: 3 });
  ok(h1.samples.every(v => Number.isFinite(v[0])), 'HMC 大 ε 不产生 NaN', '');
  // ΔH 随 ε 急剧增大（leapfrog 二阶精度的直接后果）
  function meanAbsDH(eps, L) {
    const r = M.hmc(tg, 2000, { rng: M.mulberry32(5), eps, L });
    let s = 0; for (const d of r.dH) s += Math.abs(d);
    return s / r.dH.length;
  }
  const dhSmall = meanAbsDH(0.05, 3), dhBig = meanAbsDH(1.5, 3);
  ok(dhBig / Math.max(dhSmall, 1e-300) > 100, 'HMC：ε 0.05→1.5，E|ΔH| 增大 >100×',
    `${dhSmall.toExponential(2)} → ${dhBig.toExponential(2)}`);
  // ε > 2 超出 leapfrog 稳定域（谐振子）：必然发散、几乎全拒
  const h3 = M.hmc(tg, 1000, { rng: M.mulberry32(5), eps: 2.5, L: 3 });
  ok(h3.samples.every(v => Number.isFinite(v[0])), 'HMC ε>2（超稳定域）不产生 NaN', '');
  ok(h3.acceptRate < 0.2, 'HMC ε>2 几乎全拒绝', '' + h3.acceptRate.toFixed(3));
  const h2 = M.hmc(tg, 2000, { rng: M.mulberry32(6), eps: 0.01, L: 1 });
  ok(h2.acceptRate > 0.99, 'HMC 小 ε 几乎全接受', '' + h2.acceptRate.toFixed(3));
  // 2D 目标 HMC
  const b = M.hmc(M.TARGETS.banana, 6000, { rng: M.mulberry32(7), eps: 0.12, L: 25 });
  const ex2 = M.mean(b.samples.map(v => v[0] * v[0]));
  ok(Math.abs(ex2 - 1) < 0.2, 'HMC on banana: E[x₁²]≈1', '' + ex2.toFixed(3));
  const c = M.hmc(M.TARGETS.corr, 6000, { rng: M.mulberry32(8), eps: 0.3, L: 15 });
  const exy = M.mean(c.samples.map(v => v[0] * v[1]));
  ok(Math.abs(exy - 0.95) < 0.2, 'HMC on corr: E[x₁x₂]≈0.95', '' + exy.toFixed(3));
}

// ══════════ 7) Ising：Gibbs vs 精确枚举（多 β） ══════════
{
  [0.2, 0.3, 0.4].forEach(beta => {
    const ex = M.isingExact(4, beta);
    const gb = M.gibbsIsing(4, beta, 100000, M.mulberry32(400 + beta * 1000), { init: 'hot', record: 5 });
    const gAbsM = M.mean(gb.mHist.map(Math.abs)), gE = M.mean(gb.eHist);
    const relM = Math.abs(gAbsM - ex.absM / 16) / (ex.absM / 16);
    const relE = Math.abs(gE - ex.E / 16) / Math.abs(ex.E / 16);
    ok(relM < 0.06, `Ising β=${beta} ⟨|m|⟩ 相对误差`, (relM * 100).toFixed(2) + '%');
    ok(relE < 0.04, `Ising β=${beta} ⟨E⟩/N 相对误差`, (relE * 100).toFixed(2) + '%');
  });
  // β=0 时每个构型等概率 → Z == 2^16
  {
    const z0 = M.isingExact(4, 0);
    ok(close(z0.Z, 65536, 1e-6), 'Ising β=0：Z == 2^16', '' + z0.Z);
  }
  // β=0 时每个构型等概率 → ⟨|m|⟩ = E[|M|]/N，解析 = C(16,8)/2^16 * ... 用二项分布算
  {
    const N = 16;
    let expectAbsM = 0;
    for (let k = 0; k <= N; k++) {
      let lgc = 0; // ln C(N,k)
      lgc = M.lgamma(N + 1) - M.lgamma(k + 1) - M.lgamma(N - k + 1);
      expectAbsM += Math.exp(lgc - N * Math.LN2) * Math.abs(2 * k - N) / N;
    }
    const ex0 = M.isingExact(4, 0);
    ok(close(ex0.absM / 16, expectAbsM, 1e-12), 'Ising β=0：⟨|m|⟩ == 二项分布解析值',
      `${(ex0.absM / 16).toFixed(10)} vs ${expectAbsM.toFixed(10)}`);
  }
  // Gray-code 枚举的自洽性：全 +1 能量 = −2N
  {
    const nb = M.buildNb(4), s = new Int8Array(16).fill(1);
    ok(M.isingEnergy(s, nb) === -32, 'Ising 全 +1 能量 == −2N = −32', '' + M.isingEnergy(s, nb));
  }
}

// ══════════ 8) 确定性 + 诊断工具 ══════════
{
  const a = M.rwm(M.TARGETS.normal, 500, { rng: M.mulberry32(9), step: 2.4 });
  const b = M.rwm(M.TARGETS.normal, 500, { rng: M.mulberry32(9), step: 2.4 });
  ok(a.samples.every((v, i) => v[0] === b.samples[i][0]), '同 seed RWM 确定性', '');
  const c = M.hmc(M.TARGETS.banana, 300, { rng: M.mulberry32(10), eps: 0.12, L: 25 });
  const d = M.hmc(M.TARGETS.banana, 300, { rng: M.mulberry32(10), eps: 0.12, L: 25 });
  ok(c.samples.every((v, i) => v[0] === d.samples[i][0]), '同 seed HMC 确定性', '');
  const g1 = M.gibbsIsing(4, 0.4, 2000, M.mulberry32(11), { init: 'hot', record: 5 });
  const g2 = M.gibbsIsing(4, 0.4, 2000, M.mulberry32(11), { init: 'hot', record: 5 });
  ok(g1.mHist.every((v, i) => v === g2.mHist[i]), '同 seed Gibbs 确定性', '');
  // ESS：iid 样本 ESS ≈ N
  const rng = M.mulberry32(12), iid = [];
  for (let i = 0; i < 5000; i++) iid.push(M.rnorm(rng));
  const eIid = M.ess(iid);
  ok(eIid > 4000 && eIid <= 5000, 'iid 样本 ESS ≈ N', '' + eIid.toFixed(0));
  // ESS：常数序列退化为 0
  ok(M.ess(new Array(100).fill(3)) === 0, '常数序列 ESS == 0', '');
  // ACF(0 滞后附近) 单调衰减
  const rw = M.rwm(M.TARGETS.normal, 20000, { rng: M.mulberry32(13), step: 2.4 }).samples.map(v => v[0]);
  const ac = M.acf(rw, 30);
  ok(ac[0] > 0.5 && ac[0] <= 1.0001, 'ACF(lag1) 接近 1（强自相关）', '' + ac[0].toFixed(3));
  ok(ac[29] < ac[0], 'ACF 随 lag 衰减', `${ac[0].toFixed(3)} → ${ac[29].toFixed(3)}`);
  // Gelman-Rubin：同分布的两条链 R̂ ≈ 1
  const ch = [iid.slice(0, 2000), iid.slice(2000, 4000), iid.slice(0, 2000).map(x => x + 1e-9)];
  const rh = M.gelmanRubin(ch);
  ok(rh < 1.01, '同分布多链 R̂ < 1.01', '' + rh.toFixed(4));
  // 均值差距大的链 R̂ 明显 > 1
  const ch2 = [iid.slice(0, 2000), iid.slice(2000, 4000).map(x => x + 3)];
  ok(M.gelmanRubin(ch2) > 1.5, '均值偏移 3σ 的链 R̂ >> 1', '' + M.gelmanRubin(ch2).toFixed(2));
}

fs.writeFileSync(path.join(dir, '_smoke.log'),
  `PASS ${pass} / ${pass + fail}\n` + (fail ? 'FAIL:\n' + fails.join('\n') : 'ALL GREEN') + '\n');
console.log(`PASS ${pass} / ${pass + fail}`);
if (fail) { console.log('FAIL:\n' + fails.join('\n')); process.exit(1); }
console.log('ALL GREEN');
