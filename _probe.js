// 探针：把内部状态 dump 成 ASCII，供人眼复核（断言全绿 ≠ 正确）
const fs = require('fs'), vm = require('vm'), path = require('path');
const dir = __dirname;
const html = fs.readFileSync(path.join(dir, 'index.html'), 'utf8');
const m = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
const ctx = { console }; ctx.globalThis = ctx; vm.createContext(ctx);
vm.runInContext(m[1], ctx, { filename: 'engine.js' });
const M = ctx.MCMC;

const L = [];
const say = s => L.push(s);
const f = (v, n) => (v === undefined || v === null) ? '—' : v.toFixed(n === undefined ? 6 : n);
const pad = (s, n) => String(s) + ' '.repeat(Math.max(0, n - String(s).length));

/* ── 1. 特殊函数 ── */
say('=== 特殊函数（对照解析值）===');
[['lgamma(0.5)', M.lgamma(0.5), 0.5723649429247001],
 ['lgamma(5)', M.lgamma(5), 3.1780538303479458],
 ['lgamma(10)', M.lgamma(10), 12.801827480081469],
 ['normCdf(1)', M.normCdf(1), 0.8413447460685429],
 ['normCdf(1.96)', M.normCdf(1.96), 0.9750021048517796],
 ['normCdf(-3)', M.normCdf(-3), 0.001349898031630095],
 ['gammap(3,3)', M.gammap(3, 3), 0.5768099188721559],
 ['gammap(1,2)', M.gammap(1, 2), 0.8646647167633873]].forEach(([nm, got, exp]) => {
  say('  ' + pad(nm, 16) + pad(f(got, 12), 18) + '解析 ' + pad(f(exp, 12), 18) + '误差 ' + Math.abs(got - exp).toExponential(2));
});

/* ── 2. 1D 目标：矩 + KS ── */
say('');
say('=== 1D 目标：RWM N=40000，矩与解析值对拍（ESS 校正标准误）===');
say('  ' + pad('目标', 12) + pad('E[x] MC', 12) + pad('E[x] 解析', 12) + pad('sd MC', 10) + pad('sd 解析', 10) +
    pad('z', 8) + pad('接受率', 8) + pad('ESS', 8) + 'KS·√ESS');
['normal', 'bimodal', 'gamma', 'logistic'].forEach(key => {
  const tg = M.TARGETS[key];
  const res = M.rwm(tg, 40000, { rng: M.mulberry32(4242 + key.length), step: tg.step });
  const xs = res.samples.map(v => v[0]);
  const mu = M.mean(xs), e = M.ess(xs);
  let v2 = 0; for (const v of xs) v2 += (v - mu) * (v - mu);
  const sd = Math.sqrt(v2 / xs.length);
  const z = Math.abs(mu - tg.mean) / (tg.sd / Math.sqrt(Math.max(e, 1)));
  const ks = M.ksStat(xs, tg.cdf) * Math.sqrt(Math.max(e, 1));
  say('  ' + pad(key, 12) + pad(f(mu, 4), 12) + pad(tg.mean, 12) + pad(f(sd, 4), 10) + pad(f(tg.sd, 4), 10) +
      pad(f(z, 2), 8) + pad(f(res.acceptRate, 3), 8) + pad(Math.round(e), 8) + f(ks, 2));
});
say('  （KS·√ESS 的 99% 临界值 = 1.63；iid 样本期望 ≈ 0.8）');

/* ── 3. RWM vs HMC 两条独立路径 ── */
say('');
say('=== 同一目标：RWM 与 HMC 两条独立路径对拍（N=20000）===');
['normal', 'logistic', 'gamma'].forEach(key => {
  const tg = M.TARGETS[key];
  const a = M.rwm(tg, 20000, { rng: M.mulberry32(31), step: tg.step });
  const b = M.hmc(tg, 20000, { rng: M.mulberry32(32), eps: key === 'gamma' ? 0.3 : 0.35, L: 12 });
  const ma = M.mean(a.samples.map(v => v[0])), mb = M.mean(b.samples.map(v => v[0]));
  const ea = M.ess(a.samples.map(v => v[0])), eb = M.ess(b.samples.map(v => v[0]));
  say('  ' + pad(key, 10) + 'RWM μ=' + pad(f(ma, 4), 9) + 'ESS=' + pad(Math.round(ea), 7) +
      ' | HMC μ=' + pad(f(mb, 4), 9) + 'ESS=' + pad(Math.round(eb), 7) +
      ' | 解析 μ=' + pad(tg.mean, 6) + ' | HMC/RWM ESS = ' + f(eb / Math.max(ea, 1), 1) + '×');
});

/* ── 4. HMC 可逆性与 ΔH 标度 ── */
say('');
say('=== HMC：leapfrog 可逆性 + ΔH ∝ ε² ===');
{
  const tb = M.TARGETS.banana, q0 = [0.5, 0.3], p0 = [0.2, -0.4];
  const f1 = M.leapfrog(tb, q0, p0, 0.1, 25);
  const f2 = M.leapfrog(tb, f1.q, [-f1.p[0], -f1.p[1]], 0.1, 25);
  say('  起点 q=(' + q0 + ') p=(' + p0 + ')');
  say('  正向 L=25 后 q=(' + f1.q.map(v => f(v, 6)).join(', ') + ')');
  say('  动量取负再跑 25 步 → q=(' + f2.q.map(v => f(v, 6)).join(', ') + ')  p=(' + f2.p.map(v => f(v, 6)).join(', ') + ')');
  say('  期望 p=(-0.200000, 0.400000)');
  say('  复原误差 max|q−q₀| = ' + Math.max(Math.abs(f2.q[0] - q0[0]), Math.abs(f2.q[1] - q0[1])).toExponential(2) +
      '   max|p+p₀| = ' + Math.max(Math.abs(f2.p[0] + p0[0]), Math.abs(f2.p[1] + p0[1])).toExponential(2));
  say('  ε        L     E|ΔH|      接受率   轨迹发散率');
  [0.4, 0.2, 0.1, 0.05].forEach(eps => {
    const r = M.hmc(tb, 1500, { rng: M.mulberry32(999), eps, L: 25 });
    let s = 0, c = 0, div = 0;
    // |ΔH| ≥ 100 视为发散轨迹（banana 的 leapfrog 超出稳定域），单独统计、不并入均值
    for (const d of r.dH) { if (isFinite(d) && Math.abs(d) < 100) { s += Math.abs(d); c++; } else div++; }
    say('  ' + pad(eps, 8) + pad(25, 6) + pad(f(s / Math.max(c, 1), 8), 12) + pad(f(r.acceptRate, 3), 10) +
        (100 * div / r.dH.length).toFixed(1) + '%');
  });
  say('  （ε=0.4 时 banana 的 leapfrog 超出稳定域：9% 的轨迹能量爆炸 |ΔH|>100 甚至 +∞，全部被拒绝；');
  say('   这列的 E|ΔH| 只统计未发散的轨迹）');
  say('  （ε 每减半，E|ΔH| 应降约 4× —— 这是 leapfrog 二阶精度的直接证据）');
  {
    const tn = M.TARGETS.normal;
    const rows = [0.05, 0.2, 1.0, 1.5, 2.5].map(eps => {
      const r = M.hmc(tn, 3000, { rng: M.mulberry32(5), eps, L: 3 });
      let s = 0; for (const d of r.dH) s += Math.abs(d);
      return [eps, s / r.dH.length, r.acceptRate];
    });
    say('  附注（1D 标准正态 = 谐振子，L=3）：');
    rows.forEach(([eps, dh, ar]) => say('    ε=' + pad(eps, 5) + ' E|ΔH|=' + pad(dh.toExponential(3), 12) + ' 接受率=' + f(ar, 4)));
    say('    ε=1.0 时 E|ΔH| 精确为 0：leapfrog 对谐振子的总相位 = L·2·asin(ε/2) = π，');
    say('    恰好是半周期（q→−q, p→−p），能量守恒到机器精度 —— 不是 bug。ε>2 才真正发散。');
  }
}

/* ── 5. Ising：Gibbs vs 精确枚举 ── */
say('');
say('=== Ising 4×4（周期边界 J=1）：Gibbs 100k 步 vs Gray-code 枚举 2¹⁶ ===');
say('  ' + pad('β', 6) + pad('⟨|m|⟩ Gibbs', 14) + pad('⟨|m|⟩ 精确', 14) + pad('误差', 9) +
    pad('⟨E⟩/N Gibbs', 14) + pad('⟨E⟩/N 精确', 14) + pad('误差', 9) + 'Cᵥ');
[0.2, 0.3, 0.4].forEach(beta => {
  const ex = M.isingExact(4, beta);
  const gb = M.gibbsIsing(4, beta, 100000, M.mulberry32(400 + beta * 1000), { init: 'hot', record: 5 });
  const gAbsM = M.mean(gb.mHist.map(Math.abs)), gE = M.mean(gb.eHist);
  const rm = Math.abs(gAbsM - ex.absM / 16) / (ex.absM / 16), re = Math.abs(gE - ex.E / 16) / Math.abs(ex.E / 16);
  say('  ' + pad(beta, 6) + pad(f(gAbsM, 5), 14) + pad(f(ex.absM / 16, 5), 14) + pad(f(rm * 100, 2) + '%', 9) +
      pad(f(gE, 5), 14) + pad(f(ex.E / 16, 5), 14) + pad(f(re * 100, 2) + '%', 9) + f(ex.Cv, 4));
});
{
  const z0 = M.isingExact(4, 0);
  let expAbsM = 0;
  for (let k = 0; k <= 16; k++) {
    const lgc = M.lgamma(17) - M.lgamma(k + 1) - M.lgamma(17 - k);
    expAbsM += Math.exp(lgc - 16 * Math.LN2) * Math.abs(2 * k - 16) / 16;
  }
  say('  自洽检查：β=0 时 Z=' + z0.Z + '（应 65536），⟨|m|⟩=' + f(z0.absM / 16, 8) +
      ' vs 二项分布解析 ' + f(expAbsM, 8));
}

/* ── 6. 最优接受率扫描 ── */
say('');
say('=== RWM 最优尺度（d=10 标准高斯，N=8000，理论最优接受率 0.234）===');
{
  const tg = M.ndGaussian(10);
  say('  ' + pad('σ', 8) + pad('接受率', 10) + pad('ESS(x₁)', 10) + 'ESS/N');
  let best = null;
  [0.2, 0.35, 0.5, 0.6, 0.75, 0.9, 1.0, 1.4, 2.0].forEach(st => {
    const r = M.rwm(tg, 8000, { rng: M.mulberry32(555), step: st });
    const e = M.ess(r.samples.map(v => v[0]));
    if (!best || e > best.e) best = { st, a: r.acceptRate, e };
    say('  ' + pad(st, 8) + pad(f(r.acceptRate, 4), 10) + pad(Math.round(e), 10) + f(e / 8000, 3));
  });
  say('  → ESS 峰值 σ*=' + best.st + '，对应接受率 ' + f(best.a, 3) + '（理论 0.234；2.38/√10 = 0.752）');
}

/* ── 7. banana：HMC vs RWM ── */
say('');
say('=== Banana（y|x ~ N(x²,1)）：HMC 与 RWM 的混合效率 ===');
{
  const tb = M.TARGETS.banana;
  const rw = M.rwm(tb, 4000, { rng: M.mulberry32(8), step: 0.5 });
  const hm = M.hmc(tb, 4000, { rng: M.mulberry32(8), eps: 0.12, L: 25 });
  const eR = M.ess(rw.samples.map(v => v[0])), eH = M.ess(hm.samples.map(v => v[0]));
  const acR = M.acf(rw.samples.map(v => v[0]), 25), acH = M.acf(hm.samples.map(v => v[0]), 25);
  say('  RWM  ESS=' + pad(Math.round(eR), 6) + '  接受率=' + pad(f(rw.acceptRate, 3), 7) + '  E[x₁²]=' +
      f(M.mean(rw.samples.map(v => v[0] * v[0])), 3) + '（解析 1）');
  say('  HMC  ESS=' + pad(Math.round(eH), 6) + '  接受率=' + pad(f(hm.acceptRate, 3), 7) + '  E[x₁²]=' +
      f(M.mean(hm.samples.map(v => v[0] * v[0])), 3) + '（解析 1）');
  say('  ACF lag1 / lag10 / lag25：RWM ' + [0, 9, 24].map(i => f(acR[i], 3)).join(' / ') +
      '   HMC ' + [0, 9, 24].map(i => f(acH[i], 3)).join(' / '));
  say('  → HMC / RWM 有效样本量 = ' + f(eH / Math.max(eR, 1), 1) + '×');
}

/* ── 8. Gelman-Rubin ── */
say('');
say('=== Gelman-Rubin R̂：4 条链从 (±4,±4) 过分散初值出发（2D 相关高斯）===');
{
  const tc = M.TARGETS.corr, inits = [[-4, -4], [4, 4], [-4, 4], [4, -4]], seeds = [11, 22, 33, 44];
  const chains = seeds.map((s, i) => M.rwm(tc, 2500, { rng: M.mulberry32(s), step: 0.6, x0: inits[i] })
    .samples.map(v => v[0]));
  say('  ' + pad('窗口', 16) + pad('R̂', 10) + '各链均值');
  [[0, 120], [0, 300], [0, 800], [500, 2500], [1500, 2500]].forEach(([a, b]) => {
    const sub = chains.map(c => c.slice(a, b));
    say('  ' + pad(`[${a}, ${b})`, 16) + pad(f(M.gelmanRubin(sub), 4), 10) +
        sub.map(c => f(M.mean(c), 3)).join('  '));
  });
  say('  （R̂ 从远大于 1 收敛到 < 1.01；若初值已收敛，R̂ 会一开始就近 1，指标就失去区分度）');
}

/* ── 9. Jacobian 漏项对照 ── */
say('');
say('=== 对照实验：故意去掉 log 提议的 Jacobian 修正会怎样（目标 Gamma(3,2)）===');
{
  const tg = M.TARGETS.gamma;
  // 好：正常 MH（含 Jacobian）
  const good = M.rwm(tg, 40000, { rng: M.mulberry32(66), step: 0.9 }).samples.map(v => v[0]);
  // 坏：同一个提议，但接受率里把 log q 比值当成 0（对称提议的错误假设）
  const rng = M.mulberry32(66);
  let x = 6, lpx = tg.logpi([x]);
  const bad = [];
  for (let i = 0; i < 40000; i++) {
    const y = [x * Math.exp(0.9 * M.rnorm(rng))];
    const lpy = tg.logpi(y);
    const u = rng(); if (u < 1e-300) u = 1e-300;
    if (Math.log(u) < lpy - lpx) { x = y[0]; lpx = lpy; }
    bad.push(x);
  }
  const mg = M.mean(good), mb = M.mean(bad);
  const kg = M.ksStat(good, tg.cdf), kb = M.ksStat(bad, tg.cdf);
  say('  ' + pad('', 20) + pad('E[x]', 12) + pad('KS 统计', 12) + '判定');
  say('  ' + pad('含 Jacobian（正确）', 20) + pad(f(mg, 4), 12) + pad(f(kg, 5), 12) +
      (Math.abs(mg - 6) < 0.3 ? '✓ 收敛到解析均值 6' : '✗'));
  say('  ' + pad('漏 Jacobian（错误）', 20) + pad(f(mb, 4), 12) + pad(f(kb, 5), 12) +
      (Math.abs(mb - 4) < 0.3 ? '✓ 收敛到 4 = 理论预测' : '✗ 分布系统性偏移'));
  say('  → 漏掉 Jacobian 后，MH 的平稳分布从 π(x) 变成 π(x)/x：');
  say('    Gamma(3,2) ∝ x²e^(−x/2) → Gamma(2,2) ∝ x·e^(−x/2)，解析均值 4，与实测 ' + f(mb, 3) + ' 吻合');
}

fs.writeFileSync(path.join(dir, '_probe.txt'), L.join('\n') + '\n');
console.log(L.join('\n'));
