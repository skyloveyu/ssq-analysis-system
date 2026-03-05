/**
 * ============================================================
 *  双色球多维度分析引擎 (analyzer.js)
 * ============================================================
 *  功能：
 *  1. 多周期冷热频次统计 (Sliding Windows)
 *  2. 遗漏值与均值回归指标 (Mean Reversion)
 *  3. 波动节奏与稳定性 (Volatility & Standard Deviation)
 *  4. 综合多策略打分 (Scoring System)
 *
 *  设计原则：一次遍历完成基础数据收集，后续计算基于索引数组
 * ============================================================
 */
'use strict';

// ==================== 工具函数 ====================

/** 计算数组标准差 */
function calcStdDev(arr) {
  if (arr.length <= 1) return 0;
  const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
  const variance = arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length;
  return Math.sqrt(variance);
}

/** Min-Max 归一化到 0-100 */
function normalize(values) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (max === min) return values.map(() => 50);
  return values.map(v => Math.round(((v - min) / (max - min)) * 100));
}

/** 四舍五入到两位小数 */
const r2 = v => Math.round(v * 100) / 100;

// ==================== 主分析函数 ====================

/**
 * 分析双色球数据
 * @param {Array} data - 开奖记录数组，按期号降序排列（最新在前）
 *   每条记录: { issue, date, reds: [Number], blue: Number }
 * @returns {Object} analyzed_result 完整分析结果
 */
function analyze(data) {
  if (!data || data.length === 0) {
    return { error: '没有可分析的数据' };
  }

  const totalIssues = data.length;
  const redNums = Array.from({ length: 33 }, (_, i) => i + 1);   // 1-33
  const blueNums = Array.from({ length: 16 }, (_, i) => i + 1);  // 1-16

  // 滑动窗口周期定义
  const windows = [
    { key: 'last30', size: Math.min(30, totalIssues) },
    { key: 'last100', size: Math.min(100, totalIssues) },
    { key: 'last1000', size: Math.min(1000, totalIssues) },
    { key: 'all', size: totalIssues }
  ];

  // ====== 第一步：一次遍历，收集每个号码出现的期数索引 ======
  // 索引 0 = 最新一期，索引越大 = 越早的期数
  const redIdx = {}, blueIdx = {};
  for (const n of redNums) redIdx[n] = [];
  for (const n of blueNums) blueIdx[n] = [];

  for (let i = 0; i < totalIssues; i++) {
    for (const r of data[i].reds) redIdx[r].push(i);
    blueIdx[data[i].blue].push(i);
  }

  // ====== 第二步：多周期频次统计 ======
  const frequency = { red: {}, blue: {} };
  for (const { key, size } of windows) {
    frequency.red[key] = {};
    frequency.blue[key] = {};
    for (const n of redNums) {
      // 利用索引数组已排序的特性，用二分查找更快
      frequency.red[key][n] = countBelow(redIdx[n], size);
    }
    for (const n of blueNums) {
      frequency.blue[key][n] = countBelow(blueIdx[n], size);
    }
  }

  /** 统计索引数组中小于 limit 的元素个数（数组已升序） */
  function countBelow(sortedArr, limit) {
    let lo = 0, hi = sortedArr.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (sortedArr[mid] < limit) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  // ====== 第三步：遗漏值、间隔与标准差 ======
  function calcGapStats(appearIdx, total) {
    const currentGap = appearIdx.length > 0 ? appearIdx[0] : total;
    // 相邻两次出现之间的间隔
    const intervals = [];
    for (let i = 0; i < appearIdx.length - 1; i++) {
      intervals.push(appearIdx[i + 1] - appearIdx[i]);
    }
    const avgGap = intervals.length > 0
      ? intervals.reduce((s, v) => s + v, 0) / intervals.length
      : total;
    const gapRatio = avgGap > 0 ? currentGap / avgGap : 0;
    const stdDev = calcStdDev(intervals);

    return { currentGap, avgGap: r2(avgGap), gapRatio: r2(gapRatio), stdDev: r2(stdDev) };
  }

  const numberStats = { red: {}, blue: {} };
  for (const n of redNums) numberStats.red[n] = calcGapStats(redIdx[n], totalIssues);
  for (const n of blueNums) numberStats.blue[n] = calcGapStats(blueIdx[n], totalIssues);

  // ====== 第四步：综合多策略打分（支持多周期） ======
  /**
   * @param {string} periodKey - 频次窗口 (last30/last100/last1000/all)
   * @param {boolean} writeStats - 是否将分数写入 numberStats（仅 last30 写入，供雷达图使用）
   */
  function calcScores(numbers, stats, freq, periodKey, writeStats) {
    // 策略A：追热点动量 —— 指定周期内频次越高分越高
    const rawA = numbers.map(n => freq[periodKey][n] || 0);
    const scoresA = normalize(rawA);

    // 策略B：极限抄底 —— 遗漏比值越大分越高（不受周期影响）
    const rawB = numbers.map(n => stats[n].gapRatio);
    const scoresB = normalize(rawB);

    // 策略C：稳健节奏 —— 标准差小 + 当前遗漏接近历史均值（不受周期影响）
    const maxStd = Math.max(...numbers.map(n => stats[n].stdDev), 1);
    const rawC = numbers.map(n => {
      const s = stats[n];
      const regularity = 1 - s.stdDev / maxStd;
      const timing = s.avgGap > 0 ? Math.max(0, 1 - Math.abs(s.currentGap - s.avgGap) / s.avgGap) : 0;
      return regularity * 50 + timing * 50;
    });
    const scoresC = normalize(rawC);

    // 综合指数 = 三策略均分
    const composite = numbers.map((_, i) =>
      Math.round((scoresA[i] + scoresB[i] + scoresC[i]) / 3)
    );

    // 仅 last30 时写入 numberStats（供雷达图弹窗使用）
    if (writeStats) {
      numbers.forEach((n, i) => {
        stats[n].scores = {
          strategyA: scoresA[i],
          strategyB: scoresB[i],
          strategyC: scoresC[i],
          composite: composite[i]
        };
      });
    }

    // 生成排名
    const rank = scores => numbers
      .map((n, i) => ({ number: n, score: scores[i] }))
      .sort((a, b) => b.score - a.score);

    return {
      strategyA: rank(scoresA),
      strategyB: rank(scoresB),
      strategyC: rank(scoresC),
      composite: rank(composite)
    };
  }

  // 为每个周期都计算排名
  const periodKeys = ['last30', 'last100', 'last1000', 'all'];
  const rankings = {};
  for (const pk of periodKeys) {
    rankings[pk] = {
      red: calcScores(redNums, numberStats.red, frequency.red, pk, pk === 'last30'),
      blue: calcScores(blueNums, numberStats.blue, frequency.blue, pk, pk === 'last30')
    };
  }

  // ====== 第五步：雷达图归一化数据 ======
  function calcRadar(numbers, stats, freq) {
    const rh = normalize(numbers.map(n => freq.last30[n] || 0));
    const mt = normalize(numbers.map(n => freq.last100[n] || 0));
    const gd = normalize(numbers.map(n => stats[n].gapRatio));
    const maxS = Math.max(...numbers.map(n => stats[n].stdDev), 1);
    const rs = normalize(numbers.map(n => (1 - stats[n].stdDev / maxS) * 100));

    numbers.forEach((n, i) => {
      stats[n].radarData = {
        recentHeat: rh[i],
        midTrend: mt[i],
        gapDeviation: gd[i],
        rhythmStability: rs[i],
        compositeScore: stats[n].scores.composite
      };
    });
  }

  calcRadar(redNums, numberStats.red, frequency.red);
  calcRadar(blueNums, numberStats.blue, frequency.blue);

  // ====== 第六步：终极混合推荐（跨周期加权融合） ======
  function buildMixed(numbers, type) {
    const topN = type === 'red' ? 6 : 2;
    return numbers.map(num => {
      let totalScore = 0;
      let agreement = 0;  // 在几个周期的综合排名前 topN 里出现
      const periodScores = {};

      for (const pk of periodKeys) {
        const entry = rankings[pk][type].composite.find(e => e.number === num);
        const sc = entry ? entry.score : 0;
        totalScore += sc;
        periodScores[pk] = sc;
        // 判断是否在该周期的前 topN
        const topList = rankings[pk][type].composite.slice(0, topN).map(e => e.number);
        if (topList.includes(num)) agreement++;
      }

      return {
        number: num,
        score: Math.round(totalScore / periodKeys.length),
        agreement,       // 0-4，共识度
        periodScores     // 各周期得分明细
      };
    }).sort((a, b) => b.score - a.score || b.agreement - a.agreement);
  }

  const mixedRankings = {
    red: buildMixed(redNums, 'red'),
    blue: buildMixed(blueNums, 'blue')
  };

  // ====== 第七步：节奏周期预测 ======
  // 核心思路：每个号码都有自己的"出现节奏"（平均间隔），
  // 当前遗漏越接近或超过平均间隔 → 越可能即将出现
  // 节奏越规律（变异系数低）→ 预测置信度越高
  function buildRhythmPrediction(numbers, stats) {
    return numbers.map(num => {
      const s = stats[num];
      const avgInt = s.avgGap;                         // 平均出现间隔
      const curGap = s.currentGap;                     // 当前遗漏
      const sd = s.stdDev;                             // 间隔标准差

      // 变异系数 CV = stdDev / mean，越小越规律
      const cv = avgInt > 0 ? sd / avgInt : 1;
      // 置信度 = 1/(1+CV)，范围 (0, 1]，越规律越接近1
      const confidence = r2(1 / (1 + cv));

      // 超期比 = 当前遗漏 / 平均间隔
      const overdueRatio = avgInt > 0 ? curGap / avgInt : 0;

      // 预测得分：超期越多 + 节奏越规律 → 分越高
      // 超期加权：超过1倍间隔后额外加分
      const overdueBonus = overdueRatio > 1 ? (overdueRatio - 1) * 0.5 : 0;
      const rawScore = (overdueRatio + overdueBonus) * confidence;

      // 预计还需多少期出现 (负数=已超期)
      const dueIn = r2(avgInt - curGap);

      return {
        number: num,
        avgInterval: avgInt,
        currentGap: curGap,
        stdDev: sd,
        confidence,
        overdueRatio: r2(overdueRatio),
        dueIn,          // 正数=还需X期, 负数=已超期X期
        rawScore
      };
    });
  }

  // 归一化分数并排序
  function normalizeAndSort(arr) {
    const scores = arr.map(a => a.rawScore);
    const normed = normalize(scores);
    arr.forEach((a, i) => { a.score = normed[i]; delete a.rawScore; });
    return arr.sort((a, b) => b.score - a.score);
  }

  const rhythmPrediction = {
    red: normalizeAndSort(buildRhythmPrediction(redNums, numberStats.red)),
    blue: normalizeAndSort(buildRhythmPrediction(blueNums, numberStats.blue))
  };

  // ====== 第八步：推荐组合生成（6红+1蓝） ======
  function pickTop(arr, n) {
    return arr.slice(0, n).map(function (e) { return e.number; });
  }
  function sortNums(arr) { return arr.slice().sort(function (a, b) { return a - b; }); }

  // 组合1：节奏破冰 —— 节奏预测得分最高的号码
  var combo1 = {
    name: '节奏破冰', icon: '⏱️', desc: '超期最久+节奏最规律的号码',
    reds: sortNums(pickTop(rhythmPrediction.red, 6)),
    blue: rhythmPrediction.blue[0].number
  };

  // 组合2：跨周期共识 —— 混合推荐得分最高
  var combo2 = {
    name: '跨周期共识', icon: '👑', desc: '四个周期综合排名最靠前',
    reds: sortNums(pickTop(mixedRankings.red, 6)),
    blue: mixedRankings.blue[0].number
  };

  // 组合3：近期追热 —— 近30期综合得分最高
  var combo3 = {
    name: '近期追热', icon: '🔥', desc: '近30期最热门号码',
    reds: sortNums(pickTop(rankings.last30.red.composite, 6)),
    blue: rankings.last30.blue.composite[0].number
  };

  // 组合4：极限抄底 —— 遗漏极值反转
  var combo4 = {
    name: '极限抄底', icon: '🧊', desc: '遗漏比值最高的冷号',
    reds: sortNums(pickTop(rankings.last30.red.strategyB, 6)),
    blue: rankings.last30.blue.strategyB[0].number
  };

  // 组合5：智能均衡 —— 从多个来源轮流挑选，确保三区均衡
  // 三区：1-11, 12-22, 23-33
  function balancedCombo() {
    var sources = [
      rhythmPrediction.red,
      mixedRankings.red,
      rankings.last30.red.composite,
      rankings.last30.red.strategyB,
      rankings.last30.red.strategyC
    ];
    var zones = [[], [], []]; // 三个区间各选了哪些
    var picked = new Set();
    var srcIdx = 0, attempts = 0;

    while (picked.size < 6 && attempts < 100) {
      var src = sources[srcIdx % sources.length];
      for (var j = 0; j < src.length; j++) {
        var n = src[j].number;
        if (picked.has(n)) continue;
        var zone = n <= 11 ? 0 : n <= 22 ? 1 : 2;
        // 每区最多3个，确保均衡
        if (zones[zone].length >= 3) continue;
        zones[zone].push(n);
        picked.add(n);
        break;
      }
      srcIdx++;
      attempts++;
    }
    // 如果还不够6个，从任意来源补足
    if (picked.size < 6) {
      for (var s = 0; s < sources.length && picked.size < 6; s++) {
        for (var k = 0; k < sources[s].length && picked.size < 6; k++) {
          if (!picked.has(sources[s][k].number)) picked.add(sources[s][k].number);
        }
      }
    }
    return sortNums(Array.from(picked).slice(0, 6));
  }

  // 蓝球：取各策略蓝球排名第2的（避免和其他组合重复）
  var usedBlue = new Set([combo1.blue, combo2.blue, combo3.blue, combo4.blue]);
  var balBlue = rhythmPrediction.blue[0].number;
  for (var bi = 0; bi < rhythmPrediction.blue.length; bi++) {
    if (!usedBlue.has(rhythmPrediction.blue[bi].number)) {
      balBlue = rhythmPrediction.blue[bi].number;
      break;
    }
  }

  var combo5 = {
    name: '智能均衡', icon: '🎯', desc: '多策略轮选+三区均衡分布',
    reds: balancedCombo(),
    blue: balBlue
  };

  var recommendedCombos = [combo1, combo2, combo3, combo4, combo5];

  // ====== 构建最终结果 ======
  return {
    generatedAt: new Date().toISOString(),
    totalIssues,
    latestIssue: data[0].issue,
    latestDate: data[0].date,
    latestReds: data[0].reds,
    latestBlue: data[0].blue,
    frequency,
    numberStats,
    rankings,
    mixedRankings,
    rhythmPrediction,
    recommendedCombos
  };
}

module.exports = { analyze };
