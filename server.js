/**
 * ============================================================
 *  双色球数据服务 (server.js)
 * ============================================================
 *  功能：
 *  1. Express 服务，提供 API 接口
 *  2. 从 500.com 抓取最新开奖数据
 *  3. 与本地 JSON 数据对比，增量更新
 *  4. 调用 analyzer.js 生成分析结果
 *  5. 启动时自动导入已有 CSV 数据
 *  6. 推荐数据保存与查询
 *  7. Transformer 模型训练与预测
 * ============================================================
 */
'use strict';

const express = require('express');
const axios = require('axios');
const cheerio = require('cheerio');
const fs = require('fs');
const path = require('path');
const { analyze } = require('./analyzer');
const { getStatus, trainModel, predictNext, tryLoadModel } = require('./transformer_predictor');

const app = express();

// ==================== 路径常量 ====================
const DATA_DIR = path.join(__dirname, 'data');
const DATA_FILE = path.join(DATA_DIR, 'lottery_data.json');
const ANALYZED_FILE = path.join(DATA_DIR, 'analyzed_result.json');
const RECOMMENDATIONS_FILE = path.join(DATA_DIR, 'recommendations.json');

// 确保目录存在
fs.mkdirSync(DATA_DIR, { recursive: true });
fs.mkdirSync(path.join(__dirname, 'public'), { recursive: true });

// ==================== 中间件 ====================
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

// ==================== 数据读写工具 ====================

function loadData() {
    if (fs.existsSync(DATA_FILE)) {
        return JSON.parse(fs.readFileSync(DATA_FILE, 'utf-8'));
    }
    return { lastUpdated: null, data: [] };
}

function saveData(obj) {
    obj.lastUpdated = new Date().toISOString();
    fs.writeFileSync(DATA_FILE, JSON.stringify(obj, null, 2), 'utf-8');
}

function loadAnalyzed() {
    if (fs.existsSync(ANALYZED_FILE)) {
        return JSON.parse(fs.readFileSync(ANALYZED_FILE, 'utf-8'));
    }
    return null;
}

function saveAnalyzed(result) {
    fs.writeFileSync(ANALYZED_FILE, JSON.stringify(result, null, 2), 'utf-8');
}

// ==================== 推荐数据存储 ====================

function loadRecommendations() {
    if (fs.existsSync(RECOMMENDATIONS_FILE)) {
        return JSON.parse(fs.readFileSync(RECOMMENDATIONS_FILE, 'utf-8'));
    }
    return {};
}

function saveRecommendations(recs) {
    fs.writeFileSync(RECOMMENDATIONS_FILE, JSON.stringify(recs, null, 2), 'utf-8');
}

/** 计算下一期期号和日期 */
function getNextDrawInfo(latestIssue, latestDate) {
    const nextIssue = String(Number(latestIssue) + 1);
    // 双色球开奖日：周二(2)、周四(4)、周日(0)
    const drawDays = [0, 2, 4];
    const d = new Date(latestDate + 'T20:00:00+08:00');
    let daysToAdd = 1;
    while (daysToAdd < 8) {
        const nextDay = new Date(d.getTime() + daysToAdd * 86400000);
        if (drawDays.includes(nextDay.getDay())) {
            const yyyy = nextDay.getFullYear();
            const mm = String(nextDay.getMonth() + 1).padStart(2, '0');
            const dd = String(nextDay.getDate()).padStart(2, '0');
            return { nextIssue, nextDate: `${yyyy}-${mm}-${dd}` };
        }
        daysToAdd++;
    }
    return { nextIssue, nextDate: '未知' };
}

/** 分析完成后自动保存推荐到下一期 */
function saveCurrentRecommendations(analyzed, data) {
    if (!analyzed || !analyzed.recommendedCombos || !data || data.length === 0) return;
    const recs = loadRecommendations();
    const { nextIssue, nextDate } = getNextDrawInfo(data[0].issue, data[0].date);
    recs[nextIssue] = {
        issue: nextIssue,
        date: nextDate,
        savedAt: new Date().toISOString(),
        basedOnIssue: data[0].issue,
        combos: analyzed.recommendedCombos,
        mixedRed: analyzed.mixedRankings ? analyzed.mixedRankings.red.slice(0, 6) : [],
        mixedBlue: analyzed.mixedRankings ? analyzed.mixedRankings.blue.slice(0, 2) : []
    };
    saveRecommendations(recs);
    console.log(`💾 推荐已保存至第${nextIssue}期 (${nextDate})`);
}

// ==================== CSV 导入（首次启动时） ====================

function importCSVIfNeeded() {
    if (fs.existsSync(DATA_FILE)) {
        const existing = loadData();
        if (existing.data.length > 0) {
            console.log(`📦 已有本地数据 ${existing.data.length} 条，跳过CSV导入`);
            return;
        }
    }

    const csvFiles = fs.readdirSync(__dirname)
        .filter(f => f.endsWith('.csv') && f.includes('双色球'))
        .sort((a, b) => {
            return fs.statSync(path.join(__dirname, b)).size
                - fs.statSync(path.join(__dirname, a)).size;
        });

    if (csvFiles.length === 0) {
        console.log('📭 未找到CSV文件，请通过页面"更新数据"按钮获取数据');
        return;
    }

    const csvPath = path.join(__dirname, csvFiles[0]);
    console.log(`📂 正在导入CSV: ${csvFiles[0]}`);

    const content = fs.readFileSync(csvPath, 'utf-8');
    const lines = content.split(/\r?\n/).filter(l => l.trim());
    const records = [];

    for (let i = 1; i < lines.length; i++) {
        const cols = lines[i].split(',').map(c => c.trim());
        if (cols.length < 9 || !cols[0]) continue;
        records.push({
            issue: cols[0],
            date: cols[1],
            reds: [cols[2], cols[3], cols[4], cols[5], cols[6], cols[7]].map(Number),
            blue: Number(cols[8])
        });
    }

    records.sort((a, b) => b.issue.localeCompare(a.issue));

    const lotteryData = { lastUpdated: new Date().toISOString(), data: records };
    saveData(lotteryData);
    console.log(`✅ 成功导入 ${records.length} 条CSV记录`);

    const result = analyze(records);
    saveAnalyzed(result);
    saveCurrentRecommendations(result, records);
    console.log('📊 分析完成，已生成 analyzed_result.json');
}

// ==================== 网页抓取 ====================

async function scrapeFromWeb() {
    const url = 'https://datachart.500.com/ssq/history/newinc/history.php?start=03001&end=99999';
    console.log('🌐 正在从 500.com 抓取数据...');

    const res = await axios.get(url, {
        headers: {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'zh-CN,zh;q=0.9'
        },
        timeout: 30000,
        responseType: 'arraybuffer'
    });

    let html;
    try {
        html = new TextDecoder('utf-8').decode(res.data);
        if (html.includes('�')) throw new Error('UTF-8 decode failed');
    } catch {
        html = new TextDecoder('gbk').decode(res.data);
    }

    const $ = cheerio.load(html);
    const records = [];

    $('#tdata tr').each((_, tr) => {
        const cols = $(tr).find('td');
        if (cols.length < 8) return;

        const issue = $(cols[0]).text().trim();
        if (!issue || !/^\d+$/.test(issue)) return;

        const reds = [];
        for (let j = 1; j <= 6; j++) {
            reds.push(Number($(cols[j]).text().trim()));
        }
        const blue = Number($(cols[7]).text().trim());

        if (reds.some(isNaN) || isNaN(blue)) return;

        records.push({ issue, date: $(cols[cols.length - 1]).text().trim(), reds, blue });
    });

    console.log(`🎯 抓取到 ${records.length} 条记录`);
    records.sort((a, b) => b.issue.localeCompare(a.issue));
    return records;
}

// ==================== API 接口 ====================

/** 获取分析结果 */
app.get('/api/analyzed', (req, res) => {
    const result = loadAnalyzed();
    if (!result) {
        return res.json({ error: '暂无分析数据，请先点击"更新数据"' });
    }
    // 附加下一期信息
    const d = loadData();
    if (d.data.length > 0) {
        const { nextIssue, nextDate } = getNextDrawInfo(d.data[0].issue, d.data[0].date);
        result.nextIssue = nextIssue;
        result.nextDate = nextDate;
    }
    res.json(result);
});

/** 获取原始数据摘要 */
app.get('/api/data', (req, res) => {
    const d = loadData();
    res.json({
        lastUpdated: d.lastUpdated,
        totalRecords: d.data.length,
        latest10: d.data.slice(0, 10)
    });
});

/** 更新数据：抓取 → 对比 → 合并 → 分析 */
app.post('/api/update', async (req, res) => {
    try {
        const scraped = await scrapeFromWeb();
        if (scraped.length === 0) {
            return res.json({ success: false, message: '未能从网站获取到数据，请稍后重试' });
        }

        const existing = loadData();
        const existingSet = new Set(existing.data.map(d => d.issue));

        const newRecords = scraped.filter(d => !existingSet.has(d.issue));

        const merged = { lastUpdated: null, data: scraped };
        saveData(merged);

        const result = analyze(scraped);
        saveAnalyzed(result);
        saveCurrentRecommendations(result, scraped);

        console.log(`✅ 更新完成: 总计 ${scraped.length} 条, 新增 ${newRecords.length} 条`);

        res.json({
            success: true,
            totalRecords: scraped.length,
            newCount: newRecords.length,
            newRecords: newRecords.slice(0, 20),
            message: newRecords.length > 0
                ? `成功更新！新增 ${newRecords.length} 条记录`
                : '数据已是最新，无新增记录'
        });
    } catch (err) {
        console.error('❌ 更新失败:', err.message);
        res.json({ success: false, message: `更新失败: ${err.message}` });
    }
});

/** 仅重新分析（不抓取） */
app.post('/api/reanalyze', (req, res) => {
    const d = loadData();
    if (d.data.length === 0) {
        return res.json({ success: false, message: '没有数据可分析' });
    }
    const result = analyze(d.data);
    saveAnalyzed(result);
    saveCurrentRecommendations(result, d.data);
    res.json({ success: true, message: `分析完成，共 ${d.data.length} 条数据` });
});

/** 获取某期推荐数据 */
app.get('/api/recommendations/:issue', (req, res) => {
    const recs = loadRecommendations();
    const rec = recs[req.params.issue];
    if (!rec) return res.json({ found: false });
    res.json({ found: true, data: rec });
});

/** 获取所有推荐数据的期号列表 */
app.get('/api/recommendations', (req, res) => {
    const recs = loadRecommendations();
    const issues = Object.keys(recs).sort((a, b) => b.localeCompare(a));
    res.json({ issues, count: issues.length });
});

/** 综合保存推荐（前端手动触发，包含所有推荐数据） */
app.post('/api/recommendations/save', (req, res) => {
    try {
        const d = loadData();
        if (d.data.length === 0) {
            return res.json({ success: false, message: '没有数据' });
        }
        const analyzed = loadAnalyzed();
        if (!analyzed) {
            return res.json({ success: false, message: '没有分析数据，请先更新' });
        }

        const { nextIssue, nextDate } = getNextDrawInfo(d.data[0].issue, d.data[0].date);
        const recs = loadRecommendations();

        // 构建完整推荐数据
        const recData = recs[nextIssue] || {};
        recData.issue = nextIssue;
        recData.date = nextDate;
        recData.savedAt = new Date().toISOString();
        recData.basedOnIssue = d.data[0].issue;

        // 1. 推荐号码组合
        if (analyzed.recommendedCombos) {
            recData.combos = analyzed.recommendedCombos;
        }

        // 2. 终极混合推荐（和首页一致：前6红+前2蓝）
        if (analyzed.mixedRankings) {
            recData.mixedRed = analyzed.mixedRankings.red.slice(0, 6);
            recData.mixedBlue = analyzed.mixedRankings.blue.slice(0, 2);
        }

        // 3. 策略推荐擂台赛（4个周期 × 4个策略）
        if (analyzed.rankings) {
            const arenaData = {};
            ['last30', 'last100', 'last1000', 'all'].forEach(pk => {
                if (analyzed.rankings[pk]) {
                    arenaData[pk] = {
                        red: {
                            strategyA: analyzed.rankings[pk].red.strategyA.slice(0, 6),
                            strategyB: analyzed.rankings[pk].red.strategyB.slice(0, 6),
                            strategyC: analyzed.rankings[pk].red.strategyC.slice(0, 6),
                            composite: analyzed.rankings[pk].red.composite.slice(0, 6)
                        },
                        blue: {
                            strategyA: analyzed.rankings[pk].blue.strategyA.slice(0, 2),
                            strategyB: analyzed.rankings[pk].blue.strategyB.slice(0, 2),
                            strategyC: analyzed.rankings[pk].blue.strategyC.slice(0, 2),
                            composite: analyzed.rankings[pk].blue.composite.slice(0, 2)
                        }
                    };
                }
            });
            recData.arena = arenaData;
        }

        // 4. AI 模型预测（从请求体接收）
        if (req.body && req.body.aiPrediction) {
            recData.aiPrediction = req.body.aiPrediction;
        }

        recs[nextIssue] = recData;
        saveRecommendations(recs);

        console.log(`💾 综合推荐已保存至第${nextIssue}期 (${nextDate})`);
        res.json({ success: true, message: `推荐数据已保存至第${nextIssue}期 (${nextDate})`, issue: nextIssue });
    } catch (err) {
        console.error('❌ 保存推荐失败:', err.message);
        res.json({ success: false, message: '保存失败: ' + err.message });
    }
});

/** 历史记录分页查询（含推荐标记与下一期预留行） */
app.get('/api/history', (req, res) => {
    const d = loadData();
    const page = Math.max(1, parseInt(req.query.page) || 1);
    const pageSize = Math.min(100, Math.max(10, parseInt(req.query.pageSize) || 30));
    const search = (req.query.search || '').trim();

    let filtered = d.data;
    if (search) {
        filtered = filtered.filter(r =>
            r.issue.includes(search) || (r.date && r.date.includes(search))
        );
    }

    // 标记哪些期号有推荐
    const recs = loadRecommendations();
    const recIssueSet = new Set(Object.keys(recs));

    // 计算下一期信息
    let nextDraw = null;
    if (d.data.length > 0 && page === 1 && !search) {
        const { nextIssue, nextDate } = getNextDrawInfo(d.data[0].issue, d.data[0].date);
        const hasRec = recIssueSet.has(nextIssue);
        nextDraw = { issue: nextIssue, date: nextDate, hasRecommendation: hasRec };
    }

    const total = filtered.length;
    const totalPages = Math.ceil(total / pageSize);
    const start = (page - 1) * pageSize;
    const records = filtered.slice(start, start + pageSize).map(r => ({
        ...r,
        hasRecommendation: recIssueSet.has(r.issue)
    }));

    res.json({ page, pageSize, total, totalPages, records, nextDraw });
});

// ==================== Transformer 模型 API ====================

/** 获取模型训练状态 */
app.get('/api/model/status', (req, res) => {
    res.json(getStatus());
});

/** 开始训练模型 */
app.post('/api/model/train', async (req, res) => {
    const d = loadData();
    if (d.data.length < 50) {
        return res.json({ success: false, message: '数据不足，至少需要50期数据' });
    }
    res.json({ success: true, message: '模型训练已启动' });
    try {
        await trainModel(d.data);
        console.log('🤖 Transformer 模型训练完成');
    } catch (err) {
        console.error('❌ 模型训练失败:', err.message);
    }
});

/** 模型预测 */
app.get('/api/model/predict', (req, res) => {
    try {
        const d = loadData();
        if (d.data.length < 10) {
            return res.json({ success: false, message: '数据不足' });
        }
        const prediction = predictNext(d.data);

        // 附加下一期信息
        const { nextIssue, nextDate } = getNextDrawInfo(d.data[0].issue, d.data[0].date);

        res.json({
            success: true,
            nextIssue,
            nextDate,
            prediction
        });
    } catch (err) {
        res.json({ success: false, message: err.message });
    }
});

const PREFERRED_PORT = 3001;

importCSVIfNeeded();
tryLoadModel();

function getLocalIP() {
    const os = require('os');
    const nets = os.networkInterfaces();
    for (const name of Object.keys(nets)) {
        for (const net of nets[name]) {
            if (net.family === 'IPv4' && !net.internal) return net.address;
        }
    }
    return 'localhost';
}

function tryListen(port, maxTries = 20) {
    const server = app.listen(port, () => {
        const ip = getLocalIP();
        console.log('');
        console.log('🎱 ============================================');
        console.log(`🎱  双色球智能分析系统已启动`);
        console.log(`🎱  本机访问: http://localhost:${port}`);
        console.log(`🎱  局域网:   http://${ip}:${port}`);
        console.log('🎱 ============================================');
        console.log('');
    });
    server.on('error', (err) => {
        if (err.code === 'EADDRINUSE' && maxTries > 0) {
            console.log(`⚠️  端口 ${port} 已被占用，尝试 ${port + 1}...`);
            tryListen(port + 1, maxTries - 1);
        } else {
            console.error('❌ 启动失败:', err.message);
        }
    });
}

tryListen(PREFERRED_PORT);
