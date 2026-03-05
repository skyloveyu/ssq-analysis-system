'use strict';

const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

// 使用 WASM 后端 (比纯 JS 快 10-20 倍，支持 SIMD)
let backendReady = (async () => {
    try {
        require('@tensorflow/tfjs-backend-wasm');
        const wasm = require('@tensorflow/tfjs-backend-wasm');
        wasm.setWasmPaths(
            path.join(__dirname, 'node_modules', '@tensorflow', 'tfjs-backend-wasm', 'dist') + '/'
        );
        await tf.setBackend('wasm');
        await tf.ready();
        console.log('🚀 TensorFlow.js 使用 WASM 后端 (SIMD 加速)');
    } catch (e) {
        await tf.setBackend('cpu');
        await tf.ready();
        console.log('⚠️ TensorFlow.js 使用 CPU 后端 (较慢)');
    }
})();

const MODEL_DIR = path.join(__dirname, 'data', 'model');
const WEIGHTS_FILE = path.join(MODEL_DIR, 'weights.json');
const META_FILE = path.join(MODEL_DIR, 'meta.json');

// ==================== 超参数 ====================
const SEQ_LEN = 10;
const RED_DIM = 33;
const BLUE_DIM = 16;
const FEATURE_DIM = RED_DIM + BLUE_DIM; // 49
const D_MODEL = 32;
const D_FF = 64;
const NUM_LAYERS = 2;
const EPOCHS = 30;
const BATCH_SIZE = 64;
const LEARNING_RATE = 0.001;

// ==================== 训练状态 ====================
let trainingStatus = {
    isTraining: false,
    progress: 0,
    epoch: 0,
    totalEpochs: EPOCHS,
    loss: null,
    message: '未训练',
    lastTrained: null,
    hasModel: false
};

// ==================== 数据预处理 ====================

function encodeDraw(d) {
    const vec = new Array(FEATURE_DIM).fill(0);
    d.reds.forEach(r => { vec[r - 1] = 1; });
    vec[RED_DIM + d.blue - 1] = 1;
    return vec;
}

function createSequences(data) {
    // data 按期号降序，需要反转为时间顺序
    const ordered = data.slice().reverse();
    const encoded = ordered.map(encodeDraw);
    const X = [], yRed = [], yBlue = [];

    for (let i = SEQ_LEN; i < encoded.length; i++) {
        X.push(encoded.slice(i - SEQ_LEN, i));
        yRed.push(encoded[i].slice(0, RED_DIM));
        yBlue.push(encoded[i].slice(RED_DIM));
    }
    return { X, yRed, yBlue };
}

// ==================== Layer Norm ====================

function layerNorm(x, gamma, beta) {
    const mean = x.mean(-1, true);
    const variance = x.sub(mean).square().mean(-1, true);
    return x.sub(mean).div(variance.add(1e-6).sqrt()).mul(gamma).add(beta);
}

// ==================== Transformer 模型 ====================

class TransformerPredictor {
    constructor() {
        this.weights = null;
    }

    initWeights() {
        const glorot = (rows, cols) => {
            const std = Math.sqrt(2.0 / (rows + cols));
            return tf.variable(tf.randomNormal([rows, cols], 0, std));
        };

        this.weights = {
            proj: glorot(FEATURE_DIM, D_MODEL),
            posEnc: tf.variable(tf.randomNormal([SEQ_LEN, D_MODEL], 0, 0.02)),
            blocks: Array.from({ length: NUM_LAYERS }, (_, i) => ({
                wq: glorot(D_MODEL, D_MODEL),
                wk: glorot(D_MODEL, D_MODEL),
                wv: glorot(D_MODEL, D_MODEL),
                wo: glorot(D_MODEL, D_MODEL),
                ffW1: glorot(D_MODEL, D_FF),
                ffB1: tf.variable(tf.zeros([D_FF])),
                ffW2: glorot(D_FF, D_MODEL),
                ffB2: tf.variable(tf.zeros([D_MODEL])),
                lnG1: tf.variable(tf.ones([D_MODEL])),
                lnB1: tf.variable(tf.zeros([D_MODEL])),
                lnG2: tf.variable(tf.ones([D_MODEL])),
                lnB2: tf.variable(tf.zeros([D_MODEL]))
            })),
            outRedW: glorot(D_MODEL, RED_DIM),
            outRedB: tf.variable(tf.zeros([RED_DIM])),
            outBlueW: glorot(D_MODEL, BLUE_DIM),
            outBlueB: tf.variable(tf.zeros([BLUE_DIM]))
        };
    }

    getAllVariables() {
        const vars = [this.weights.proj, this.weights.posEnc,
        this.weights.outRedW, this.weights.outRedB,
        this.weights.outBlueW, this.weights.outBlueB];
        for (const b of this.weights.blocks) {
            vars.push(b.wq, b.wk, b.wv, b.wo, b.ffW1, b.ffB1, b.ffW2, b.ffB2,
                b.lnG1, b.lnB1, b.lnG2, b.lnB2);
        }
        return vars;
    }

    forward(x) {
        const batchSize = x.shape[0];
        const seqLen = x.shape[1];

        // 投影到 D_MODEL 维度
        let h = tf.matMul(x.reshape([-1, FEATURE_DIM]), this.weights.proj)
            .reshape([batchSize, seqLen, D_MODEL]);

        // 加位置编码
        h = h.add(this.weights.posEnc);

        // Transformer 编码层
        for (const block of this.weights.blocks) {
            // Self-Attention
            const flat = h.reshape([-1, D_MODEL]);
            const q = tf.matMul(flat, block.wq).reshape([batchSize, seqLen, D_MODEL]);
            const k = tf.matMul(flat, block.wk).reshape([batchSize, seqLen, D_MODEL]);
            const v = tf.matMul(flat, block.wv).reshape([batchSize, seqLen, D_MODEL]);

            const scale = Math.sqrt(D_MODEL);
            const scores = tf.matMul(q, k, false, true).div(tf.scalar(scale));
            const attnW = tf.softmax(scores, -1);
            const attnOut = tf.matMul(attnW, v);

            const attnProj = tf.matMul(attnOut.reshape([-1, D_MODEL]), block.wo)
                .reshape([batchSize, seqLen, D_MODEL]);

            h = layerNorm(h.add(attnProj), block.lnG1, block.lnB1);

            // Feed-Forward Network
            let ff = tf.relu(
                tf.matMul(h.reshape([-1, D_MODEL]), block.ffW1).add(block.ffB1)
            );
            ff = tf.matMul(ff, block.ffW2).add(block.ffB2);
            ff = ff.reshape([batchSize, seqLen, D_MODEL]);

            h = layerNorm(h.add(ff), block.lnG2, block.lnB2);
        }

        // 均值池化
        const pooled = h.mean(1);

        const redLogits = tf.matMul(pooled, this.weights.outRedW).add(this.weights.outRedB);
        const blueLogits = tf.matMul(pooled, this.weights.outBlueW).add(this.weights.outBlueB);

        return { redLogits, blueLogits };
    }

    async train(data, statusCb) {
        if (this.weights) this.disposeWeights();
        this.initWeights();

        const { X, yRed, yBlue } = createSequences(data);
        const totalSamples = X.length;
        if (totalSamples < BATCH_SIZE) {
            throw new Error(`数据不足: 需要至少 ${SEQ_LEN + BATCH_SIZE} 期数据`);
        }

        const xTensor = tf.tensor3d(X);
        const yRedTensor = tf.tensor2d(yRed);
        const yBlueTensor = tf.tensor2d(yBlue);

        const optimizer = tf.train.adam(LEARNING_RATE);
        const allVars = this.getAllVariables();
        const numBatches = Math.ceil(totalSamples / BATCH_SIZE);

        for (let epoch = 0; epoch < EPOCHS; epoch++) {
            let epochLoss = 0;

            for (let b = 0; b < numBatches; b++) {
                const start = b * BATCH_SIZE;
                const size = Math.min(BATCH_SIZE, totalSamples - start);

                const xB = xTensor.slice([start, 0, 0], [size, -1, -1]);
                const yRB = yRedTensor.slice([start, 0], [size, -1]);
                const yBB = yBlueTensor.slice([start, 0], [size, -1]);

                const loss = optimizer.minimize(() => {
                    const { redLogits, blueLogits } = this.forward(xB);
                    const rL = tf.losses.sigmoidCrossEntropy(yRB, redLogits);
                    const bL = tf.losses.softmaxCrossEntropy(yBB, blueLogits);
                    return rL.add(bL);
                }, true, allVars);

                epochLoss += loss.dataSync()[0];
                loss.dispose();
                xB.dispose(); yRB.dispose(); yBB.dispose();
            }

            const avgLoss = (epochLoss / numBatches).toFixed(4);
            if (statusCb) statusCb(epoch + 1, EPOCHS, avgLoss);

            // 让出事件循环
            await new Promise(r => setTimeout(r, 0));
        }

        xTensor.dispose(); yRedTensor.dispose(); yBlueTensor.dispose();
        optimizer.dispose();
    }

    predict(data) {
        if (!this.weights) throw new Error('模型未训练');

        // 取最近 SEQ_LEN 期数据作为输入
        const recent = data.slice(0, SEQ_LEN).reverse();
        const encoded = recent.map(encodeDraw);

        const result = tf.tidy(() => {
            const input = tf.tensor3d([encoded]);
            const { redLogits, blueLogits } = this.forward(input);
            const redProbs = tf.sigmoid(redLogits).dataSync();
            const blueProbs = tf.softmax(blueLogits).dataSync();
            return { redProbs: Array.from(redProbs), blueProbs: Array.from(blueProbs) };
        });

        // 选出概率最高的6个红球和1个蓝球
        const redRanked = result.redProbs
            .map((p, i) => ({ number: i + 1, prob: p }))
            .sort((a, b) => b.prob - a.prob);

        const blueRanked = result.blueProbs
            .map((p, i) => ({ number: i + 1, prob: p }))
            .sort((a, b) => b.prob - a.prob);

        const pickedReds = redRanked.slice(0, 6)
            .map(r => r.number).sort((a, b) => a - b);
        const pickedBlue = blueRanked[0].number;

        return {
            reds: pickedReds,
            blue: pickedBlue,
            redProbabilities: redRanked.slice(0, 10),
            blueProbabilities: blueRanked.slice(0, 5),
            allRedProbs: result.redProbs,
            allBlueProbs: result.blueProbs
        };
    }

    saveWeights() {
        fs.mkdirSync(MODEL_DIR, { recursive: true });
        const data = {};
        function saveObj(obj, prefix) {
            for (const [key, val] of Object.entries(obj)) {
                if (val && typeof val.dataSync === 'function') {
                    data[prefix + key] = { values: Array.from(val.dataSync()), shape: val.shape };
                } else if (Array.isArray(val)) {
                    val.forEach((item, i) => saveObj(item, prefix + key + '_' + i + '_'));
                } else if (val && typeof val === 'object' && !ArrayBuffer.isView(val)) {
                    saveObj(val, prefix + key + '_');
                }
            }
        }
        saveObj(this.weights, '');
        fs.writeFileSync(WEIGHTS_FILE, JSON.stringify(data), 'utf-8');
        fs.writeFileSync(META_FILE, JSON.stringify({
            lastTrained: new Date().toISOString(),
            params: { SEQ_LEN, D_MODEL, D_FF, NUM_LAYERS, EPOCHS, BATCH_SIZE }
        }), 'utf-8');
    }

    loadWeights() {
        if (!fs.existsSync(WEIGHTS_FILE)) return false;
        this.initWeights();
        const data = JSON.parse(fs.readFileSync(WEIGHTS_FILE, 'utf-8'));
        function loadObj(obj, prefix) {
            for (const [key, val] of Object.entries(obj)) {
                if (val && typeof val.assign === 'function') {
                    const saved = data[prefix + key];
                    if (saved) val.assign(tf.tensor(saved.values, saved.shape));
                } else if (Array.isArray(val)) {
                    val.forEach((item, i) => loadObj(item, prefix + key + '_' + i + '_'));
                } else if (val && typeof val === 'object' && typeof val.assign !== 'function' && !ArrayBuffer.isView(val)) {
                    loadObj(val, prefix + key + '_');
                }
            }
        }
        loadObj(this.weights, '');
        const meta = JSON.parse(fs.readFileSync(META_FILE, 'utf-8'));
        trainingStatus.lastTrained = meta.lastTrained;
        trainingStatus.hasModel = true;
        trainingStatus.message = '模型已加载';
        return true;
    }

    disposeWeights() {
        if (!this.weights) return;
        function disposeObj(obj) {
            for (const val of Object.values(obj)) {
                if (val && typeof val.dispose === 'function') val.dispose();
                else if (Array.isArray(val)) val.forEach(disposeObj);
                else if (val && typeof val === 'object') disposeObj(val);
            }
        }
        disposeObj(this.weights);
        this.weights = null;
    }
}

// ==================== 单例 ====================
const predictor = new TransformerPredictor();

function getStatus() { return { ...trainingStatus }; }

async function trainModel(data) {
    await backendReady;
    if (trainingStatus.isTraining) throw new Error('模型正在训练中');

    trainingStatus.isTraining = true;
    trainingStatus.progress = 0;
    trainingStatus.epoch = 0;
    trainingStatus.message = '正在训练...';

    try {
        await predictor.train(data, (epoch, total, loss) => {
            trainingStatus.epoch = epoch;
            trainingStatus.totalEpochs = total;
            trainingStatus.progress = Math.round((epoch / total) * 100);
            trainingStatus.loss = loss;
            trainingStatus.message = `训练中 ${epoch}/${total} loss=${loss}`;
        });
        predictor.saveWeights();
        trainingStatus.isTraining = false;
        trainingStatus.hasModel = true;
        trainingStatus.progress = 100;
        trainingStatus.lastTrained = new Date().toISOString();
        trainingStatus.message = '训练完成';
        return { success: true, message: '模型训练完成' };
    } catch (err) {
        trainingStatus.isTraining = false;
        trainingStatus.message = '训练失败: ' + err.message;
        throw err;
    }
}

function predictNext(data) {
    if (!predictor.weights) {
        if (!predictor.loadWeights()) throw new Error('No trained model');
    }
    return predictor.predict(data);
}

async function tryLoadModel() {
    await backendReady;
    try { predictor.loadWeights(); } catch (e) { /* ignore */ }
}

module.exports = { getStatus, trainModel, predictNext, tryLoadModel };
