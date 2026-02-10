const { RestClientV5 } = require('bybit-api');
const fs = require('fs');
const path = require('path');
const { EMA, RSI, BollingerBands, ATR } = require('trading-signals');
require('dotenv').config();

const ENV_FILENAME = '.env';
const envPath = path.resolve(__dirname, ENV_FILENAME);
if (!fs.existsSync(envPath)) {
    console.error(`❌ Lỗi: Thiếu file ${ENV_FILENAME}`); process.exit(1);
}
const clean = (s) => (s ? s.trim() : '');
const BYBIT_KEY = clean(process.env.BYBIT_KEY);
const BYBIT_SECRET = clean(process.env.BYBIT_SECRET);
const GEMINI_API_KEY = clean(process.env.GEMINI_API_KEY);

const CONFIG_PATH = path.join(__dirname, 'config.json');
let cfg;
try { cfg = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8').trim()); } catch (e) { process.exit(1); }

// --- KHAI BÁO BIẾN TOÀN CỤC ---
const G = cfg.global;
const S = cfg.symbol_settings.BTCUSDT;
const SYMBOL = 'BTCUSDT';
const CATEGORY = 'linear';

const USE_DEMO = G.ENV === 'DEMO';
const client = new RestClientV5({
    key: BYBIT_KEY, secret: BYBIT_SECRET, testnet: false,
    baseUrl: USE_DEMO ? 'https://api-demo.bybit.com' : 'https://api.bybit.com',
    recvWindow: 10000,
});

const FILE = {
    LOG: path.join(__dirname, 'trade_history.csv'),
    STATS: path.join(__dirname, 'performance_stats.csv'),
    PROMPT: path.join(__dirname, 'prompt.txt'),
    CACHE: path.join(__dirname, 'ai_cache.json'),
    STATE: path.join(__dirname, 'bot_state.json')
};

const C = { reset: '\x1b[0m', red: '\x1b[31m', green: '\x1b[32m', yellow: '\x1b[33m', cyan: '\x1b[36m', magenta: '\x1b[35m', dim: '\x1b[2m', bold: '\x1b[1m' };
const paint = (s, c) => c + String(s) + C.reset;
const time = () => new Date().toLocaleString('vi-VN', { timeZone: 'Asia/Ho_Chi_Minh' });

let googleKeys = [];
let hfKeys = [];

try {
    const keyContent = fs.readFileSync(G.KEY_FILE_PATH, 'utf8');
    const lines = keyContent.split('\n').map(l => l.trim()).filter(l => l.length > 10);
    
    lines.forEach(line => {
        const parts = line.split('|').map(p => p.trim());
        parts.forEach(part => {
            // Tự động nhận diện key dựa trên prefix
            if (part.startsWith('AIza')) {
                googleKeys.push(part);
            } else if (part.startsWith('hf_')) {
                hfKeys.push(part);
            }
        });
    });

    // Log chi tiết số lượng từng loại key
    console.log(paint(`🔑 Đã load tài nguyên:`, C.green));
    console.log(`   - Google Gemini Keys: ${paint(googleKeys.length, C.yellow)}`);
    console.log(`   - HuggingFace Keys:   ${paint(hfKeys.length, C.yellow)}`);
    
    if (googleKeys.length === 0 && hfKeys.length === 0) {
        console.error("❌ Không tìm thấy key nào hợp lệ!");
        process.exit(1);
    }
} catch (e) {
    console.error(`❌ Lỗi đọc file key: ${e.message}`);
    process.exit(1);
}

// Hàm lấy key ngẫu nhiên từ Pool tương ứng
const getGoogleKey = () => googleKeys.length > 0 ? googleKeys[Math.floor(Math.random() * googleKeys.length)] : null;
const getHfKey = () => hfKeys.length > 0 ? hfKeys[Math.floor(Math.random() * hfKeys.length)] : null;

const printClean = (label, color, ...args) => {
    // \r: Về đầu dòng | \x1b[K: Xóa sạch dòng hiện tại
    process.stdout.write('\r\x1b[K'); 
    console.log(time(), paint(label, color), ...args);
};

const log = {
    info: (...a) => printClean('INFO:', C.green, ...a),
    warn: (...a) => printClean('WARN:', C.yellow, ...a),
    error: (...a) => printClean('ERROR:', C.red, ...a),
    trade: (...a) => printClean('TRADE:', C.cyan, ...a),
    ai: (...a) => printClean('AI:', C.magenta, ...a),
    dim: (...a) => printClean('DIM:', C.dim, ...a),
};

// --- FIX: GLOBAL ERROR HANDLERS (Chống Crash) ---
process.on('uncaughtException', (err) => {
    log.error(`UNCAUGHT EXCEPTION: ${err.message}`);
    // Không exit process để bot tiếp tục chạy
});

process.on('unhandledRejection', (reason, promise) => {
    log.error(`UNHANDLED REJECTION: ${reason instanceof Error ? reason.message : reason}`);
});

// --- STATE MANAGEMENT ---
const state = {
    capital: 0, lastPrice: 0,
    filters: { tickSize: null, qtyStep: null, minQty: null, minNotional: null },
    position: { active: false, side: 'None', qty: 0, entryPrice: 0, currentSL: 0, initialR: 0, initialEntry: 0, reason: '' },
    pending: { active: false, side: 'None', qty: 0, entryPrice: 0, initialSL: 0, initialR: 0, reason: '', orderId: '' },
    lastAnalysis: 0, cooldownUntil: 0
};

function saveState() {
    // Lưu thêm cooldownUntil để nếu tắt bot bật lại vẫn nhớ thời gian chờ
    try { fs.writeFileSync(FILE.STATE, JSON.stringify({ position: state.position, pending: state.pending, cooldownUntil: state.cooldownUntil }, null, 2)); } catch (e) {}
}

function loadState() {
    try {
        if (fs.existsSync(FILE.STATE)) {
            const d = JSON.parse(fs.readFileSync(FILE.STATE, 'utf8'));
            if (d.position) state.position = d.position;
            if (d.pending) state.pending = d.pending;
	    if (d.cooldownUntil) state.cooldownUntil = d.cooldownUntil;
        }
    } catch (e) {}
}

// --- CSV REPORTING ---
function logTradeHistory(data) {
    const headers = 'Time,Symbol,Type,Side,Price,Qty,PnL_USDT,PnL_R,Reason,Status\n';
    if (!fs.existsSync(FILE.LOG)) fs.writeFileSync(FILE.LOG, headers);
    const row = [
        time().replace(',', ''), SYMBOL, data.type, data.side, data.price, data.qty,
        data.pnl || 0, data.pnlR || 0, `"${(data.reason || '').replace(/"/g, '""')}"`, data.status
    ].join(',') + '\n';
    fs.appendFileSync(FILE.LOG, row);
}

function updatePerformance(isWin, rMultiple, realPnL = 0) {
    const headers = 'TotalTrades,Wins,Losses,WinRate,TotalR,TotalPnL_USDT,Longs,Shorts\n';
    let stats = { total: 0, wins: 0, losses: 0, totalR: 0, totalPnL: 0, longs: 0, shorts: 0 };
    
    if (fs.existsSync(FILE.STATS)) {
        try {
            const rows = fs.readFileSync(FILE.STATS, 'utf8').trim().split('\n');
            if (rows.length > 1) {
                const lastRow = rows[rows.length - 1].split(',');
                stats.total = Number(lastRow[0]);
                stats.wins = Number(lastRow[1]);
                stats.losses = Number(lastRow[2]);
                stats.totalR = Number(lastRow[4]);
                stats.totalPnL = Number(lastRow[5]) || 0;
                stats.longs = Number(lastRow[6]);
                stats.shorts = Number(lastRow[7]);
            }
        } catch (e) {}
    }
    
    stats.total++;
    stats.totalR += rMultiple;
    stats.totalPnL += realPnL;
    
    if (isWin) stats.wins++; else stats.losses++;
    if (state.position.side === 'Buy') stats.longs++; else stats.shorts++;
    
    const winRate = ((stats.wins / stats.total) * 100).toFixed(2) + '%';
    const newRow = [stats.total, stats.wins, stats.losses, winRate, stats.totalR.toFixed(2), stats.totalPnL.toFixed(2), stats.longs, stats.shorts].join(',');
    
    if (!fs.existsSync(FILE.STATS)) fs.writeFileSync(FILE.STATS, headers);
    fs.appendFileSync(FILE.STATS, newRow + '\n');
    
    log.info(`📊 Stats: ${stats.total} lệnh | WinRate: ${winRate} | Total R: ${stats.totalR.toFixed(2)}R | PnL: ${stats.totalPnL.toFixed(2)}$`);
}

// --- UTILS ---
async function apiCallWithRetry(fn, maxRetries = 3) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            return await fn();
        } catch (e) {
            // Nếu là lỗi 4xx (Client Error) như sai params thì không retry, throw luôn
            if (e.code && e.code.toString().startsWith('4')) throw e; 
            
            if (i === maxRetries - 1) throw e; // Hết lượt retry thì ném lỗi ra ngoài
            
            log.dim(`⚠️ Lỗi kết nối (${e.message}). Đang thử lại lần ${i + 1}/${maxRetries}...`);
            await new Promise(resolve => setTimeout(resolve, 2000 * (i + 1))); // Đợi tăng dần: 2s, 4s, 6s
        }
    }
}

const roundPrice = (p) => {
    const ts = Number(state.filters.tickSize);
    return (Math.round(Number(p) / ts) * ts).toFixed(ts.toString().split('.')[1]?.length || 0);
};
const roundQty = (q) => {
    const qs = Number(state.filters.qtyStep);
    return (Math.max(qs, Math.floor(Number(q) / qs) * qs)).toFixed(qs.toString().split('.')[1]?.length || 0);
};
const isSamePrice = (p1, p2) => Math.abs(parseFloat(p1) - parseFloat(p2)) < (parseFloat(state.filters.tickSize) * 2);

async function syncState() {
    try {
	// --- FIX: Bọc API Call trong Retry ---
        const [bal, pos, orders, ticker] = await apiCallWithRetry(() => Promise.all([
            client.getWalletBalance({ accountType: 'UNIFIED', coin: 'USDT' }),
            client.getPositionInfo({ category: CATEGORY, symbol: SYMBOL }),
            client.getActiveOrders({ category: CATEGORY, symbol: SYMBOL, openOnly: 0 }),
            client.getTickers({ category: CATEGORY, symbol: SYMBOL })
        ]));

        if (ticker.retCode === 0 && ticker.result?.list?.length > 0) {
            state.lastPrice = parseFloat(ticker.result.list[0].lastPrice);
        }

	// --- FIX: Lấy đúng Coin USDT & Hỗ trợ Equity ---
        if (bal.retCode === 0 && bal.result?.list?.length > 0) {
            const account = bal.result.list[0];
            // Tìm chính xác coin USDT trong danh sách tài sản
            const usdtAsset = account.coin?.find(c => c.coin === 'USDT');
            
            if (usdtAsset) {
                // Nếu cấu hình dùng Equity (Balance + Unrealized PnL) thì lấy field 'equity', ngược lại lấy 'walletBalance'
                // Lưu ý: Bybit V5 trả về 'equity' trong object coin. Nếu không có (trường hợp hiếm), fallback về walletBalance.
                state.capital = G.USE_EQUITY 
                    ? parseFloat(usdtAsset.equity || usdtAsset.walletBalance || 0)
                    : parseFloat(usdtAsset.walletBalance || 0);
            } else {
                // Trường hợp không tìm thấy USDT (ví dụ tài khoản mới tinh chưa nạp tiền hoặc lỗi API)
                state.capital = 0;
                log.warn('⚠️ Không tìm thấy tài sản USDT trong ví!');
            }
        }

        const onExOrders = orders.result?.list || [];
        const onExPos = pos.result?.list?.[0];
        const posSize = parseFloat(onExPos?.size || 0);

        // 1. Xử lý Vị thế
        if (posSize > 0) {
            if (!state.position.active) {
                log.trade(paint(`🔔 KHỚP LỆNH: ${onExPos.side} Size: ${posSize}`, C.green));
                state.position = {
                    active: true, side: onExPos.side, qty: posSize,
                    entryPrice: parseFloat(onExPos.avgPrice),
                    initialEntry: parseFloat(onExPos.avgPrice),
                    currentSL: parseFloat(onExPos.stopLoss) || 0,
                    initialR: state.pending.initialR || (Math.abs(parseFloat(onExPos.avgPrice) - (parseFloat(onExPos.stopLoss)||0))) || 0,
                    reason: state.pending.reason || 'Synced'
                };
                logTradeHistory({ type: 'ENTRY', side: onExPos.side, price: onExPos.avgPrice, qty: posSize, reason: state.position.reason, status: 'OPEN' });
                state.pending = { active: false, side: 'None', qty: 0, reason: '' };
                saveState();
            } else {
                const apiSL = parseFloat(onExPos.stopLoss) || 0;
                if (Math.abs(apiSL - state.position.currentSL) > parseFloat(state.filters.tickSize)) {
                    state.position.currentSL = apiSL;
                    saveState();
                }
            }
        } else {
            if (state.position.active) {
                log.trade(paint(`🔒 ĐÃ ĐÓNG VỊ THẾ ${state.position.side}`, C.yellow));
                const exitPrice = state.position.currentSL || state.lastPrice; 
                const entry = state.position.initialEntry;
                const R = state.position.initialR;
                let rawPnL = state.position.side === 'Buy' ? (exitPrice - entry) * state.position.qty : (entry - exitPrice) * state.position.qty;
                let pnlR = 0;
                if (R > 0) pnlR = state.position.side === 'Buy' ? (exitPrice - entry) / R : (entry - exitPrice) / R;
                
                logTradeHistory({ type: 'EXIT', side: state.position.side, price: exitPrice, qty: state.position.qty, pnl: rawPnL.toFixed(2), pnlR: pnlR.toFixed(2), status: 'CLOSED' });
                updatePerformance(pnlR > 0, pnlR, rawPnL);
                state.position = { active: false, side: 'None', qty: 0 };
                saveState();
            }
        }

        // 2. Xử lý Lệnh chờ
        if (posSize === 0) {
            let matched = onExOrders.find(o => o.orderId === state.pending.orderId);
            if (!matched && state.pending.active) {
                matched = onExOrders.find(o => o.side === state.pending.side && isSamePrice(o.price, state.pending.entryPrice));
            }

            if (!matched && !state.pending.active && onExOrders.length > 0) {
                matched = onExOrders[0];
                log.warn(`⚠️ Nhận nuôi lệnh: ${matched.orderId}`);
                if (onExOrders.length > 1) {
                    log.warn(paint(`🧹 Dọn dẹp ${onExOrders.length - 1} lệnh thừa...`, C.yellow));
                    for (let i = 1; i < onExOrders.length; i++) {
                        client.cancelOrder({ category: CATEGORY, symbol: SYMBOL, orderId: onExOrders[i].orderId });
                    }
                }
            }

            if (matched) {
                state.pending = {
                    active: true, side: matched.side, qty: parseFloat(matched.qty),
                    entryPrice: parseFloat(matched.price), initialSL: parseFloat(matched.stopLoss),
                    initialR: state.pending.initialR || 0, reason: state.pending.reason || 'Synced',
                    orderId: matched.orderId
                };
                saveState();
            } else if (state.pending.active) {
                log.info(`❌ Lệnh chờ bị hủy/không tìm thấy.`);
                logTradeHistory({ type: 'CANCEL', side: state.pending.side, price: state.pending.entryPrice, reason: 'Market moved or Expired', status: 'CANCELLED' });
                state.pending = { active: false, side: 'None', qty: 0 };
                saveState();
                try { if(fs.existsSync(FILE.CACHE)) fs.unlinkSync(FILE.CACHE); } catch(e) {}
            }
        }

	// --- SỬA ĐỔI: DASHBOARD REAL-TIME (Cập nhật tại chỗ & Thêm Số dư) ---
	const balanceStr = paint(` | ${USE_DEMO ? 'DEMO' : 'REAL'}: ${state.capital.toFixed(2)}$`, C.magenta);
        
        if (posSize > 0 && onExPos) {
            const pnl = parseFloat(onExPos.unrealisedPnl || 0);
            const entry = parseFloat(onExPos.avgPrice || 0);
            const markPrice = state.lastPrice;
            
            // Định màu sắc: Lãi màu Xanh, Lỗ màu Đỏ
            const color = pnl >= 0 ? C.green : C.red;
            const pnlSign = pnl > 0 ? '+' : '';
            
            // Thêm balanceStr vào cuối
            const msg = paint(`⚡ [VỊ THẾ ${onExPos.side}] Khớp @ ${entry} | Giá: ${markPrice} | PnL: ${pnlSign}${pnl.toFixed(2)} USDT | SL: ${onExPos.stopLoss || 'None'} | TP: ${parseFloat(onExPos.takeProfit) > 0 ? onExPos.takeProfit : 'Trailing'}`, color);
            
	    process.stdout.write(`\r\x1b[K${msg}${balanceStr}`);
        
        } else if (state.pending.active) {
            const entry = state.pending.entryPrice;
            const dist = state.lastPrice - entry;
            const distStr = Math.abs(dist).toFixed(2);
            const direction = dist > 0 ? 'cao hơn' : 'thấp hơn';
            
            const msg = paint(`⏳ [CHỜ KHỚP ${state.pending.side}] Limit @ ${entry} | Giá: ${state.lastPrice} (Đang ${direction} ${distStr}) | SL: ${state.pending.initialSL}`, C.cyan);
            
            process.stdout.write(`\r\x1b[K${msg}`);
        }

        return true;
    } catch (e) {
	// Khi có lỗi, in xuống dòng để không bị ghi đè lên dashboard
        process.stdout.write('\n');
        log.error(`Sync Error: ${e.message}`); return false;
    }
}

// --- STEP TRAILING STOP ---
async function manageTrailingStop() {
    if (!state.position.active || state.position.initialR <= 0) return;
    const { side, initialEntry, initialR, currentSL } = state.position;
    const curPrice = state.lastPrice;
    
    let currentProfitR = side === 'Buy' ? (curPrice - initialEntry) / initialR : (initialEntry - curPrice) / initialR;
    
    let targetLockR = Math.floor(currentProfitR - 1); 
    if (currentProfitR >= 1.0 && targetLockR < 0) targetLockR = 0;

    if (targetLockR >= 0) {
        let newSL = 0;
        if (side === 'Buy') {
            newSL = parseFloat(roundPrice(initialEntry + (targetLockR * initialR)));
            if (newSL > currentSL) { 
                log.trade(`🚀 GỒNG LỜI (Long): Lãi ${currentProfitR.toFixed(2)}R -> Dời SL về ${targetLockR}R (${newSL})`);
                await client.setTradingStop({ category: CATEGORY, symbol: SYMBOL, stopLoss: String(newSL), positionIdx: 0 });
                state.position.currentSL = newSL;
                saveState();
            }
        } else {
            newSL = parseFloat(roundPrice(initialEntry - (targetLockR * initialR)));
            if (currentSL === 0 || newSL < currentSL) { 
                log.trade(`🚀 GỒNG LỜI (Short): Lãi ${currentProfitR.toFixed(2)}R -> Dời SL về ${targetLockR}R (${newSL})`);
                await client.setTradingStop({ category: CATEGORY, symbol: SYMBOL, stopLoss: String(newSL), positionIdx: 0 });
                state.position.currentSL = newSL;
                saveState();
            }
        }
    }
}

function getIndicators(closes, highs, lows, settings = { rsi: 14, ema: 200, bb: 20, bbDev: 2 }) {
    if (closes.length < 200) return null;

    const rsi = new RSI(settings.rsi);
    const ema = new EMA(settings.ema);
    const bb = new BollingerBands(settings.bb, settings.bbDev);
    const atr = new ATR(14); 

    closes.forEach((price, i) => {
        rsi.update(price);
        ema.update(price);
        bb.update(price);
        if (highs && lows) atr.update({ high: highs[i], low: lows[i], close: price });
    });

    if (!rsi.isStable || !ema.isStable || !bb.isStable) return null;

    const bbRes = bb.getResult();
    const lastClose = closes[closes.length - 1];
    const width = (Number(bbRes.upper) - Number(bbRes.lower)) / Number(bbRes.middle); // BB Width
    // %B Calculation
    const pb = (Number(bbRes.upper) - Number(bbRes.lower)) === 0 ? 0.5 : (lastClose - Number(bbRes.lower)) / (Number(bbRes.upper) - Number(bbRes.lower));

    return {
        rsi: Number(rsi.getResult()),
        ema: Number(ema.getResult()),
        atr: atr.isStable ? Number(atr.getResult()) : 0,
        bb: { upper: Number(bbRes.upper), middle: Number(bbRes.middle), lower: Number(bbRes.lower), width, pb }
    };
}

// --- NEW: SYSTEM PROMPT LOADER ---
// Hàm cắt prompt.txt thành 3 phần riêng biệt dựa trên tag
function loadSystemPrompts() {
    try {
        const content = fs.readFileSync(FILE.PROMPT, 'utf8');
        const extract = (tag) => {
            const start = content.indexOf(tag);
            if (start === -1) return "";
            // Tìm tag tiếp theo bắt đầu bằng [--- hoặc hết file
            const nextTagIndex = content.indexOf('[---', start + tag.length);
            return content.substring(start + tag.length, nextTagIndex === -1 ? content.length : nextTagIndex).trim();
        };

        return {
            quant: extract('[---QUANT_AGENT_SYSTEM---]'),
            strategist: extract('[---STRATEGIST_AGENT_SYSTEM---]'),
            risk: extract('[---RISK_MASTER_SYSTEM---]')
        };
    } catch (e) {
        log.error(`Lỗi đọc prompt.txt: ${e.message}`);
        return null;
    }
}

// --- NEW: API CALLER FOR AGENTS ---
// 1. Gọi HuggingFace (Có Retry cho lỗi 503/504)
async function callHuggingFace(modelConfig, systemPrompt, userContent, apiKey) {
    const url = "https://router.huggingface.co/v1/chat/completions";
    const payload = {
        model: modelConfig.model_id,
        messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: userContent }
        ],
        temperature: modelConfig.temperature,
        max_tokens: modelConfig.max_tokens,
        stream: false
    };

    // Thử lại tối đa 1 lần nếu gặp lỗi Server (5xx)
    const maxRetries = 1;
    for (let i = 0; i < maxRetries; i++) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 6000000); // Timeout cứng 6000s phía Client

            const res = await fetch(url, {
                method: "POST",
                headers: { "Authorization": `Bearer ${apiKey}`, "Content-Type": "application/json" },
                body: JSON.stringify(payload),
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            // QUAN TRỌNG: Lấy text trước, không parse JSON vội
            const rawText = await res.text();

            // Xử lý các trường hợp lỗi thường gặp của HF
            if (res.status === 503 || res.status === 504) {
                log.warn(`⚠️ HF Busy/Loading (Status ${res.status}). Đợi 5s thử lại (${i + 1}/${maxRetries})...`);
                await new Promise(r => setTimeout(r, 5000)); // Đợi 5s để model load
                continue; // Thử lại vòng lặp
            }

            if (!res.ok) {
                // In ra 200 ký tự đầu của lỗi để debug (tránh in cả trang HTML dài)
                const errPreview = rawText.replace(/\n/g, ' ').substring(0, 200);
                throw new Error(`HF Status ${res.status}: ${errPreview}...`);
            }

            // Nếu thành công 200 OK
            try {
                const data = JSON.parse(rawText);
                return data.choices?.[0]?.message?.content || null;
            } catch (jsonErr) {
                throw new Error(`HF trả về 200 OK nhưng không phải JSON hợp lệ. Raw: ${rawText.substring(0, 50)}...`);
            }

        } catch (e) {
            // Nếu là lần thử cuối cùng thì in lỗi và bỏ qua
            if (i === maxRetries - 1) {
                log.error(`HF Error (${modelConfig.role}) sau ${maxRetries} lần thử: ${e.message}`);
                return null;
            }
            // Nếu lỗi mạng (fetch failed) thì đợi chút rồi retry
            if (e.name === 'AbortError') log.warn(`⚠️ HF Request Timeout (Client side). Thử lại...`);
        }
    }
    return null;
}

// 2. Gọi Gemini (Cho Risk Manager)
// --- SỬA ĐỔI: Thêm log vào hàm gọi API Gemini ---
async function callGemini(modelConfig, systemPrompt, userContent, apiKey) {
    // 1. Trim key lần cuối để chắc chắn không có khoảng trắng
    const cleanKey = apiKey ? apiKey.trim() : ""; 
    
    if (!cleanKey) {
        log.error("❌ Gemini Call Aborted: API Key rỗng!");
        return null;
    }

    const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelConfig.model_id}:generateContent?key=${cleanKey}`;
    
    const payload = {
        contents: [{ parts: [{ text: userContent }] }],
        systemInstruction: { parts: [{ text: systemPrompt }] },
        generationConfig: {
            temperature: modelConfig.temperature,
            maxOutputTokens: modelConfig.max_tokens,
            responseMimeType: "application/json"
        }
    };

    try {
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            // Log lỗi chi tiết hơn
            const errText = await res.text();
            throw new Error(`Gemini Status ${res.status}: ${errText.substring(0, 200)}`); 
        }
        const data = await res.json();
        return data.candidates?.[0]?.content?.parts?.[0]?.text || null;
    } catch (e) {
        log.error(`Gemini Error: ${e.message}`);
        return null;
    }
}

// --- SỬA ĐỔI: Hàm parse JSON thông minh hơn (Hỗ trợ lọc thẻ <think> của DeepSeek) ---
function cleanAndParseJSON(text) {
    if (!text) return null;
    try {
        // 1. Log raw text để debug nếu cần (ẩn đi cho gọn, mở ra nếu muốn xem full)
        // log.dim(`Raw AI Response: ${text.substring(0, 100)}...`); 

        // 2. Xóa các thẻ <think>...</think> nếu có (đặc trưng của DeepSeek R1)
        let cleanText = text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();

        // 3. Tìm vị trí bắt đầu { và kết thúc }
        const firstBrace = cleanText.indexOf('{');
        const lastBrace = cleanText.lastIndexOf('}');
        
        if (firstBrace !== -1 && lastBrace !== -1) {
            // Chỉ lấy đúng đoạn JSON
            cleanText = cleanText.substring(firstBrace, lastBrace + 1);
        } else {
            // Không tìm thấy JSON structure
            log.warn(`⚠️ Không tìm thấy JSON trong phản hồi AI.`);
            return null;
        }

        // 4. Parse
        return JSON.parse(cleanText);
    } catch (e) {
        log.error(`JSON Parse Error: ${e.message} | Input snippet: ${text.substring(0, 50)}...`);
        return null;
    }
}

// --- NEW: DATA PREPARATION HELPERS ---
// Tạo mô tả nến cho "Strategist" (Model Llama)
function generateCandleDescription(klines) {
    // Lấy 5 nến gần nhất
    const last5 = klines.slice(-5);
    let desc = "Recent Price Action (Last 5 candles):\n";
    
    last5.forEach((k, index) => {
        const open = parseFloat(k[1]);
        const high = parseFloat(k[2]);
        const low = parseFloat(k[3]);
        const close = parseFloat(k[4]);
        
        const bodySize = Math.abs(close - open);
        const wickUpper = high - Math.max(open, close);
        const wickLower = Math.min(open, close) - low;
        const type = close > open ? "BULLISH" : "BEARISH";
        
        let shape = "Normal";
        if (bodySize < (high - low) * 0.1) shape = "Doji/Spinning Top";
        else if (wickUpper > bodySize * 2) shape = "Shooting Star/Wick Rejection Top";
        else if (wickLower > bodySize * 2) shape = "Hammer/Wick Rejection Bottom";
        else if (bodySize > (high - low) * 0.8) shape = "Marubozu/Strong Momentum";

        desc += `T-${4-index}: ${type} candle. Shape: ${shape}. Close: ${close}.\n`;
    });
    return desc;
}

// --- CORE: AI HUNTING LOGIC ---
async function huntForEntry() {
    if (state.position.active || state.pending.active) return;
    const now = Date.now();

    // Logic Cooldown - Sẽ chạy mượt mỗi giây nhờ POLL_SECONDS=5
    if (state.cooldownUntil && now < state.cooldownUntil) {
        const waitSec = Math.ceil((state.cooldownUntil - now) / 1000);
        // Luôn hiển thị (ghi đè dòng cũ)
        process.stdout.write(`\r\x1b[K${time()} ${paint(`⏳ Đang Cooldown sau khi bị Reject. Thử lại sau ${waitSec}s...`, C.dim)}`);
        return;
    }

    // Nếu hết cooldown, in một dòng trống để log tiếp theo không bị dính vào dòng cooldown cũ
    if (state.cooldownUntil && now >= state.cooldownUntil) {
        process.stdout.write('\n'); 
        state.cooldownUntil = 0;    	
    }
    
    // Cooldown mặc định giữa các lần gọi (để tránh spam API liên tục dù không bị reject)
    if (now - state.lastAnalysis < (G.POLL_SECONDS * 1000)) return;

    // 1. Lấy dữ liệu thị trường
    log.info('🤖 AI ARENA: Bắt đầu phiên phân tích đa luồng...');
    
    // Retry fetch klines
    const [klinesRes, klinesHTFRes] = await apiCallWithRetry(() => Promise.all([
        client.getKline({ category: CATEGORY, symbol: SYMBOL, interval: G.TIMEFRAME, limit: 1000 }),
        client.getKline({ category: CATEGORY, symbol: SYMBOL, interval: G.TREND_FILTER_TIMEFRAME, limit: 1000 })
    ]));

    if (klinesRes.retCode !== 0 || klinesHTFRes.retCode !== 0) return;

    const klines = klinesRes.result.list.sort((a, b) => a[0] - b[0]);
    const klinesHTF = klinesHTFRes.result.list.sort((a, b) => a[0] - b[0]); // Sắp xếp cũ -> mới

    // Tách mảng High/Low/Close để tính ATR và Indicator
    const closes = klines.map(k => parseFloat(k[4]));
    const highs = klines.map(k => parseFloat(k[2]));
    const lows = klines.map(k => parseFloat(k[3]));
    const volumes = klines.map(k => parseFloat(k[5])); // Lấy thêm Volume
    const closesHTF = klinesHTF.map(k => parseFloat(k[4]));

    const ind = getIndicators(closes, highs, lows);
    const indHTF = getIndicators(closesHTF);

    if (!ind || !indHTF) { log.warn('Chưa đủ dữ liệu tính chỉ báo.'); return; }

    // --- NEW: TÍNH TOÁN RELATIVE VOLUME (RVOL) ---
    // Tính trung bình volume 20 cây nến gần nhất
    const volSma20 = volumes.slice(-21, -1).reduce((a, b) => a + b, 0) / 20;
    const curVol = volumes[volumes.length - 1];
    const rvol = volSma20 > 0 ? (curVol / volSma20).toFixed(2) : 1.0; // RVOL > 1.5 là đột biến

    const curPrice = state.lastPrice;
    
    // Cập nhật %B realtime
    const bw = ind.bb.upper - ind.bb.lower;
    ind.bb.pb = bw === 0 ? 0.5 : (curPrice - ind.bb.lower) / bw;

    // --- NEW: LOG KIỂM TRA DỮ LIỆU ĐẦU VÀO ---
    log.info(paint(`📊 DỮ LIỆU KỸ THUẬT (M${G.TIMEFRAME}):`, C.cyan));
    log.info(`   • Price: ${curPrice} | EMA200: ${ind.ema.toFixed(2)} | Xu hướng: ${curPrice > indHTF.ema ? 'BULLISH (H4)' : 'BEARISH (H4)'}`);
    log.info(`   • RSI(14): ${ind.rsi.toFixed(2)} (${ind.rsi < 30 ? 'OVERSOLD' : ind.rsi > 70 ? 'OVERBOUGHT' : 'NEUTRAL'})`);
    log.info(`   • BB %B: ${ind.bb.pb.toFixed(2)} | Width: ${(ind.bb.width * 100).toFixed(2)}%`);
    log.info(`   • ATR: ${ind.atr.toFixed(2)}`);
    log.info(`   • RVOL: ${rvol} (Volume Strength) | ATR: ${ind.atr.toFixed(2)}`); // Log thêm RVOL

    const prompts = loadSystemPrompts();
    if (!prompts) return;

    // LẤY KEY ĐỘC LẬP
    const currentHfKey = getHfKey();
    const currentGoogleKey = getGoogleKey();

    // Mask key để log cho gọn
    const mask = (k) => k ? `...${k.substring(k.length-4)}` : 'NULL';
    log.dim(paint(`🔑 Session Keys -> HF: [${mask(currentHfKey)}] | Google: [${mask(currentGoogleKey)}]`, C.yellow));

    if (!currentHfKey && !currentGoogleKey) { log.error("Hết key để chạy!"); return; }

    // --- PHASE 1: THE QUANT (Logic & Math) ---
    // Model: DeepSeek (qua HF)
    log.dim(`🔹 Calling Agent 1: The Quant (${cfg.ai_agents.quant.model_id})...`);
    
    const quantInput = JSON.stringify({
        current_price: curPrice,
        indicators_m15: {
            rsi: ind.rsi.toFixed(2),
            bb_percent_b: ind.bb.pb.toFixed(2),
            bb_width: ind.bb.width.toFixed(4),
            atr: ind.atr.toFixed(2),
            rvol: rvol // Gửi thêm Volume cho Quant
        },
        macro_trend: curPrice > indHTF.ema ? "BULLISH_ABOVE_EMA200" : "BEARISH_BELOW_EMA200",
	strategy_params: {
            sl_atr_multiplier: S.STOP_LOSS_ATR_MULTIPLIER // Gửi 2.0
        }
    });

    const quantResText = await callHuggingFace(cfg.ai_agents.quant, prompts.quant, quantInput, currentHfKey);
    const quantJson = cleanAndParseJSON(quantResText);
    
    if (!quantJson) { log.warn('Quant Agent trả về lỗi hoặc không đúng JSON.'); return; }
    log.ai(`Quant Output: ${quantJson.signal} (${quantJson.probability}%) | Lý do: ${quantJson.statistical_reason}`);

    // --- PHASE 2: THE STRATEGIST (Price Action) ---
    // Model: Llama 3 (qua HF)
    log.dim(`🔹 Calling Agent 2: The Strategist (${cfg.ai_agents.strategist.model_id})...`);
    
    const candleDesc = generateCandleDescription(klines);
    const stratInput = `
    CURRENT PRICE: ${curPrice}
    MACRO CONTEXT: ${curPrice > indHTF.ema ? "Uptrend" : "Downtrend"}
    VOLUME CONTEXT: RVOL is ${rvol} (Normal is 1.0, High is > 1.5).
    CANDLE DATA:
    ${candleDesc}
    `;

    const stratResText = await callHuggingFace(cfg.ai_agents.strategist, prompts.strategist, stratInput, currentHfKey);
    const stratJson = cleanAndParseJSON(stratResText);

    if (!stratJson) { log.warn('Strategist Agent trả về lỗi.'); return; }
    log.ai(`Strategist Output: ${stratJson.bias} | Structure: ${stratJson.structure_phase} | Trap?: ${stratJson.is_trap_suspected}`);

    // --- PHASE 3: THE RISK MANAGER (Final Judge) ---
    // Model: Gemini (qua Google)
    log.dim(`🔹 Calling Agent 3: The Risk Manager (${cfg.ai_agents.risk_manager.model_id})...`);

    const riskInput = JSON.stringify({
        account_info: {
            balance: state.capital,
            current_pnl: 0, // Chưa có lệnh mở
        },
        proposal_quant: quantJson,
        proposal_strategist: stratJson,
        market_volatility: {
            atr: ind.atr,
            is_high_volatility: ind.bb.width > (cfg.risk_management.volatility_threshold_atr_multiplier * 0.01) 
        },
	strategy_params: {
             target_reward_risk_ratio: S.TAKE_PROFIT_R_MULTIPLE 
        }
    });

    const riskResText = await callGemini(cfg.ai_agents.risk_manager, prompts.risk, riskInput, currentGoogleKey);
    const riskJson = cleanAndParseJSON(riskResText);

    state.lastAnalysis = now;

    if (!riskJson) { log.warn('Risk Manager không phản hồi.'); return; }

    // --- EXECUTION LOGIC ---
    if (riskJson.final_decision === 'APPROVED') {
        const side = riskJson.approved_side;
        if (side !== 'LONG' && side !== 'SHORT') return;

        log.ai(paint(`🏆 PHÁN QUYẾT CUỐI CÙNG: APPROVED ${side}`, C.green));
        log.ai(`📝 Lý do: ${riskJson.reasoning}`);

        let entry = parseFloat(riskJson.final_entry) || curPrice;
        let sl = parseFloat(riskJson.final_sl);

        // Tinh chỉnh Entry để đảm bảo không FOMO
        if (side === 'LONG' && entry > curPrice) entry = curPrice - 1.25 * parseFloat(state.filters.tickSize);
        if (side === 'SHORT' && entry < curPrice) entry = curPrice + 1.25 * parseFloat(state.filters.tickSize);

	// Nếu giá Entry AI đưa ra quá xa giá hiện tại (AI bị lag dữ liệu cũ), hãy dùng giá hiện tại
        if (Math.abs(entry - curPrice) / curPrice > 0.01) {
             entry = curPrice;
        }
        
        entry = parseFloat(roundPrice(entry));
        sl = parseFloat(roundPrice(sl));

        // Tính Size
        const riskPerShare = Math.abs(entry - sl);
        if (riskPerShare <= 0) { log.warn('SL không hợp lệ (trùng Entry).'); return; }

        const riskAmount = state.capital * S.RISK_PER_TRADE_PERCENT * (riskJson.position_size_modifier || 1.0);
        let qty = Math.floor((riskAmount / riskPerShare) / parseFloat(state.filters.qtyStep)) * parseFloat(state.filters.qtyStep);

        // Check Min Qty
        if (qty < parseFloat(state.filters.minQty)) {
            log.warn(`Risk ${riskAmount}$ quá nhỏ -> Qty ${qty} < Min ${state.filters.minQty}. Bỏ lệnh.`);
            return;
        }

        // Đặt lệnh
        const qtyStr = qty.toFixed(state.filters.qtyStep.split('.')[1]?.length || 0);
        
        log.trade(`🚀 Executing ${side} | Qty: ${qtyStr} | Entry: ${entry} | SL: ${sl}`);
        
        const order = await client.submitOrder({
            category: CATEGORY, symbol: SYMBOL, side: side === 'LONG' ? 'Buy' : 'Sell',
            orderType: 'Limit', qty: qtyStr, price: String(entry), stopLoss: String(sl),
            timeInForce: 'GTC', positionIdx: 0
        });

        if (order.retCode === 0) {
            state.pending = {
                active: true, side: side === 'LONG' ? 'Buy' : 'Sell',
                qty: qty, entryPrice: entry, initialSL: sl, 
                initialR: riskPerShare, reason: riskJson.reasoning, 
                orderId: order.result.orderId
            };
	    state.cooldownUntil = 0;
            saveState();
        } else {
            log.error(`Đặt lệnh thất bại: ${order.retMsg}`);
        }

    } else {
        log.dim(paint(`🛑 REJECTED: ${riskJson.reasoning}`, C.red)); 
        const cooldownMs = (G.COOLDOWN_AFTER_REJECT_SECONDS) * 1000;
        state.cooldownUntil = Date.now() + cooldownMs;
        log.warn(`⏳ Kích hoạt Cooldown: Bot sẽ tạm dừng phân tích trong ${G.COOLDOWN_AFTER_REJECT_SECONDS}s.`);
        saveState(); 
    }
}

// --- MAIN LOOP (GIỮ NGUYÊN KHUNG, CHỈ SỬA GỌI HÀM) ---
let isProcessing = false;
async function main() {
    log.info(paint('=== BYBIT AI ARENA BOT (Multi-Agent: Quant + Strategist + Risk) ===', C.bold));

    const inst = await client.getInstrumentsInfo({ category: CATEGORY, symbol: SYMBOL });

    if (inst.retCode !== 0 || !inst.result.list || inst.result.list.length === 0) {
        log.error(`❌ Không lấy được thông tin cặp coin ${SYMBOL}. Dừng bot.`);
        process.exit(1);
    }

    const f = inst.result.list[0];
    state.filters = { 
        tickSize: f.priceFilter.tickSize, 
        qtyStep: f.lotSizeFilter.qtyStep,
        minQty: f.lotSizeFilter.minOrderQty, 
        minNotional: f.lotSizeFilter.minNotionalValue || '5'
    };

    await client.setLeverage({ category: CATEGORY, symbol: SYMBOL, buyLeverage: String(S.LEVERAGE), sellLeverage: String(S.LEVERAGE) }).catch(()=>{});
    loadState();
    await syncState();

    const run = async () => {
        if (isProcessing) return;
        isProcessing = true; 
        try {
            await syncState();
            await manageTrailingStop();
            if (state.pending.active && state.pending.initialR > 0) {
                const dist = Math.abs(state.lastPrice - state.pending.entryPrice);
                if (dist > state.pending.initialR * 3.0) { 
                    log.warn(`Hủy lệnh chờ do giá chạy xa.`);
                    await client.cancelOrder({ category: CATEGORY, symbol: SYMBOL, orderId: state.pending.orderId });
                }
            }
            await huntForEntry();
        } catch (e) {
            log.error(`Loop Error: ${e.message}`);
        } finally {
            isProcessing = false;
        }
    };

    await run();
    setInterval(run, G.POLL_SECONDS * 1000);
}

main();