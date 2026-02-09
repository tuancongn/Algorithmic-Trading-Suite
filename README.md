# 🤖 AI-Driven Algorithmic Trading System

> **A robust, full-cycle quantitative trading system featuring Python-based backtesting (VectorBT) and a Node.js execution engine powered by Multi-Agent LLMs (Gemini & Hugging Face).**

## 📌 Project Overview

This project represents a complete workflow for algorithmic cryptocurrency trading, specifically designed for the Bybit Futures market. It demonstrates the bridge between **Data Science** (Strategy Research) and **Software Engineering** (Live Execution).

The system is composed of two core modules:
1.  **The Research Engine (Python):** Uses `VectorBT` and `Pandas` for high-performance backtesting, parameter optimization (Grid Search), and risk modeling.
2.  **The Live Engine (Node.js):** An event-driven bot that manages real-time state, handles WebSocket data, and employs a **Multi-Agent AI architecture** to validate trades before execution.

---

## 🛠 Tech Stack

### Quant & Backtesting (Python)
* **VectorBT:** For vectorized backtesting of millions of data points in seconds.
* **Pandas & NumPy:** Data manipulation and statistical analysis.
* **Pybit:** Data fetching from Bybit API.
* **Technical Analysis:** Custom implementation of ATR, RSI, ADX, Bollinger Bands, and Moving Averages.

### Live Execution (Node.js)
* **Runtime:** Node.js (Asynchronous Event Loop).
* **API Integration:** `bybit-api` (V5), Google Gemini API, Hugging Face Inference API.
* **Logic:** State Machine pattern for order management (Entry/Exit/Trailing Stop).
* **Reliability:** Implemented Retry mechanisms with exponential backoff and global error handling to prevent crashes.

---

## 🧠 AI Multi-Agent Architecture

Unlike standard trading bots that rely solely on hard-coded indicators, this system utilizes a **"Council of Agents"** approach to reduce false signals. The Node.js engine orchestrates three distinct AI roles:

1.  **The Quant (DeepSeek/HF):** Analyzes pure numerical data (RSI, %B, Volatility) to detect statistical anomalies.
2.  **The Strategist (Llama 3/HF):** Analyzes Price Action and Candle Patterns (Hammer, Doji, Marubozu) to understand market sentiment.
3.  **The Risk Manager (Google Gemini):** Acts as the final judge. It aggregates inputs from the Quant and Strategist, checks account health, and issues the final `APPROVED` or `REJECTED` verdict along with dynamic position sizing.

---

## ✨ Key Features

### 1. Robust Backtesting & Optimization
* Fetches historical data efficiently with pagination handling.
* Implements **Compound Interest** logic in backtesting.
* **Grid Search Optimization:** Automatically finds the best parameters (EMA period, RSI thresholds, SL multipliers) and saves them to `op_config.json`.
* **Excel Reporting:** Exports detailed trade logs, equity curves, and exposure analysis.

### 2. Intelligent Execution Engine
* **State Persistence:** Uses `bot_state.json` to survive restarts/crashes without losing track of active positions.
* **Smart Order Routing:** Adjusts entry prices to ensure Maker orders (when possible) or strictly defined Limit orders.
* **Dynamic Trailing Stop:** Implements a step-based trailing stop logic to lock in profits as the trend progresses (R-multiple based).
* **Anti-Knife Catching:** Logic to prevent buying into a "Waterfall" crash using Bollinger Band %B and Volatility filters.
* **Real-time Dashboard:** Custom CLI interface showing PnL, Price, and AI thoughts in real-time.

---

## 🚀 How It Works

### Step 1: Research (Python)
Run the backtest to find optimal parameters for the current market regime.
### Step 2: Live Trading Setup (Node.js)

Once the strategy parameters are optimized, the Node.js engine takes over for real-time execution.

1. Environment Configuration:
Create a .env file to store sensitive credentials (API Keys are managed securely and never hardcoded).

BYBIT_KEY=your_bybit_key
BYBIT_SECRET=your_bybit_secret
GEMINI_API_KEY=your_google_ai_key

2. Smart Key Rotation (Advanced):
To bypass rate limits and ensure high availability for AI inference, the system utilizes a key pooling mechanism (keyfull.txt). The bot automatically rotates between multiple HuggingFace and Google API keys to maintain 100% uptime during high-frequency analysis.

3. Launch the Engine:
# Run the Standard Hybrid Bot
node go.js

# OR: Run the Advanced Multi-Agent Bot (Quant + Strategist + Risk)
node "go - Copy.js"

🧠 The AI Multi-Agent Architecture

This system implements a "Council of Agents" pattern to solve the common problem of false signals in algorithmic trading. Instead of a single model making decisions, three specialized AI agents collaborate:

1. 📊 The Quant Agent (DeepSeek via HuggingFace)

- Role: Pure Mathematical Analyst.
- Input: RSI, Bollinger Band %B, ATR, EMA Divergence.
- Logic: Calculates probabilities based on statistical anomalies (e.g., "Price is 2 standard deviations below mean, but Momentum is diverging").
- Output: Statistical Signal & Confidence Score.

2. 🕯️ The Strategist Agent (Llama 3 via HuggingFace)
- Role: Price Action Specialist.
- Input: OHLC Candle patterns (last 5 candles), Market Structure (Higher Highs/Lower Lows).
- Logic: Identifies patterns like "Bullish Engulfing," "Pin Bar Rejection," or "Liquidity Sweeps."
- Output: Market Bias & Structural Analysis.

3. 🛡️ The Risk Manager (Google Gemini Pro)
- Role: The Supreme Judge.
- Input: Proposals from Quant & Strategist + Account Equity + Volatility metrics.
- Logic: * Validates if the trade aligns with the current macro trend.
  a. Checks if the Stop Loss distance exceeds the max risk threshold.
  b. Circuit Breaker: Rejects trades if volatility indicates a "Waterfall" crash (Anti-Knife Catching logic).
- Output: APPROVED / REJECTED decision + Dynamic Position Sizing.

🏗️ Engineering Highlights

This project demonstrates production-grade software engineering practices:
- Robust Error Handling & Retry Logic: Implemented apiCallWithRetry wrapper for all network requests. If the Exchange API or AI Model times out (503/504), the system performs exponential backoff retries instead of crashing.
- State Persistence & Crash Recovery: The bot maintains its state in bot_state.json. If the server restarts unexpectedly, the bot re-syncs with the exchange, detects active positions, and resumes management without manual intervention.
- Event-Driven & Asynchronous: Built on Node.js non-blocking I/O to handle WebSocket streams and REST polling simultaneously without lag.
- Real-time CLI Dashboard: A custom console UI that renders PnL (Profit and Loss), active orders, and AI reasoning logs in real-time using ANSI escape codes for better observability.

📂 Project Structure
.
├── Backtesting.py        # Python VectorBT Engine (Research & Optimization)
├── go.js                 # Production Trading Bot (Node.js)
├── trading.py            # Python-based alternative execution engine
├── config.json           # Trading configurations (SL/TP/Trailing)
├── bot_state.json        # Runtime state persistence
├── smc_prompt.txt        # System Prompts for AI Agents
└── utils/                # Helper functions for Indicators & Logging

🚀 Future Improvements
- Dockerize the application for easier deployment on AWS/GCP.
- Implement a Webhook listener to trigger trades from TradingView alerts.
- Add a Redis layer for caching high-frequency market data.

📬 Contact
I am a software engineer passionate about FinTech, Data Science, and Backend Systems. I built this project to demonstrate my ability to bridge complex financial logic with robust code.

GitHub: https://github.com/tuancongn
Email: lapnhan3@gmail.com

python Backtesting.py
# Output: Returns s
harpe ratio, win rate, and generates vectorbt_optimized.xlsx
