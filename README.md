# 🤖 AI-Driven Quantitative Trading System (High-Frequency & Swing)
> **"A comprehensive algorithmic trading ecosystem combining Technical Analysis, Ensemble Machine Learning (XGBoost/LightGBM), and Neuro-Symbolic LLM reasoning."**

---

## 📖 **Introduction**
This project represents a sophisticated, modular **cryptocurrency trading ecosystem** designed for the **Bybit Futures** market. Unlike traditional bots that rely solely on rigid if-then indicators, this system introduces a **Neuro-Symbolic architecture**. It fuses **mathematical precision** (RSI, Bollinger Bands, ATR) with the **contextual reasoning** capabilities of Large Language Models (LLMs) to detect market regimes, manage risk dynamically, and execute trades with precision.

The repository consists of two main pillars:
1.  **🟢 Production Trading Engine (Node.js)**: A multi-agent system managing live orders, trailing stops, and AI inference.
2.  **🔵 Quantitative Research Lab (Python)**: A high-performance backtesting suite using `vectorbt` for strategy validation and parameter optimization.

---

## 🚀 **Key Features**
### 🧠 **1. Multi-Agent AI Architecture (Live Engine)**
The core logic (found in `go.js`) operates based on a **"Committee of Experts"** model:
*   **The Quant (DeepSeek/HuggingFace)**: Analyzes raw numerical data (RSI, %B, ATR) to calculate statistical probabilities.
*   **The Strategist (Llama 3)**: Interprets Price Action, candlestick patterns (Hammer, Shooting Star), and macro trends.
*   **The Risk Manager (Google Gemini)**: Acts as the final judge. It synthesizes inputs from the Quant and Strategist, checks account health, and issues the final **APPROVED** or **REJECTED** verdict along with position sizing adjustments.

### 🛡️ **2. Institutional-Grade Risk Management**
Safety is prioritized over profit. The system implements:
*   **Dynamic Position Sizing**: Calculates lot size based on account balance and specific invalidation points (Stop Loss distance) to maintain consistent R-risk.
*   **Anti-Knife Catching**: Logic to block "Long" signals during "Waterfall" market crashes (e.g., when Price < Lower Band & Momentum is high).
*   **Volatility Guards**: Automatic cooldown periods after rejected trades to prevent over-trading.
*   **Trailing Stop**: An intelligent step-based trailing mechanism to lock in profits (R-multiples) as the trend develops.

### 📊 **3. High-Performance Backtesting**
The Python module (`Backtesting.py`) ensures strategies are mathematically sound before deployment:
*   **Vectorized Testing**: Uses `vectorbt` to simulate thousands of candles in seconds.
*   **Grid Search Optimization**: Automatically iterates through combinations of parameters (EMA periods, RSI thresholds, SL multipliers) to find the **"Global Maxima"** for profitability.
*   **Detailed Reporting**: Generates Excel reports with Equity Curves, Sharpe Ratios, and Drawdown analysis.

### 🤖 **4. Advanced ML Signal Generator (gemini.py)**
A dedicated **Machine Learning research module** designed to predict short-term price movements with high statistical confidence:

*   **🧠 Ensemble Learning**: Aggregates predictions from 5 distinct models (LightGBM, XGBoost, Neural Networks, Random Forest, Gradient Boosting) to reduce variance and overfitting.
*   **🔄 Walk-Forward Optimization**: Implements strict time-series cross-validation (Train/Val/Test splits) to simulate real-world performance and eliminate look-ahead bias.
*   **🏷️ Market Regime Awareness**: Automatically classifies market states (Bull, Bear, Sideways) using Momentum & Trend strength to adjust position sizing dynamically.
*   **🔧 Advanced Feature Engineering**: Computes **38+ indicators** including Kaufman Efficiency Ratio, Volatility-Scaled Returns, and Shadow Asymmetry to capture hidden market microstructure.

---

## 🛠️ **Tech Stack & Architecture**
**Live Execution (Node.js)**
*   **Runtime**: Node.js
*   **Exchange Connection**: `bybit-api` (V5 Unified Trading)
*   **Analysis**: `trading-signals` (EMA, RSI, BollingerBands, ATR)
*   **AI Integration**: REST API calls to Google Gemini & HuggingFace Inference Endpoints.
*   **State Management**: JSON-based persistence for crash recovery.
*   **Utils**: Robust Retry Logic (Exponential Backoff), Error Handling, and Console Dashboard.

**Quantitative Research (Python)**
*   **Core**: Python 3.10+
*   **Data Science**: `pandas`, `numpy`, `scikit-learn`
*   **Machine Learning**: `LightGBM`, `XGBoost`, `TensorFlow/Keras` (Optional)
*   **Backtesting**: `vectorbt` (Vectorized Backtesting)
*   **Exchange**: `pybit` (Data fetching)

---

## 🧩 **System Logic Visualization**
```
graph TD
    A[Market Data Stream] -->|WebSockets/REST| B(Technical Indicators)
    B --> C{Market Regime Check}
    C -->|Trending| D[Trend Following Logic]
    C -->|Sideways| E[Mean Reversion Logic]
    D & E --> F[AI Context Analysis]
    F -->|Prompt Engineering| G[LLM Committee]
    G -->|JSON Response| H[Risk Manager Module]
    H -->|Approved| I[Order Execution Engine]
    H -->|Rejected| J[Cooldown System]
    I --> K[Position Management & Trailing Stop]
```
> *Note: The above is a conceptual representation of the flow implemented in `go.js`.*

---

## 📂 **Project Structure**
```
├── 🟢 Node.js (Live Trading)
│   ├── go.js               # Main entry point (Hybrid Strategy)
│   ├── config.json         # Strategy parameters & API keys
│   ├── prompt.txt          # System prompts for AI Agents
│   └── trade_history.csv   # Trade logging
│
├── 🔵 Python (Research & ML)
│   ├── Backtesting.py      # VectorBT simulation (Indicator-based)
│   ├── gemini.py           # ML Ensemble Strategy (LightGBM/XGBoost/NN)
│   ├── trading.py          # Lightweight Python execution script
│   └── requirements.txt    # Python dependencies
│
├── 📄 README.md            # Project documentation
└── 📜 .gitignore           # Git ignore file
```

---

## 💡 **Engineering Challenges Solved**
Throughout the development of this project, several complex challenges were addressed:

1.  **🚦 API Rate Limiting**: Implemented a `KeyPool` system to rotate between API keys and smart retry logic with exponential backoff to handle HTTP 429/503 errors.
2.  **🤖 LLM Hallucinations**: Enforced strict JSON output schemas and added a validation layer (`cleanAndParseJSON`) to ensure the AI's output is always machine-readable and executable.
3.  **⚡ Latency**: Optimized the critical path by running lightweight indicator calculations locally and only calling external AI APIs when a high-probability setup is detected.
4.  **🤕 Crash Recovery**: Designed a state file (`bot_state.json`) mechanism that allows the bot to resume managing active positions seamlessly after a restart.

---

## 📈 **Performance & Results**
*(You can add a screenshot of your equity curve or terminal output here)*

*   **Strategy**: Trend Following with Mean Reversion filters.
*   **Win Rate**: ~[82]% (Based on backtesting data).
*   **Profit Factor**: > 1.5.
*   **Risk Profile**: Fixed % Risk per Trade (No Martingale).

---

## 👨‍💻 **About Me**
I am a developer with a passion for **FinTech** and **Automated Systems**. I enjoy solving complex problems where software engineering meets financial markets. This project demonstrates my ability to:
*   Build robust, fault-tolerant backend systems.
*   Integrate state-of-the-art AI into practical applications.
*   Analyze data quantitatively to drive decision-making.

---

## 🤝 **Contact**
*   **Full Name**: Nguyen Cong Lap Nhan
*   **Phone**: `0799302502`
*   **Email**: `lapnhan3@gmail.com`
*   **GitHub**: [https://github.com/tuancongn](https://github.com/tuancongn)
