# StockAlpha 📈

**StockAlpha** is an AI-Augmented Stock Portfolio Optimization Engine. It applies Modern Portfolio Theory (MPT) to compute mathematically optimal portfolios from real historical market data, benchmarks them against naive allocation strategies using risk-adjusted metrics, and layers a Retrieval-Augmented Generation (RAG) system on top to generate plain-language, research-grounded explanations of the results.

## Features

1. **Portfolio Optimizer (Core)**: Computes the optimal allocation (maximum return for minimum risk, max Sharpe ratio) and benchmarks it against a naive equal-weight split using historical data from `yfinance`.
2. **AI-Explained Portfolio Brief (Core)**: Uses a local RAG pipeline to generate a grounded, plain-language explanation of the optimizer's results based on local research notes.
3. **Conversational Q&A (Stretch)**: Chat interface for users to ask follow-up questions about their generated portfolio.

## Technical Architecture

- **Data Ingestion**: `yfinance`, stored locally as Parquet
- **Optimization Core**: `pandas`, `numpy`, `scipy.optimize`
- **RAG / AI Layer**: `chromadb` (vector store), `google-generativeai` (Gemini embedding and chat models)
- **Dashboard**: `streamlit`

## Getting Started

### Prerequisites

Ensure you have Python 3.10+ installed and a virtual environment set up. This project expects the virtual environment to be located in the parent directory, named `aiml` (e.g., `../aiml/.venv`).

### Installation

1. Activate your virtual environment:
   ```bash
   # Windows
   ..\aiml\.venv\Scripts\activate
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Create a `.env` file in the root directory to store your API keys. You can use `.env.example` as a template if one is provided.
```env
GEMINI_API_KEY=your_api_key_here
```

### Running the Code

**Current Status:** Data Ingestion Module (UC1 Step 1) completed.

You can test fetching and saving historical stock data to a local Parquet file by running:
```bash
python src/quant/ingest.py
```
This script will fetch the last 3 years of data for a preset list of tickers, handle any missing data, and generate `example_prices.parquet`.
