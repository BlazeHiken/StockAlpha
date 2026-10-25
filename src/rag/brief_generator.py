import os
import logging
from google import genai
from google.genai import types
from src.quant.models import PortfolioResult, BenchmarkResult
from src.rag.retriever import retrieve_research_for_portfolio

logger = logging.getLogger(__name__)

DISCLAIMER = "This analysis is generated for educational purposes only and does not constitute financial advice."

_PCT_ROWS = ("Return", "Volatility", "Max Drawdown")

def _fmt_metric(name: str, value: float) -> str:
    if any(key in name for key in _PCT_ROWS):
        return f"{float(value) * 100:.2f}%"
    return f"{float(value):.3f}"

def format_portfolio_data(opt_result: PortfolioResult, benchmark_results: BenchmarkResult | None = None) -> str:
    weights = [f"{ticker}: {weight*100:.2f}%" for ticker, weight in opt_result.weights.items() if weight > 0.005]

    def block(title: str, m) -> str:
        if m is None:
            return f"{title}: not available"
        return (f"{title}:\n"
                f"- Return (annualized): {m.expected_return*100:.2f}%\n"
                f"- Volatility (annualized): {m.volatility*100:.2f}%\n"
                f"- Sharpe Ratio: {m.sharpe_ratio:.3f}\n"
                f"- Max Drawdown: {m.max_drawdown*100:.2f}%")

    text = (
        "Allocations (Weights > 0.5%):\n" + "\n".join(weights) + "\n\n"
        + block("In-Sample (training period) metrics of the optimized portfolio", opt_result.in_sample_metrics) + "\n\n"
        + block("Out-of-Sample (test period) metrics of the optimized portfolio", opt_result.out_of_sample_metrics)
    )

    if benchmark_results is not None:
        df = benchmark_results.comparison_df
        lines = []
        for metric in df.index:
            cells = ", ".join(f"{col}: {_fmt_metric(metric, df.loc[metric, col])}" for col in df.columns)
            lines.append(f"- {metric} -> {cells}")
        text += "\n\nBenchmark comparison (IS = in-sample, OOS = out-of-sample):\n" + "\n".join(lines)
    return "\n" + text + "\n"

def generate_portfolio_brief(opt_result: PortfolioResult, benchmark_results: BenchmarkResult | None = None) -> str | None:
    if not os.environ.get("GEMINI_API_KEY"):
        return "⚠️ **GEMINI_API_KEY not found**. Please add it to your `.env` file to enable AI-powered portfolio briefs."

    try:
        tickers = [ticker for ticker, weight in opt_result.weights.items() if weight > 0.005]
        research_snippets = retrieve_research_for_portfolio(tickers)
        context = "\n\n".join(research_snippets)
        if not context.strip():
            context = "No specific qualitative research available for these tickers in the local corpus."

        portfolio_str = format_portfolio_data(opt_result, benchmark_results)

        prompt = f"""
You are a neutral financial data analyst writing an explanatory note. You are not an advisor, manager or broker, and you must not speak in the first person.
Your task is to explain the results of a portfolio optimization to a non-technical reader.

### 1. The Quantitative Output (Max Sharpe Optimization, including benchmark comparison)
{portfolio_str}

### 2. Qualitative Research Context
{context}

### Instructions
Write a brief, professional, objective, and easy-to-understand explanation of the portfolio.

- Explain the rationale behind the allocation by synthesizing the quantitative results with the provided qualitative research.
- Clearly distinguish between **In-Sample (IS)** optimization performance and **Out-of-Sample (OOS)** test performance.
- Evaluate the portfolio's OOS performance in context by comparing it with the provided **Equal-Weight** and **NIFTY 50** benchmark results.
- Do not judge the portfolio solely by whether its absolute return or Sharpe Ratio is positive or negative. Explain whether it performed better or worse than the provided benchmarks.
- Explain why stocks received relatively high allocations based on the quantitative results and, where supported, the provided research context.
- Do not assume that the optimized portfolio is successful, stable, efficient, or superior. If the OOS results reveal weaknesses or deterioration from IS performance, state this clearly.
- Do not just repeat the numbers; explain what they mean in practical, plain language.
- Keep the tone professional and objective. Do not use promotional, reassuring, or investment-advisory language.
- Use markdown formatting (headings, bullet points, and bold text) for readability.

### Research Grounding
- Use ONLY the provided qualitative research context for company-specific claims.
- Do not introduce company-specific facts that are not supported by the provided research.
- If the research context does not support a claim, explicitly state that the available research does not establish it.
- Do not infer fundamental characteristics such as cash-flow strength, management quality, growth drivers, or sector advantages unless they are supported by the provided research.
- Do not invent or fabricate citations.
- Cite research-backed claims using the exact `[Source: filename]` format provided in the research context.

### Important
The purpose of this brief is to **explain the optimizer's results objectively, not to justify or promote the portfolio**. If the results are unfavorable, say so. If the portfolio outperformed the benchmarks despite having a negative absolute return or Sharpe Ratio, explain that distinction clearly.

### Hard Constraints
- Never generate, suggest, or imply new portfolio weights or allocation changes.
- Never modify, adjust, or reinterpret the optimizer's numerical outputs.
- Never predict future stock prices or forecast future performance.
- Never recommend or imply that the user should buy, sell, or hold any stock or the portfolio.
- Keep all discussion focused on explaining the supplied optimizer results and research context.
- Always include this exact disclaimer at the end of the brief: "This analysis is generated for educational purposes only and does not constitute financial advice."

Begin your response directly with the brief.
"""

        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
        )

        text = response.text or ""
        if DISCLAIMER not in text:
            text = text.rstrip() + "\n\n*" + DISCLAIMER + "*"
        return text

    except Exception as e:
        logger.error(f"Error generating brief: {e}")
        return f"⚠️ **Error generating AI brief**: {str(e)}\n\nPlease ensure your API key is valid and you have internet connectivity."