# ===============================
# G3 宏观资金雷达系统 - 生产最终版
# ===============================
# 支持：
# - 宏观状态识别 S1-S4
# - AI策略解读（OpenAI API）
# - 图表生成
# - CSV导出
# - 周报PDF生成
# - 多微信(Server酱 Turbo)推送
# - GitHub Actions 定时运行
# - GitHub Pages Web Dashboard 自动更新
# ===============================

import os
import io
import json
import requests
import datetime
from pathlib import Path

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fredapi import Fred
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from openai import OpenAI

# ========== 环境变量 ==========
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SERVER_KEYS = os.getenv("SERVERCHAN_KEYS")  # 多个用逗号分隔
FRED_KEY = os.getenv("FRED_API_KEY")

if not OPENAI_KEY:
    raise RuntimeError("❌ 缺少 OPENAI_API_KEY")
if not SERVER_KEYS:
    raise RuntimeError("❌ 缺少 SERVERCHAN_KEYS")
if not FRED_KEY:
    raise RuntimeError("❌ 缺少 FRED_API_KEY")

client = OpenAI(api_key=OPENAI_KEY)
fred = Fred(api_key=FRED_KEY)

# ========== 目录 ==========
OUTPUT_DIR = Path("outputs")
WEB_DIR = Path("docs/data")

OUTPUT_DIR.mkdir(exist_ok=True)
WEB_DIR.mkdir(parents=True, exist_ok=True)

# ========== 仓位模板 ==========
PORTFOLIO_TEMPLATE = {
    "S1": {"Stocks": "60%", "BTC": "20%", "Gold": "10%", "Cash": "10%"},
    "S2": {"Stocks": "40%", "BTC": "20%", "Gold": "20%", "Cash": "20%"},
    "S3": {"Stocks": "20%", "BTC": "10%", "Gold": "30%", "Cash": "40%"},
    "S4": {"Stocks": "0%", "BTC": "0%", "Gold": "40%", "Cash": "60%"},
}

# ========== 数据获取 ==========
def get_market_data():
    print("📥 拉取市场数据...")

    sp500 = yf.download("^GSPC", period="6mo", interval="1d")[["Close"]]
    btc = yf.download("BTC-USD", period="6mo", interval="1d")[["Close"]]
    gold = yf.download("GC=F", period="6mo", interval="1d")[["Close"]]

    dxy = fred.get_series("DTWEXBGS")
    rates = fred.get_series("DFF")

    dxy = pd.DataFrame(dxy, columns=["Close"])
    rates = pd.DataFrame(rates, columns=["Close"])

    sp500.columns = ["SP500"]
    btc.columns = ["BTC"]
    gold.columns = ["GOLD"]
    dxy.columns = ["DXY"]
    rates.columns = ["RATES"]

    df = pd.concat(
        [sp500, btc, gold, dxy, rates],
        axis=1,
        join="inner"
    ).dropna()

    return df

# ========== 指数计算 ==========
def compute_indices(df):
    returns = df.pct_change().dropna()

    li = returns["GOLD"].mean() - returns["DXY"].mean()
    ri = returns["BTC"].std() + returns["SP500"].std()

    return round(float(li), 4), round(float(ri), 4)

# ========== 状态判断 ==========
def classify_state(li, ri):
    if li > 0.5 and ri < 0.5:
        return "S1"
    elif li > 0 and ri >= 0.5:
        return "S2"
    elif li <= 0 and ri >= 0.5:
        return "S3"
    else:
        return "S4"

# ========== 转换概率 ==========
def transition_probability(li, ri):
    score = abs(li) * 0.6 + abs(ri) * 0.4
    return round(min(95, score * 100), 2)

# ========== 图表 ==========
def generate_chart(df):
    path = OUTPUT_DIR / "market_chart.png"

    df[["SP500", "BTC", "GOLD"]].tail(60).plot(figsize=(10, 5))
    plt.title("Market Trend - Last 60 Days")
    plt.grid(True)
    plt.savefig(path)
    plt.close()

    return str(path)

# ========== CSV ==========
def export_csv(df):
    path = OUTPUT_DIR / "market_data.csv"
    df.to_csv(path)
    return str(path)

# ========== PDF ==========
def generate_pdf(report_text):
    path = OUTPUT_DIR / "weekly_report.pdf"

    styles = getSampleStyleSheet()
    pdf = SimpleDocTemplate(str(path))

    elements = []
    for line in report_text.split("\n"):
        elements.append(Paragraph(line, styles["Normal"]))
        elements.append(Spacer(1, 12))

    pdf.build(elements)
    return str(path)

# ========== AI解读 ==========
def ai_macro_analysis(raw_data):
    print("🤖 请求AI分析中...")

    prompt = f"""
你是华尔街宏观策略师，请解读以下市场数据并给出专业投资建议：

{raw_data}

请输出：
1. 当前资金流向判断
2. 风险等级评估
3. 黄金 / 美股 / 加密货币策略
4. 未来7天关键观察点
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

# ========== 微信推送 ==========
def send_wechat(title, content):
    keys = SERVER_KEYS.split(",")

    for key in keys:
        url = f"https://sctapi.ftqq.com/{key.strip()}.send"
        data = {
            "title": title,
            "desp": content
        }
        r = requests.post(url, data=data, timeout=15)
        print("📨 微信状态:", r.status_code)

# ========== Web Dashboard导出 ==========
def export_dashboard_data(li, ri, state, prob, portfolio, summary):
    payload = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "li": round(li, 4),
        "ri": round(ri, 4),
        "state": state,
        "transition": round(prob, 2),
        "portfolio": portfolio,
        "summary": summary
    }

    path = WEB_DIR / "latest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("🌐 Dashboard 数据已导出:", path)

# ========== 主引擎 ==========
def run_engine():
    print("📡 G3 宏观雷达系统启动")

    df = get_market_data()

    li, ri = compute_indices(df)
    state = classify_state(li, ri)
    prob = transition_probability(li, ri)

    chart = generate_chart(df)
    csv_file = export_csv(df)

    raw_report = f"""
时间: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

状态: {state}
流动性指数 LI: {li}
风险指数 RI: {ri}
转换概率: {prob}%

推荐仓位:
{PORTFOLIO_TEMPLATE[state]}
"""

    ai_text = ai_macro_analysis(raw_report)
    pdf_path = generate_pdf(ai_text)

    full_msg = f"""
📊 G3 宏观资金雷达日报

{raw_report}

🧠 AI策略解读:
{ai_text}

📎 本地文件:
图表: {chart}
数据: {csv_file}
周报PDF: {pdf_path}
"""

    send_wechat("📡 G3 宏观雷达日报", full_msg)

    export_dashboard_data(
        li,
        ri,
        state,
        prob,
        PORTFOLIO_TEMPLATE[state],
        ai_text
    )

    print("✅ 系统运行完成")

# ========== 程序入口 ==========
if __name__ == "__main__":
    run_engine()
