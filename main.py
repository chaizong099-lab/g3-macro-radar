# ===============================
# G3 宏观资金雷达系统 - 生产版
# 功能：
# - 宏观状态识别 S1-S4
# - OpenAI AI策略解读
# - 图表生成
# - CSV导出
# - 周报PDF生成
# - 多微信(Server酱 Turbo)推送
# - Web Dashboard(JSON导出)
# - GitHub Actions 自动发布
# ===============================

import os
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

# ===============================
# 环境变量
# ===============================
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SERVER_KEYS = os.getenv("SERVERCHAN_KEYS")  # 多个key用逗号分隔
FRED_KEY = os.getenv("FRED_API_KEY")

if not OPENAI_KEY:
    raise RuntimeError("缺少 OPENAI_API_KEY")
if not SERVER_KEYS:
    raise RuntimeError("缺少 SERVERCHAN_KEYS")
if not FRED_KEY:
    raise RuntimeError("缺少 FRED_API_KEY")

client = OpenAI(api_key=OPENAI_KEY)
fred = Fred(api_key=FRED_KEY)

# ===============================
# 路径
# ===============================
BASE_DIR = Path(".")
OUTPUT_DIR = BASE_DIR / "outputs"
DOCS_DIR = BASE_DIR / "docs"

OUTPUT_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

# ===============================
# 投资模板
# ===============================
PORTFOLIO_TEMPLATE = {
    "S1": {"Stocks": "60%", "BTC": "20%", "Gold": "10%", "Cash": "10%"},
    "S2": {"Stocks": "40%", "BTC": "20%", "Gold": "20%", "Cash": "20%"},
    "S3": {"Stocks": "20%", "BTC": "10%", "Gold": "30%", "Cash": "40%"},
    "S4": {"Stocks": "0%", "BTC": "0%", "Gold": "40%", "Cash": "60%"},
}

# ===============================
# 数据获取
# ===============================
def get_market_data():
    print("📥 获取市场数据...")

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

    print(f"✅ 数据行数: {len(df)}")
    return df

# ===============================
# 指数计算
# ===============================
def compute_indices(df):
    returns = df.pct_change().dropna()

    li = returns["GOLD"].mean() - returns["DXY"].mean()
    ri = returns["BTC"].std() + returns["SP500"].std()

    return round(float(li), 4), round(float(ri), 4)

# ===============================
# 状态判断
# ===============================
def classify_state(li, ri):
    if li > 0.5 and ri < 0.5:
        return "S1"
    elif li > 0 and ri >= 0.5:
        return "S2"
    elif li <= 0 and ri >= 0.5:
        return "S3"
    else:
        return "S4"

# ===============================
# 转换概率
# ===============================
def transition_probability(li, ri):
    score = abs(li) * 0.6 + abs(ri) * 0.4
    return round(min(95, score * 100), 2)

# ===============================
# 图表生成
# ===============================
def generate_chart(df):
    print("📊 生成市场图表...")

    path = OUTPUT_DIR / "market_chart.png"

    plt.figure(figsize=(10, 5))
    df[["SP500", "BTC", "GOLD"]].tail(60).plot()
    plt.title("Market Trend - Last 60 Days")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return str(path)

# ===============================
# CSV导出
# ===============================
def export_csv(df):
    path = OUTPUT_DIR / "market_data.csv"
    df.to_csv(path)
    return str(path)

# ===============================
# 周报PDF
# ===============================
def generate_pdf(report_text):
    print("🧾 生成周报PDF...")

    path = OUTPUT_DIR / "weekly_report.pdf"
    styles = getSampleStyleSheet()
    pdf = SimpleDocTemplate(str(path))

    elements = []
    for line in report_text.split("\n"):
        elements.append(Paragraph(line, styles["Normal"]))
        elements.append(Spacer(1, 12))

    pdf.build(elements)
    return str(path)

# ===============================
# AI解读
# ===============================
def ai_macro_analysis(raw_data):
    print("🤖 请求AI策略分析...")

    prompt = f"""
你是华尔街宏观策略师，请基于以下宏观资金雷达数据输出专业分析报告：

{raw_data}

请结构化输出：
1. 当前资金流向解读
2. 风险等级评估
3. 股票 / 黄金 / 加密资产策略
4. 未来7天关键观察事件
5. 普通投资者操作建议
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )

    return response.choices[0].message.content.strip()

# ===============================
# 微信推送
# ===============================
def send_wechat(title, content):
    print("📨 推送微信通知...")
    keys = SERVER_KEYS.split(",")

    for key in keys:
        url = f"https://sctapi.ftqq.com/{key.strip()}.send"
        data = {
            "title": title,
            "desp": content
        }
        r = requests.post(url, data=data, timeout=15)
        print("微信状态:", r.status_code)

# ===============================
# Web Dashboard导出
# ===============================
def export_dashboard_data(li, ri, state, prob, portfolio):
    print("🌐 导出Web仪表盘数据...")

    payload = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "li": round(float(li), 4),
        "ri": round(float(ri), 4),
        "state": state,
        "transition_probability": round(float(prob), 2),
        "portfolio": portfolio
    }

    json_path = DOCS_DIR / "latest.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("✅ Dashboard数据生成:", json_path)
    return str(json_path)

# ===============================
# 主引擎
# ===============================
def run_engine():
    print("🚀 G3 宏观资金雷达启动")

    df = get_market_data()

    li, ri = compute_indices(df)
    state = classify_state(li, ri)
    prob = transition_probability(li, ri)

    chart = generate_chart(df)
    csv_file = export_csv(df)

    raw_report = f"""
时间(UTC): {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}

状态: {state}
流动性指数 LI: {li}
风险指数 RI: {ri}
转换概率: {prob}%

推荐仓位:
{json.dumps(PORTFOLIO_TEMPLATE[state], indent=2, ensure_ascii=False)}
"""

    ai_text = ai_macro_analysis(raw_report)
    pdf_path = generate_pdf(ai_text)

    full_msg = f"""
📡 G3 宏观资金雷达日报

{raw_report}

🧠 AI策略解读:
{ai_text}

📎 附件:
📊 图表: {chart}
📄 数据CSV: {csv_file}
🧾 周报PDF: {pdf_path}
"""

    send_wechat("📡 G3 宏观雷达日报", full_msg)
    export_dashboard_data(li, ri, state, prob, PORTFOLIO_TEMPLATE[state])

    print("🎉 系统运行完成")

# ===============================
# 程序入口
# ===============================
if __name__ == "__main__":
    run_engine()
