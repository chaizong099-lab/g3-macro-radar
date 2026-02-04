# ===============================
# G3 宏观资金雷达系统 - Web联动生产版
# 自动推送 GitHub Pages
# ===============================

import os
import json
import requests
import datetime
import subprocess
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
SERVER_KEYS = os.getenv("SERVERCHAN_KEYS")
FRED_KEY = os.getenv("FRED_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")  
# 格式: username/repo 例: chaizong099/g3-macro-radar

client = OpenAI(api_key=OPENAI_KEY)
fred = Fred(api_key=FRED_KEY)

# ========== 参数 ==========
PORTFOLIO_TEMPLATE = {
    "S1": {"Stocks": "60%", "BTC": "20%", "Gold": "10%", "Cash": "10%"},
    "S2": {"Stocks": "40%", "BTC": "20%", "Gold": "20%", "Cash": "20%"},
    "S3": {"Stocks": "20%", "BTC": "10%", "Gold": "30%", "Cash": "40%"},
    "S4": {"Stocks": "0%", "BTC": "0%", "Gold": "40%", "Cash": "60%"},
}

DATA_DIR = "outputs"
REPORT_DIR = "reports"

Path(DATA_DIR).mkdir(exist_ok=True)
Path(REPORT_DIR).mkdir(exist_ok=True)

# ========== 数据获取 ==========
def get_market_data():
    print("📥 拉取市场数据...")
    sp500 = yf.download("^GSPC", period="6mo")[["Close"]]
    btc = yf.download("BTC-USD", period="6mo")[["Close"]]
    gold = yf.download("GC=F", period="6mo")[["Close"]]

    dxy = fred.get_series("DTWEXBGS")
    rates = fred.get_series("DFF")

    dxy = pd.DataFrame(dxy, columns=["Close"])
    rates = pd.DataFrame(rates, columns=["Close"])

    sp500.columns = ["SP500"]
    btc.columns = ["BTC"]
    gold.columns = ["GOLD"]
    dxy.columns = ["DXY"]
    rates.columns = ["RATES"]

    df = pd.concat([sp500, btc, gold, dxy, rates], axis=1, join="inner").dropna()
    return df

# ========== 指标 ==========
def compute_indices(df):
    returns = df.pct_change().dropna()
    li = returns["GOLD"].mean() - returns["DXY"].mean()
    ri = returns["BTC"].std() + returns["SP500"].std()
    return round(li, 4), round(ri, 4)

# ========== 状态 ==========
def classify_state(li, ri):
    if li > 0.5 and ri < 0.5:
        return "S1"
    elif li > 0 and ri >= 0.5:
        return "S2"
    elif li <= 0 and ri >= 0.5:
        return "S3"
    else:
        return "S4"

def transition_probability(li, ri):
    score = abs(li) * 0.6 + abs(ri) * 0.4
    return round(min(95, score * 100), 2)

# ========== 图表 ==========
def generate_chart(df):
    path = os.path.join(DATA_DIR, "market_chart.png")
    df[["SP500", "BTC", "GOLD"]].tail(60).plot(figsize=(10, 5))
    plt.title("Market Trend - 60 Days")
    plt.grid(True)
    plt.savefig(path)
    plt.close()
    return path

# ========== CSV ==========
def export_csv(df):
    path = os.path.join(DATA_DIR, "market_data.csv")
    df.to_csv(path)
    return path

# ========== PDF ==========
def generate_pdf(text):
    path = os.path.join(DATA_DIR, "weekly_report.pdf")
    styles = getSampleStyleSheet()
    pdf = SimpleDocTemplate(path)
    elements = []

    for line in text.split("\n"):
        elements.append(Paragraph(line, styles["Normal"]))
        elements.append(Spacer(1, 12))

    pdf.build(elements)
    return path

# ========== AI ==========
def ai_macro_analysis(raw_data):
    print("🤖 请求 AI 分析...")
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
    return response.choices[0].message.content

# ========== 微信 ==========
def send_wechat(title, content):
    keys = SERVER_KEYS.split(",")
    for key in keys:
        url = f"https://sctapi.ftqq.com/{key.strip()}.send"
        r = requests.post(url, data={"title": title, "desp": content})
        print("📨 微信推送:", r.status_code)

# ========== Web数据导出 ==========
def export_dashboard_data(li, ri, state, prob, portfolio):
    payload = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "li": li,
        "ri": ri,
        "state": state,
        "transition_probability": prob,
        "portfolio": portfolio
    }

    path = Path(REPORT_DIR) / "latest.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print("🌐 Web 数据已生成:", path)

# ========== GitHub推送 ==========
def push_to_github():
    print("🚀 推送 Web 数据到 GitHub Pages...")

    subprocess.run(["git", "config", "--global", "user.email", "bot@g3radar.ai"])
    subprocess.run(["git", "config", "--global", "user.name", "G3 Radar Bot"])

    subprocess.run(["git", "add", "reports/latest.json"])
    subprocess.run(["git", "commit", "-m", "update dashboard data"], check=False)

    repo_url = f"https://x-access-token:{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"
    subprocess.run(["git", "push", repo_url, "HEAD:main"])

# ========== 主程序 ==========
def run_engine():
    print("📡 G3 宏观雷达启动")

    df = get_market_data()
    li, ri = compute_indices(df)
    state = classify_state(li, ri)
    prob = transition_probability(li, ri)

    chart = generate_chart(df)
    csv_file = export_csv(df)

    raw_report = f"""
时间: {datetime.datetime.utcnow()}

状态: {state}
流动性指数 LI: {li}
风险指数 RI: {ri}
状态转换概率: {prob}%

推荐仓位:
{PORTFOLIO_TEMPLATE[state]}
"""

    ai_text = ai_macro_analysis(raw_report)
    pdf_path = genera_
