# ===============================
# G3 宏观资金雷达系统 - Web + 微信生产版
# ===============================

import os
import json
import requests
import datetime
from pathlib import Path
from datetime import date

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
SERVER_KEYS = os.getenv("SERVERCHAN_KEYS", "")
FRED_KEY = os.getenv("FRED_API_KEY")

client = OpenAI(api_key=OPENAI_KEY)
fred = Fred(api_key=FRED_KEY)

# ===============================
# 目录
# ===============================
DATA_DIR = "outputs"
REPORT_DIR = "reports"
Path(DATA_DIR).mkdir(exist_ok=True)
Path(REPORT_DIR).mkdir(exist_ok=True)

# ===============================
# 组合模板
# ===============================
PORTFOLIO_TEMPLATE = {
    "S1": {"Stocks": "60%", "BTC": "20%", "Gold": "10%", "Cash": "10%"},
    "S2": {"Stocks": "40%", "BTC": "20%", "Gold": "20%", "Cash": "20%"},
    "S3": {"Stocks": "20%", "BTC": "10%", "Gold": "30%", "Cash": "40%"},
    "S4": {"Stocks": "0%", "BTC": "0%", "Gold": "40%", "Cash": "60%"},
}

# ===============================
# ✅ 推送保险（核心）
# ===============================
PUSH_FLAG_FILE = os.path.join(REPORT_DIR, "last_push_date.json")

def already_pushed_today():
    today = date.today().isoformat()
    if not os.path.exists(PUSH_FLAG_FILE):
        return False
    try:
        with open(PUSH_FLAG_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("date") == today
    except Exception:
        return False

def mark_pushed_today():
    with open(PUSH_FLAG_FILE, "w", encoding="utf-8") as f:
        json.dump({"date": date.today().isoformat()}, f, indent=2)

# ===============================
# 数据获取
# ===============================
def get_market_data():
    print("📡 加载市场数据...")
    sp500 = yf.download("^GSPC", period="6mo")[["Close"]]
    btc = yf.download("BTC-USD", period="6mo")[["Close"]]
    gold = yf.download("GC=F", period="6mo")[["Close"]]

    dxy = fred.get_series("DTWEXBGS")
    rates = fred.get_series("DFF")

    df = pd.concat([
        sp500.rename(columns={"Close": "SP500"}),
        btc.rename(columns={"Close": "BTC"}),
        gold.rename(columns={"Close": "GOLD"}),
        pd.DataFrame(dxy, columns=["DXY"]),
        pd.DataFrame(rates, columns=["RATES"])
    ], axis=1).dropna()

    return df

# ===============================
# 指标计算
# ===============================
def compute_indices(df):
    returns = df.pct_change().dropna()
    li = returns["GOLD"].mean() - returns["DXY"].mean()
    ri = returns["BTC"].std() + returns["SP500"].std()
    return round(float(li), 4), round(float(ri), 4)

# ===============================
# 状态识别
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
# 图表
# ===============================
def generate_chart(df):
    print("📈 生成图表...")
    path = os.path.join(DATA_DIR, "market_chart.png")
    df[["SP500", "BTC", "GOLD"]].tail(60).plot(figsize=(10, 5))
    plt.title("G3 Macro Radar - 60 Days")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

# ===============================
# CSV
# ===============================
def export_csv(df):
    path = os.path.join(DATA_DIR, "market_data.csv")
    df.to_csv(path)
    return path

# ===============================
# PDF
# ===============================
def generate_pdf(text):
    path = os.path.join(DATA_DIR, "weekly_report.pdf")
    styles = getSampleStyleSheet()
    pdf = SimpleDocTemplate(path)
    elements = []
    for line in text.split("\n"):
        elements.append(Paragraph(line, styles["Normal"]))
        elements.append(Spacer(1, 10))
    pdf.build(elements)
    return path

# ===============================
# Web 数据
# ===============================
def export_dashboard_data(li, ri, state, prob, portfolio):
    payload = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "li": li,
        "ri": ri,
        "state": state,
        "transition_probability": prob,
        "portfolio": portfolio
    }
    with open(os.path.join(REPORT_DIR, "latest.json"), "w") as f:
        json.dump(payload, f, indent=2)

# ===============================
# AI 分析
# ===============================
def ai_macro_analysis(raw):
    resp = client.responses.create(
        model="gpt-5",
        input=f"你是宏观策略师，请解读以下数据：\n{raw}"
    )
    return resp.output_text.strip()

# ===============================
# 微信推送
# ===============================
def send_wechat(title, content):
    for key in SERVER_KEYS.split(","):
        if key.strip():
            url = f"https://sctapi.ftqq.com/{key.strip()}.send"
            requests.post(url, data={"title": title, "desp": content}, timeout=10)

# ===============================
# 主引擎
# ===============================
def run_engine():
    print("🚀 G3 Macro Radar 启动")

    df = get_market_data()
    li, ri = compute_indices(df)
    state = classify_state(li, ri)
    prob = transition_probability(li, ri)

    chart = generate_chart(df)
    csv_file = export_csv(df)

    raw_report = f"""
时间: {datetime.datetime.utcnow()}
状态: {state}
LI: {li}
RI: {ri}
概率: {prob}%
仓位: {PORTFOLIO_TEMPLATE[state]}
"""

    ai_text = ai_macro_analysis(raw_report)
    pdf = generate_pdf(ai_text)

    # Web 永远更新
    export_dashboard_data(li, ri, state, prob, PORTFOLIO_TEMPLATE[state])

    # 微信只推一次
    if already_pushed_today():
        print("✅ 今日已推送，跳过微信")
        return

    msg = f"""
📡 G3 宏观雷达日报

{raw_report}

🧠 AI解读:
{ai_text}
"""
    send_wechat("📊 G3 宏观雷达", msg)
    mark_pushed_today()
    print("🔔 微信已推送")

# ===============================
# 入口
# ===============================
if __name__ == "__main__":
    run_engine()
