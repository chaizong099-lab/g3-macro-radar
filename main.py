# ===============================
# G3 宏观资金雷达系统 - Web + 微信生产版
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
from sklearn.preprocessing import StandardScaler
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
# 参数
# ===============================
DATA_DIR = "outputs"
REPORT_DIR = "reports"
STATE_FILE = "state.json"   # ⭐ 新增：推送状态文件

Path(DATA_DIR).mkdir(exist_ok=True)
Path(REPORT_DIR).mkdir(exist_ok=True)

PORTFOLIO_TEMPLATE = {
    "S1": {"Stocks": "60%", "BTC": "20%", "Gold": "10%", "Cash": "10%"},
    "S2": {"Stocks": "40%", "BTC": "20%", "Gold": "20%", "Cash": "20%"},
    "S3": {"Stocks": "20%", "BTC": "10%", "Gold": "30%", "Cash": "40%"},
    "S4": {"Stocks": "0%", "BTC": "0%", "Gold": "40%", "Cash": "60%"},
}

# ===============================
# 🔒 当天只推一次：工具函数
# ===============================
def today_str():
    return datetime.date.today().isoformat()


def already_sent_today():
    if not os.path.exists(STATE_FILE):
        return False
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        return state.get("last_sent") == today_str()
    except Exception:
        return False


def mark_sent_today():
    with open(STATE_FILE, "w") as f:
        json.dump({"last_sent": today_str()}, f)

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

    dxy = pd.DataFrame(dxy, columns=["Close"])
    rates = pd.DataFrame(rates, columns=["Close"])

    sp500.columns = ["SP500"]
    btc.columns = ["BTC"]
    gold.columns = ["GOLD"]
    dxy.columns = ["DXY"]
    rates.columns = ["RATES"]

    df = pd.concat([sp500, btc, gold, dxy, rates], axis=1, join="inner").dropna()
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
# CSV导出
# ===============================
def export_csv(df):
    path = os.path.join(DATA_DIR, "market_data.csv")
    df.to_csv(path)
    return path

# ===============================
# PDF周报
# ===============================
def generate_pdf(report_text):
    print("📄 生成PDF...")
    path = os.path.join(DATA_DIR, "weekly_report.pdf")

    styles = getSampleStyleSheet()
    pdf = SimpleDocTemplate(path)

    elements = []
    for line in report_text.split("\n"):
        elements.append(Paragraph(line, styles["Normal"]))
        elements.append(Spacer(1, 12))

    pdf.build(elements)
    return path

# ===============================
# Web仪表盘导出
# ===============================
def export_dashboard_data(li, ri, state, prob, portfolio):
    print("🌍 生成Web数据...")
    payload = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "li": li,
        "ri": ri,
        "state": state,
        "transition_probability": prob,
        "portfolio": portfolio
    }

    path = os.path.join(REPORT_DIR, "latest.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"🌍 Web数据已生成: {path}")

# ===============================
# AI分析
# ===============================
def ai_macro_analysis(raw_data):
    prompt = f"""
你是全球宏观策略师，请根据系统数据给出专业投资解读：

{raw_data}

输出：
1. 当前阶段判断
2. 风险提示
3. 黄金 / 美股 / 加密策略
4. 未来7天观察点
"""

    resp = client.responses.create(
        model="gpt-5",
        input=prompt
    )

    return resp.output_text.strip()

# ===============================
# 微信推送
# ===============================
def send_wechat(title, content):
    print("🔔 推送微信...")
    keys = SERVER_KEYS.split(",")

    for key in keys:
        key = key.strip()
        if not key:
            continue
        url = f"https://sctapi.ftqq.com/{key}.send"
        r = requests.post(url, data={"title": title, "desp": content}, timeout=10)
        print("📨", key, r.status_code)

# ===============================
# 主引擎
# ===============================
def run_engine():
    print("📡 正在运行宏观雷达系统...")

    # 🔒 保险：当天已推送直接退出
    if already_sent_today():
        print("🛑 今日已推送，跳过执行")
        return

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
转换概率: {prob}%

推荐仓位:
{PORTFOLIO_TEMPLATE[state]}
"""

    print("🤖 请求AI分析中...")
    ai_text = ai_macro_analysis(raw_report)

    pdf_path = generate_pdf(ai_text)

    full_msg = f"""
📊 G3 宏观资金雷达

{raw_report}

🧠 AI解读:
{ai_text}

📎 文件:
图表: {chart}
数据: {csv_file}
周报PDF: {pdf_path}
"""

    send_wechat("📡 G3 宏观雷达日报", full_msg)

    # ✅ 只有成功推送后才记录
    mark_sent_today()

    # ===== Web仪表盘数据输出 =====
    export_dashboard_data(li, ri, state, prob, PORTFOLIO_TEMPLATE[state])

# ===============================
# 入口
# ===============================
if __name__ == "__main__":
    run_engine()
