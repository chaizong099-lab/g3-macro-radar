# ===============================
# G3 宏观资金雷达系统 - 生产版
# 支持：
# - 宏观状态识别 S1-S4
# - AI策略解读（OpenAI API）
# - 图表生成 + 图床上传
# - CSV导出
# - 周报PDF生成
# - 多微信(Server酱 Turbo)推送
# - GitHub Actions 定时运行
# ===============================

import os
import io
import requests
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from fredapi import Fred
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from openai import OpenAI

# ========== 环境变量 ==========
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SERVER_KEYS = os.getenv("SERVERCHAN_KEYS")  # 多个用逗号分隔
FRED_KEY = os.getenv("FRED_API_KEY")

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
os.makedirs(DATA_DIR, exist_ok=True)

# ========== 数据获取 ==========
def get_market_data():
    sp500 = yf.download("^GSPC", period="6mo", interval="1d")[["Close"]]
    btc = yf.download("BTC-USD", period="6mo", interval="1d")[["Close"]]
    gold = yf.download("GC=F", period="6mo", interval="1d")[["Close"]]

    dxy = fred.get_series("DTWEXBGS")
    rates = fred.get_series("DFF")

    # 转成带时间索引的 DataFrame
    dxy = pd.DataFrame(dxy, columns=["Close"])
    rates = pd.DataFrame(rates, columns=["Close"])

    # 统一列名
    sp500.columns = ["SP500"]
    btc.columns = ["BTC"]
    gold.columns = ["GOLD"]
    dxy.columns = ["DXY"]
    rates.columns = ["RATES"]

    # 按日期对齐（量化标准做法）
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

    return round(li, 3), round(ri, 3)

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
    path = os.path.join(DATA_DIR, "market_chart.png")

    df[["SP500", "BTC", "GOLD"]].tail(60).plot(figsize=(10, 5))
    plt.title("Market Trend - 60 Days")
    plt.grid(True)
    plt.savefig(path)
    plt.close()

    return path

# ========== CSV导出 ==========
def export_csv(df):
    path = os.path.join(DATA_DIR, "market_data.csv")
    df.to_csv(path)
    return path

# ========== 周报PDF ==========
def generate_pdf(report_text):
    path = os.path.join(DATA_DIR, "weekly_report.pdf")

    styles = getSampleStyleSheet()
    pdf = SimpleDocTemplate(path)

    elements = []
    for line in report_text.split("\n"):
        elements.append(Paragraph(line, styles["Normal"]))
        elements.append(Spacer(1, 12))

    pdf.build(elements)
    return path

# ========== AI解读 ==========
def ai_macro_analysis(raw_data):
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

# ========== 微信推送 ==========
def send_wechat(title, content):
    keys = SERVER_KEYS.split(",")

    for key in keys:
        url = f"https://sctapi.ftqq.com/{key.strip()}.send"
        data = {
            "title": title,
            "desp": content
        }
        r = requests.post(url, data=data)
        print("📨 微信状态:", r.status_code, r.text)

# ========== 主引擎 ==========
def run_engine():
    print("📡 正在运行宏观雷达系统...")

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

🧠 AI策略解读:
{ai_text}

📎 附件:
图表: {chart}
数据: {csv_file}
周报PDF: {pdf_path}
"""

    send_wechat("📡 G3 宏观雷达日报", full_msg)

# ========== 程序入口 ==========
if __name__ == "__main__":
    run_engine()
