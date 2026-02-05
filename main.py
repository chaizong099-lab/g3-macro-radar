# ===============================
# G3 宏观资金雷达系统
# Web + 微信 + AI 分析（稳定生产版）
# ===============================

import os
import json
import requests
import datetime
from pathlib import Path

import yfinance as yf
import pandas as pd
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

if not OPENAI_KEY:
    raise RuntimeError("❌ 缺少 OPENAI_API_KEY")

if not FRED_KEY:
    raise RuntimeError("❌ 缺少 FRED_API_KEY")

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
# 资产配置模板
# ===============================
PORTFOLIO_TEMPLATE = {
    "S1": {"Stocks": "60%", "BTC": "20%", "Gold": "10%", "Cash": "10%"},
    "S2": {"Stocks": "40%", "BTC": "20%", "Gold": "20%", "Cash": "20%"},
    "S3": {"Stocks": "20%", "BTC": "10%", "Gold": "30%", "Cash": "40%"},
    "S4": {"Stocks": "0%", "BTC": "0%", "Gold": "40%", "Cash": "60%"},
}

# ===============================
# 数据获取（已做稳定性处理）
# ===============================
def get_market_data():
    print("📡 加载市场数据...")

    sp500 = yf.download("^GSPC", period="6mo", progress=False)
    btc = yf.download("BTC-USD", period="6mo", progress=False)
    gold = yf.download("GC=F", period="6mo", progress=False)

    if sp500.empty or btc.empty or gold.empty:
        raise RuntimeError("❌ Yahoo Finance 返回空数据")

    sp500 = sp500[["Close"]].rename(columns={"Close": "SP500"})
    btc = btc[["Close"]].rename(columns={"Close": "BTC"})
    gold = gold[["Close"]].rename(columns={"Close": "GOLD"})

    dxy = pd.DataFrame(fred.get_series("DTWEXBGS"), columns=["DXY"])
    rates = pd.DataFrame(fred.get_series("DFF"), columns=["RATES"])

    df = pd.concat(
        [sp500, btc, gold, dxy, rates],
        axis=1,
        join="inner"
    ).dropna()

    required = {"SP500", "BTC", "GOLD", "DXY", "RATES"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"❌ 数据缺失列: {missing}")

    print("✅ 数据加载成功:", df.columns.tolist())
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
# 图表
# ===============================
def generate_chart(df):
    path = os.path.join(DATA_DIR, "market_chart.png")

    df[["SP500", "BTC", "GOLD"]].tail(60).plot(figsize=(10, 5))
    plt.title("G3 Macro Radar - 60 Days")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path

# ===============================
# CSV 导出
# ===============================
def export_csv(df):
    path = os.path.join(DATA_DIR, "market_data.csv")
    df.to_csv(path)
    return path

# ===============================
# PDF 周报
# ===============================
def generate_pdf(text):
    path = os.path.join(DATA_DIR, "weekly_report.pdf")

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(path)

    elements = []
    for line in text.split("\n"):
        elements.append(Paragraph(line, styles["Normal"]))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    return path

# ===============================
# Web 仪表盘数据
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

    path = os.path.join(REPORT_DIR, "latest.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print("🌍 Web 数据已生成:", path)

# ===============================
# AI 解读（OpenAI 新接口）
# ===============================
def ai_macro_analysis(raw_text):
    resp = client.responses.create(
        model="gpt-5",
        input=f"""
你是全球宏观策略师，请解读以下系统输出并给出投资建议：

{raw_text}

请包括：
1. 当前市场阶段判断
2. 风险级别
3. 黄金 / 美股 / 加密策略
4. 未来7天观察点
"""
    )

    return resp.output_text.strip()

# ===============================
# 微信推送
# ===============================
def send_wechat(title, content):
    keys = SERVER_KEYS.split(",")

    for key in keys:
        key = key.strip()
        if not key:
            continue

        url = f"https://sctapi.ftqq.com/{key}.send"
        r = requests.post(url, data={"title": title, "desp": content}, timeout=10)
        print("📨 微信:", key, r.status_code)

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

流动性指数 LI: {li}
风险指数 RI: {ri}
转换概率: {prob}%

推荐仓位:
{PORTFOLIO_TEMPLATE[state]}
"""

    print("🤖 AI 解读中...")
    ai_text = ai_macro_analysis(raw_report)

    pdf_path = generate_pdf(ai_text)

    message = f"""
📊 G3 宏观资金雷达

{raw_report}

🧠 AI 解读:
{ai_text}

📎 文件:
图表: {chart}
CSV: {csv_file}
PDF: {pdf_path}
"""

    send_wechat("📡 G3 宏观雷达日报", message)
    export_dashboard_data(li, ri, state, prob, PORTFOLIO_TEMPLATE[state])

# ===============================
# 程序入口
# ===============================
if __name__ == "__main__":
    run_engine()
