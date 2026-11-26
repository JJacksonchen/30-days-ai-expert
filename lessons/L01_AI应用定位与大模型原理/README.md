# 第 01 课：别再死磕算法了！AI 应用工程师与 FDE 定位 + 生活例子讲透大模型原理

## 🧭 本课知识结构图（现象 👉 底层本质 👉 生活举例）

```mermaid
flowchart LR
    A["转行大局<br/>造发动机(算法岗 1%)<br/>vs 开车队落地(AI应用/FDE 99%)"] --> B["故事1: Token<br/>老外背成语卡片编号<br/>为何数不清strawberry有几个r"]
    B --> C["故事2: 向量地图<br/>外卖口味打分卡<br/>只看方向不看嗓门 -> RAG底座"]
    C --> D["故事3: 自注意力<br/>全员进微信群查资料<br/>动态区分水果苹果 vs 苹果手机"]
    D --> E["故事4: 超级接龙<br/>猜下一个字 + 发草稿纸<br/>分析理由放结论前准确率暴涨!"]
```

## 💻 6 行代码让大模型干活（DeepSeek / OpenAI / Qwen 全通用）

```python
from openai import OpenAI

client = OpenAI(api_key="你的API_KEY", base_url="https://api.deepseek.com")
resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是资深合同审核顾问，请先分条分析风险，再给出最终建议"},
        {"role": "user", "content": "请帮我检查这段租房合同有没有押金霸王条款..."},
    ],
)
print(resp.choices[0].message.content)
```
