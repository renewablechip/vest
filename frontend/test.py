from openai import OpenAI

OPENAI_API_KEY = "sk-b34cff67e35f46f686d1423c0b0bcd35"

# 初始化客户端
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.deepseek.com"
)

# 对话历史（含推理提示）
messages = [
    {
        "role": "system",
        "content":"You are a helpful assistant."
    }
]

while True:
    # 用户输入
    user_input = input("你：")
    if user_input.lower() in ["exit", "quit", "退出"]:
        break

    # 添加用户消息到对话历史
    messages.append({"role": "user", "content": user_input})

    # 初始化 response 累积内容
    reasoning_content = ""
    final_content = ""

    # 发起流式请求
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        stream=True
    )

    print("思维过程：", end="", flush=True)
    in_reasoning = False

    for chunk in response:
        # 输出推理内容
        if chunk.choices[0].delta.reasoning_content:
            delta = chunk.choices[0].delta.reasoning_content
            print(delta, end="", flush=True)
            reasoning_content += delta
            in_reasoning = True

        # 输出最终回答内容
        elif chunk.choices[0].delta.content:
            if in_reasoning:
                print()  # 分隔 reasoning 和 answer
                print("最终答案：", end="", flush=True)
                in_reasoning = False
            delta = chunk.choices[0].delta.content
            print(delta, end="", flush=True)
            final_content += delta

    print()  # 输出换行

    # 添加最终回答（只添加 final_content）到对话历史
    messages.append({"role": "assistant", "content": final_content})
