# local_test_agent.py
import asyncio, json, sys, litellm
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters

async def main():
    # 1)  spawn the server **locally** and connect over its stdin/stdout
    server_params = StdioServerParameters(
        command=sys.executable,          # e.g. "/usr/bin/python3"
        args=["mcp_server.py"],   # script lives alongside this file
    )
    stdio_pair = await stdio_client(server_params)    # (reader, writer)
    session = await ClientSession(*stdio_pair)
    await session.initialize()

    # 2)  Fetch tool metadata → convert to OpenAI tool schema for LiteLLM
    tools_resp = await session.list_tools()
    tools = [{
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.inputSchema,
        },
    } for t in tools_resp.tools]
    
    breakpoint()

    # 3)  Give the LLM a trivial goal (“list windows”)
    messages = [{"role": "user", "content": "List the currently open windows"}]

    while True:
        resp = litellm.completion(
            model="gpt-4o-mini",         # or any provider that follows OpenAI fn-call spec
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        # a)  LLM decided to call a tool?
        if msg.tool_calls:
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                tool_result = await session.call_tool(call.function.name, args)

                # feed the result back to the model for the next turn
                messages.extend([
                    {"role": "assistant", "tool_calls": [call]},
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.function.name,
                        "content": tool_result.content,
                    },
                ])
            continue

        # b)  LLM is done (no more function calls)
        print("LLM final answer:\n", msg.content)
        break

asyncio.run(main())
