import json
import os
from safety import safe_call


class Agent:

    
    

    def __init__(self, llm, tools, max_steps=5):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.memory = {}
        self.current_image=None
        if os.path.exists("memory.json"):
            with open("memory.json", "r") as f:
                self.memory = json.load(f)
        else:
            self.memory = {}

    @safe_call
    def call_llm(self, prompt):
        return self.llm.invoke(prompt)
    
    @safe_call
    def execute_tool(self, tool_name, tool_input, thread_id):

        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' does not exist.")

        if tool_name == "search_documents":
            return self.tools[tool_name].func(
                tool_input,
                thread_id
            )

        elif tool_name == "describe_image":

            if self.current_image is None:
                raise ValueError("No image has been uploaded.")

            else:
                return self.tools[tool_name].invoke(self.current_image)
        else:
            return self.tools[tool_name].invoke(tool_input)
    @safe_call
    def persist_memory(self):
        with open("memory.json", "w") as f:
         json.dump(self.memory, f, indent=4)


    def _build_prompt(self, user_query, scratchpad,history):
        tool_descriptions = "\n".join(
            f"- {name}" for name in self.tools.keys()
        )

        return f"""
        You are an AI Agent.

        Available tools:
        {tool_descriptions}

        search_documents
        Search the user's uploaded documents for information relevent to the query.Use this when the 
        user asks about content from a PDF they uploaded, or references 'the document', 'my notes',
        or 'the file'.
        Do NOT use this for general knowledge or current events

        web_search
        Use this tool to search the web for real-time information.
        Input: a search query string.
        Output: a summary of the most relevant search results.
        Use this for current events, news, live prices, or anything published after 2024.
        
        describe_image
        Use ONLY when an uploaded image must be analysed.


        You MUST respond ONLY with valid JSON.

        If you need a tool:

        {{
            "action": "tool",
            "tool_name": "<tool_name>",
            "tool_input": "<query>"
        }}

        If the tool you are using is describe_image then you can respond like this:
        {{
            "action": "tool",
            "tool_name": "<tool_name>",
            "tool_input": "CURRENT_IMAGE"
        }}
        here CURRENT_IMAGE is a special keyword that tells the agent to use the image that was last uploaded by the user.
        Never invent another value.Never include the image itself.The runtime will automatically replace CURRENT_IMAGE with the uploaded image.
        
        
        If you have enough information to answer the user's question, respond like this:
        {{
            "action": "final",
            "answer": "<answer>"
        }}

        Conversation history:
        {history}

        Current user question:
        {user_query}

        Previous reasoning:
        {scratchpad}
        """
    
    
    def save_message(self, thread_id, role, content):

        if thread_id not in self.memory:
            self.memory[thread_id] = []

        self.memory[thread_id].append(
        {
            "role": role,
            "content": content
        }
    )
        self.persist_memory()


    def get_history(self, thread_id):

        return self.memory.get(thread_id, [])

    def final_summary(self,user_query,scratchpad,thread_id,history):
        summary_prompt = f"""
        History:
        {history}

        Question:
        {user_query}

        Collected facts:
        {scratchpad}

        Write a concise answer for the user.
        """
        response = self.call_llm(summary_prompt)
        if isinstance(response, str):
            return response
        answer = response.content
        self.save_message(thread_id,"assistant",answer)
        return answer
   
   
    def run(self, user_query,thread_id):
        scratchpad = ""

        self.save_message(
        thread_id,
        "user",
        user_query
    )
        history = self.get_history(thread_id)

        for step in range(self.max_steps):

            prompt = self._build_prompt(
                user_query=user_query,
                scratchpad=scratchpad,
                history=history
            )

            response = self.call_llm(prompt)
            if isinstance(response, str):
                return response, scratchpad
            text = response.content

            try:
                decision = json.loads(text)
            except Exception:
                return (f"Invalid JSON returned:\n{text}",scratchpad)

            action = decision.get("action")
            if action not in ("tool", "final"):
                return (
            "⚠️ The planner returned an invalid action.",
            scratchpad
            )

            if action == "final":
                return self.final_summary(user_query,scratchpad,thread_id,history),scratchpad

            if action == "tool":

                tool_name = decision["tool_name"]
                tool_input = decision["tool_input"]
                
                result = self.execute_tool(
                tool_name,
                tool_input,
                thread_id
                )
                if result == "No documents uploaded yet":
                    return (
        "📄 No document has been uploaded yet.\n\n"
        "Please upload a PDF in the **Document QA** tab first.",
        scratchpad
    )
                if result == "No relevant content found in the uploaded documents.":
                    return (
        "I couldn't find information about that in the uploaded document.",
        scratchpad
    )
                tool_failed = (isinstance(result, str) and 
                (result.startswith("⚠️")
                or result.startswith("❌")
                or result.startswith("🌐")
                or result.startswith("🔑")
                or result.startswith("🚫")
    )
)                
                
                scratchpad += f"""
                Step {step+1}

                Tool Used:
                {tool_name}

                Tool Input:
                {tool_input}

                Tool Result:
                {result}
                """
            if tool_failed:
                continue
        return ("Maximum reasoning steps exceeded.",scratchpad)