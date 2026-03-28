class SafeToolExecutor:
    @staticmethod
    def execute(agent_executor, query: str):
        try:
            response = agent_executor.invoke({"input": query})
            return response["output"]
        except Exception as e:
            return f"Tool execution failed: {str(e)}"
