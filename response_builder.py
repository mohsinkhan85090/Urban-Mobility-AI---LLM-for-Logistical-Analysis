# response_builder.py


class ResponseBuilder:

    @staticmethod
    def build_tool_response(result: dict):

        if not result:
            return "No result available."

        formatted = "\n".join(
            [f"{k}: {v}" for k, v in result.items()]
        )

        return formatted