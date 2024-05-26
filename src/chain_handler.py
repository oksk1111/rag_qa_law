from langchain.callbacks.base import BaseCallbackHandler

class ChainHandler(BaseCallbackHandler):
    def __init__(self, domain=None, details=None):
        self.domain = domain
        self.details = details


    def on_tool_start(self, serialized: dict[str, any], **kwargs: any) -> any:
        """Run when tool starts running."""
        self.domain = serialized


    def on_tool_start(
        self, serialized: dict[str, any], input_str: str, **kwargs: any
    ) -> any:
        """Run when tool starts running.

        Parameters
          - serialized: 어떤 툴로 분류 되었는지 알 수 있다.
          - input_str: 어떤 질의어인지 알 수 있다.
        """
        self.domain = serialized


    def on_tool_end(self, output: str, **kwargs: any) -> any:
        """Run when tool ends running.

        Parameters
          - output: 검색을 통해 얻게 된 값. 부산물들.
        """
        self.details = output