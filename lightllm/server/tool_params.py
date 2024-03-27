from typing import Dict, List


class ToolParameter:
    def __init__(self, obj, name):
        self.description = obj.description
        self.enum = obj.enum
        self.name = name
        return


class ToolItem:
    def __init__(self, obj):
        # print(obj)
        self.description = obj.function.description
        self.name = obj.function.name
        self.parameters = {k: ToolParameter(v, k) for k, v in obj.function.parameters.properties.items()}
        self.required = obj.function.parameters.required
        return


class ToolParams:
    def __init__(self, tools: List[dict]):
        self.tools = [ToolItem(i) for i in tools]
        return


class ToolChoiceParams:
    def __init__(self, obj):
        self.choice_function_name = obj.function.name
        return
