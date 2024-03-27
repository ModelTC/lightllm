import time

from pydantic import BaseModel, Field, Json
from typing import Dict, List, Optional, Union, Literal
import uuid


class FunctionParameterProperty(BaseModel):
    type: str
    description: Optional[str] = None
    enum: Optional[List[str]] = None


class FunctionParameters(BaseModel):
    type: str = Field(default="object")
    required: Optional[List[str]] = None
    properties: Dict[str, FunctionParameterProperty]


class FunctionDefinition(BaseModel):
    description: Optional[str] = None
    name: str
    parameters: FunctionParameters
    required: Optional[List[str]] = None


class ToolsChoice(BaseModel):
    type: str = Field(default="function")
    function: Dict[str, str]


class Tools(BaseModel):
    # The type of the tool. Currently only "function" is supported.
    type: str = Field(default="function")
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    # The openai api native parameters
    model: str
    messages: List[Dict[str, str]]
    function_call: Optional[str] = "none"
    temperature: Optional[float] = 1
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 16
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    # Tools Call Adapted from OpenAI
    tools: Optional[List[Tools]] = None
    tools_choice: Optional[Union[ToolsChoice, Literal["none", "auto"]]] = None

    # Additional parameters supported by LightLLM
    do_sample: Optional[bool] = False
    top_k: Optional[int] = -1
    ignore_eos: Optional[bool] = False


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: Optional[int] = 0
    total_tokens: int = 0


class ChatMessage(BaseModel):
    role: str
    content: str


class FunctionCall(BaseModel):
    name: str
    arguments: Json


class ChatCompletionResponseToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"call-{uuid.uuid4().hex}")
    function: FunctionCall
    type: str = Field(default="function")


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "function_call", "tool_calls"]] = None
    tool_calls: Optional[List[ChatCompletionResponseToolCall]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionStreamResponseChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamResponseChoice]
