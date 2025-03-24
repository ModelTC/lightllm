import time

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Union, Literal
import uuid

class ImageData(BaseModel):
    type: str
    data: str

class ImageURL(BaseModel):
    url: List[ImageData]

    @field_validator("url", mode="before")
    def ensure_list(cls, v):
        if isinstance(v, list):
            new_list = []
            for item in v:
                if isinstance(item, str):
                    new_list.append({"type": "base64", "data": item})
                elif isinstance(item, dict):
                    if "type" not in item:
                        item["type"] = "base64"
                    new_list.append(item)
                else:
                    new_list.append(item)
            return new_list
        elif isinstance(v, str):
            return [{"type": "base64", "data": v}]
        elif isinstance(v, dict):
            if "type" not in v:
                v["type"] = "base64"
            return [v]
        return v

class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[MessageContent]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
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

    # Additional parameters supported by LightLLM
    do_sample: Optional[bool] = False
    top_k: Optional[int] = -1
    repetition_penalty: Optional[float] = 1.0
    ignore_eos: Optional[bool] = False
    role_settings: Optional[Dict[str, str]] = None
    character_settings: Optional[List[Dict[str, str]]] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: Optional[int] = 0
    total_tokens: int = 0


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "function_call"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo

    @field_validator("id", mode="before")
    def ensure_id_is_str(cls, v):
        return str(v)


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionStreamResponseChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamResponseChoice]

    @field_validator("id", mode="before")
    def ensure_id_is_str(cls, v):
        return str(v)
