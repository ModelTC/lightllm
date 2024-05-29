from packaging import version
from lightllm.server.api_models import ChatMessage
from typing import List, Dict

try:
    import fastchat
    from fastchat.conversation import Conversation, SeparatorStyle
    from fastchat.model.model_adapter import get_conversation_template

    _fastchat_available = True
except ImportError:
    _fastchat_available = False


# only consider str message content
def parse_chat_message_content(
    message: Dict[str, str],
) -> List[ChatMessage]:
    role = message["role"]
    content = message.get("content")

    if content is None:
        return []
    if isinstance(content, str):
        messages = [{"role":role, "content":content}]
        return messages

    return []


async def build_prompt_v2(request, tokenizer) -> str:
    conversation = []

    for msg in request.messages:
        parsed_msg = parse_chat_message_content(msg)
        conversation.extend(parsed_msg)
    
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
                    conversation=conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                 )
    else: 
        return build_prompt(request)
    
    print(f"Prompt: {prompt}")
    return prompt


async def build_prompt(request) -> str:
    if not _fastchat_available:
        raise ModuleNotFoundError(
            "fastchat is not installed. Please install fastchat to use "
            "the chat completion and conversation APIs: `$ pip install 'fschat[model_worker,webui]'`"
        )
    if version.parse(fastchat.__version__) < version.parse("0.2.23"):
        raise ImportError(
            f"fastchat version is low. Current version: {fastchat.__version__} "
            "Please upgrade fastchat to use: `$ pip install 'fschat[model_worker,webui]'`")

    conv = get_conversation_template(request.model)
    conv = Conversation(
        name=conv.name,
        system_template=conv.system_template,
        system_message=conv.system_message,
        roles=conv.roles,
        messages=list(conv.messages),  # prevent in-place modification
        offset=conv.offset,
        sep_style=SeparatorStyle(conv.sep_style),
        sep=conv.sep,
        sep2=conv.sep2,
        stop_str=conv.stop_str,
        stop_token_ids=conv.stop_token_ids,
    )

    if isinstance(request.messages, str):
        prompt = request.messages
    else:
        for message in request.messages:
            msg_role = message["role"]
            if msg_role == "system":
                conv.system_message = message["content"]
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")
        # Add a blank message for the assistant. Meaning it's the assistant's turn to talk.
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    return prompt
