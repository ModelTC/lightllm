import os
import sys
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from format_out.impl import ChatSession
from format_out.impl import SamplingParams

# server model is Meta-Llama-3-8B-Instruct

system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>you are a smart assistant<|eot_id|>"""
user_start = """<|start_header_id|>user<|end_header_id|>"""
user_end = """<|eot_id|>"""
assistant_start = """<|start_header_id|>assistant<|end_header_id|>"""
assistant_end = """<|eot_id|>"""

chat_session = ChatSession(
    chat_his=system_prompt, url="http://localhost:8017/generate", sampling_param=SamplingParams(do_sample=True)
)
chat_session.sampling_param.top_p = 0.7
chat_session.sampling_param.top_k = 12
# 修改采样参数
chat_session.sampling_param.stop_sequences = [assistant_end, " " + assistant_end]

chat_session.add_prompt(user_start)
chat_session.add_prompt("请问1+1+300+2=?")
chat_session.add_prompt(user_end)
chat_session.add_prompt(assistant_start)

print(chat_session.gen_int())
# 621
print(chat_session.generate(max_new_tokens=100))
# Let me calculate that for you! 😊
# 1 + 1 = 2
# 2 + 300 = 302
# 302 + 2 = 304
# So the answer is 304! 🎉<|eot_id|>


class Result(BaseModel):
    calculation_steps: str
    answer: int


print(chat_session.gen_json_object(Result, max_new_tokens=100))
# {"calculation_steps": "1+1+300+2", "answer": 304}
