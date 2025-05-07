..multimodal_model_quickstart.rst
-------------------------

下载多模态模型（如llava系列、internvl系列、qwen_vl系列等）的模型以后，在终端使用下面的代码部署API服务：

.. code-block:: console

    $ python -m lightllm.server.api_server --model_dir ~/models/llava-7b-chat --use_dynamic_prompt_cache --enable_multimodal

.. note::
    上面代码中的 ``--model_dir`` 参数需要修改为你本机实际的模型路径。
