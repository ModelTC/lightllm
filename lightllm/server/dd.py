def show():
    import sys
    import traceback

    # 获取当前所有线程的调用栈帧
    for thread_id, frame in sys._current_frames().items():
        print("Stack for thread {}:".format(thread_id))
        # 打印调用栈信息
        traceback.print_stack(frame)
        print("")
