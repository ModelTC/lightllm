import os
import importlib.util

# 默认的load_image函数
def default_load_image(image_path):
    print(f"Loading image using default function: {image_path}")
    # 默认的加载图像逻辑（这里只是示例）
    return image_path


# 用户提供的目录路径
directory = "./user_directory"

# 设定默认的load_image函数为default_load_image
load_image = default_load_image

# 检查目录中是否有pre_process.py文件
pre_process_path = os.path.join(directory, "pre_process.py")

if os.path.exists(pre_process_path):
    print(f"Found pre_process.py in {directory}, attempting to load load_image from it.")

    # 使用importlib来加载模块
    spec = importlib.util.spec_from_file_location("pre_process", pre_process_path)
    pre_process = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pre_process)

    # 如果pre_process.py中有load_image函数，则替换默认函数
    if hasattr(pre_process, "load_image"):
        load_image = pre_process.load_image
        print("load_image function replaced by the one in pre_process.py.")
    else:
        print("load_image function not found in pre_process.py.")
else:
    print(f"pre_process.py not found in {directory}, using default load_image.")

# 使用当前的load_image函数
image = load_image("path/to/image.jpg")
