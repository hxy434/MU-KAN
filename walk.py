from PIL import Image
import os
import glob

# 设置输入和输出文件夹路径
input_folder = "/mnt/bn/bruceyanglq/code/sphinx/seg/inputs/lizi/masks/0"  # 替换为你的输入文件夹路径
output_folder = "vis"  # 替换为你的输出文件夹路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 支持的图片格式
extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')

# 遍历所有图片文件
for ext in extensions:
    for img_path in glob.glob(os.path.join(input_folder, ext)):
        try:
            with Image.open(img_path) as img:
                # 转换为灰度图像
                gray_img = img.convert('L')
                # 应用二值化：像素值 > 1 的设为255，否则0
                binary_img = gray_img.point(lambda x: 255 if x > 1 else 0)
                # 构建输出路径
                filename = os.path.basename(img_path)
                output_path = os.path.join(output_folder, filename)
                # 保存处理后的图片
                binary_img.save(output_path)
                print(f"处理完成: {filename}")
        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {e}")