import os
import cv2

# 设置文件夹路径 (使用原始字符串避免转义问题)
folder_path = r'D:\python-learn\K\seg-MOGA\outputs\lizi_UKAN\out_val'  #预测结果mask的路径
input_folder_path = r'D:\python-learn\K\seg-MOGA\inputs\lizi\images' #输入图片的路径
save_folder_path = r'D:\python-learn\K\seg-MOGA\UKAN\out_val_resiz' #重新resize结果

# 检查目录是否存在
print(f"🔍 检查目录...")
print(f"预测结果目录: {folder_path} - {'存在' if os.path.exists(folder_path) else '不存在'}")
print(f"输入图片目录: {input_folder_path} - {'存在' if os.path.exists(input_folder_path) else '不存在'}")

# 创建输出目录
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)
    print(f"✅ 创建输出目录: {save_folder_path}")

# 检查预测结果目录是否存在
if not os.path.exists(folder_path):
    print(f"❌ 预测结果目录不存在: {folder_path}")
    exit(1)

# 获取所有图片的路径
image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.png', '.jpg', '.jpeg'))]
print(f"📊 找到 {len(image_paths)} 个预测结果文件")

# 读取并处理每张图片
processed_count = 0
error_count = 0

for image_path in image_paths:
    name = os.path.basename(image_path).replace('.jpg', '.png')
    input_path = os.path.join(input_folder_path, name)
    
    try:
        # 读取原始图像和预测mask
        image = cv2.imread(input_path)
        mask = cv2.imread(image_path, 0)
        
        # 检查图像是否成功读取
        if image is None:
            print(f"⚠️ 无法读取输入图像: {name}")
            error_count += 1
            continue
            
        if mask is None:
            print(f"⚠️ 无法读取预测mask: {name}")
            error_count += 1
            continue
        
        # 获取输入图片的尺寸
        base_height, base_width = image.shape[:2]

        # 设置需要调整的图片的目标尺寸，这里我们使用基准图片的宽度和高度
        new_width = base_width
        new_height = base_height

        # 使用cv2.resize()调整图片大小
        resized_mask = cv2.resize(mask, (new_width, new_height))
        
        # 保存调整后的mask
        output_path = os.path.join(save_folder_path, name)
        cv2.imwrite(output_path, resized_mask)
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"✅ 已处理 {processed_count} 个文件...")
            
    except Exception as e:
        print(f"❌ 处理文件 {name} 时出错: {e}")
        error_count += 1

print(f"\n📊 处理完成:")
print(f"   成功处理: {processed_count} 个文件")
print(f"   错误文件: {error_count} 个文件")
print(f"   输出目录: {save_folder_path}")