#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取百度OCR access_token
"""

import requests
import json

def get_access_token(api_key, secret_key):
    """获取百度OCR access_token"""
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": api_key,
        "client_secret": secret_key
    }
    
    try:
        print(f"🔍 正在获取access_token...")
        print(f"   API Key: {api_key[:10]}...")
        print(f"   Secret Key: {secret_key[:10]}...")
        
        response = requests.post(url, params=params)
        result = response.json()
        
        if "access_token" in result:
            access_token = result["access_token"]
            expires_in = result.get("expires_in", "未知")
            print(f"✅ 成功获取access_token!")
            print(f"   Token: {access_token}")
            print(f"   有效期: {expires_in} 秒")
            return access_token
        else:
            print(f"❌ 获取access_token失败: {result}")
            return None
            
    except Exception as e:
        print(f"❌ 获取access_token异常: {e}")
        return None

def test_ocr_api(access_token):
    """测试OCR API是否可用"""
    print(f"\n🧪 测试OCR API...")
    
    # 创建一个简单的测试图像（1x1像素的白色图像）
    import base64
    test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic?access_token={access_token}"
    
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }
    
    data = {
        'image': test_image_base64
    }
    
    try:
        response = requests.post(url, headers=headers, data=data)
        result = response.json()
        
        if "error_code" in result:
            print(f"❌ OCR API测试失败: {result}")
            return False
        else:
            print(f"✅ OCR API测试成功!")
            return True
            
    except Exception as e:
        print(f"❌ OCR API测试异常: {e}")
        return False

if __name__ == '__main__':
    print("🔑 百度OCR Access Token 获取工具")
    print("="*50)
    
    # 请在这里填入您的API Key和Secret Key
    print("📝 请从百度AI控制台获取以下信息:")
    print("   1. 登录 https://console.bce.baidu.com/")
    print("   2. 进入 '文字识别' -> '应用列表'")
    print("   3. 找到您的应用，获取API Key和Secret Key")
    print()
    
    # 这里需要您手动填入
    api_key = input("请输入您的API Key: ").strip()
    secret_key = input("请输入您的Secret Key: ").strip()
    
    if not api_key or not secret_key:
        print("❌ API Key或Secret Key不能为空")
        exit(1)
    
    # 获取access_token
    access_token = get_access_token(api_key, secret_key)
    
    if access_token:
        # 测试API
        if test_ocr_api(access_token):
            print(f"\n🎉 成功! 您的新access_token是:")
            print(f"   {access_token}")
            print(f"\n💡 请将这个token更新到您的代码中")
        else:
            print(f"\n⚠️ 获取token成功，但API测试失败，请检查您的配额")
    else:
        print(f"\n❌ 获取access_token失败，请检查您的API Key和Secret Key")
