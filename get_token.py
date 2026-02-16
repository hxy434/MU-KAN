#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取百度AI access_token的脚本
使用API Key和Secret Key获取新的access token
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
        response = requests.post(url, params=params)
        result = response.json()
        if "access_token" in result:
            print(f"✅ 成功获取access_token: {result['access_token']}")
            print(f"📅 过期时间: {result.get('expires_in', '未知')}秒")
            return result["access_token"]
        else:
            print(f"❌ 获取access_token失败: {result}")
            return None
    except Exception as e:
        print(f"❌ 获取access_token异常: {e}")
        return None

def main():
    print("🔑 百度AI Access Token 获取工具")
    print("=" * 50)
    
    # 从图片中获取的API Key和Secret Key
    api_key = "KfmMn5t7vxT124ZmyNjhhhlf"
    secret_key = "kjl6Ic65AO2hoVe65TyUwaoNRGOrpmf6"
    
    print(f"📋 API Key: {api_key}")
    print(f"🔐 Secret Key: {secret_key}")
    print("-" * 50)
    
    # 获取access token
    access_token = get_access_token(api_key, secret_key)
    
    if access_token:
        print("\n📝 使用说明:")
        print("1. 将上面的access_token复制到你的代码中")
        print("2. 在 particle_analysis_demo.py 中更新 ACCESS_TOKEN 变量")
        print("3. 在 mock_ocr.py 中更新 ACCESS_TOKEN 变量")
        print("\n💡 注意: access_token通常有效期为30天，过期后需要重新获取")
        
        # 生成更新代码的示例
        print("\n🔄 需要更新的文件:")
        print("1. particle_analysis_demo.py 第32行:")
        print(f"   ACCESS_TOKEN = \"{access_token}\"")
        print("2. mock_ocr.py 第18行:")
        print(f"   ACCESS_TOKEN = \"{access_token}\"")
    else:
        print("❌ 获取token失败，请检查API Key和Secret Key是否正确")

if __name__ == "__main__":
    main() 