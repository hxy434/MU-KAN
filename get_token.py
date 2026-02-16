#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to get Baidu AI access_token
Use API Key and Secret Key to obtain a new access token
"""

import requests
import json

def get_access_token(api_key, secret_key):
    """Get Baidu OCR access_token"""
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
            print(f"✅ Successfully obtained access_token: {result['access_token']}")
            print(f"📅 Expiration time: {result.get('expires_in', 'Unknown')} seconds")
            return result["access_token"]
        else:
            print(f"❌ Failed to obtain access_token: {result}")
            return None
    except Exception as e:
        print(f"❌ Exception occurred while getting access_token: {e}")
        return None

def main():
    print("🔑 Baidu AI Access Token Retrieval Tool")
    print("=" * 50)
    
    # API Key and Secret Key obtained from image
    api_key = "KfmMn5t7vxT124ZmyNjhhhlf"
    secret_key = "kjl6Ic65AO2hoVe65TyUwaoNRGOrpmf6"
    
    print(f"📋 API Key: {api_key}")
    print(f"🔐 Secret Key: {secret_key}")
    print("-" * 50)
    
    # Get access token
    access_token = get_access_token(api_key, secret_key)
    
    if access_token:
        print("\n📝 Usage Instructions:")
        print("1. Copy the access_token above into your code")
        print("2. Update the ACCESS_TOKEN variable in particle_analysis_demo.py")
        print("3. Update the ACCESS_TOKEN variable in mock_ocr.py")
        print("\n💡 Note: The access_token is typically valid for 30 days and needs to be re-obtained after expiration")
        
        # Generate example code for updates
        print("\n🔄 Files to be updated:")
        print("1. particle_analysis_demo.py line 32:")
        print(f"   ACCESS_TOKEN = \"{access_token}\"")
        print("2. mock_ocr.py line 18:")
        print(f"   ACCESS_TOKEN = \"{access_token}\"")
    else:
        print("❌ Failed to retrieve token, please check if API Key and Secret Key are correct")

if __name__ == "__main__":
    main()
