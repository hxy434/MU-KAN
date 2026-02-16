import base64
from os import access
import urllib
import requests
import json
import cv2


def process(image_path, access_token):
        
    url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general?access_token=%s" % access_token
    
    image_base64 = get_file_content_as_base64(image_path, True)
    payload = 'image=%s&detect_direction=false&detect_language=false&vertexes_location=false&paragraph=false&probability=false' % image_base64
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'Authorization': 'Bearer '
    }
    
    response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))
    
    # print(response.text)
    result = json.loads(response.text)
    # print(result)

    return result
    

def get_file_content_as_base64(path, urlencoded=False):

    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content

def is_number(s):
    try:
        float(s)  
        return True
    except ValueError:
        return False

if __name__ == '__main__':
    
    image_root = "orc_test_images"
    save_root = "orc_test_images/results"

    image_path = "orc_test_images/6.jpg"
    save_path = "orc_test_images/results/6.jpg"
    access_token = "24.7774c8e19ead52ab4031be41030b4c3b.2592000.1752027000.282335-119170487"

   
    result = process(image_path, access_token)

    print(result)

    image = cv2.imread(image_path)
    image_width = image.shape[1]
    image_height = image.shape[0]

    # μm, nm

    words_result_num = result["words_result_num"]
    words_result = result["words_result"]
  
    box = None
    box_number = None
    for item in words_result:
        words = item["words"]
        location = item["location"]
        top = location["top"]
        left = location["left"]
        width = location["width"]
        height = location["height"]

        box_words = [left, top, width, height]
        cv2.rectangle(image, (box_words[0], box_words[1]), (box_words[0]+box_words[2], box_words[1]+box_words[3]), color=(255, 0, 0))

        for ss in ["nm", "μm", "um"]:
            if ss in words:
                box = [left, top, width, height]
                if words == ss:
              
                    for item2 in words_result:
                        words2 = item2["words"]
                        location2 = item2["location"]
                        top2 = location2["top"]
                        left2 = location2["left"]
                        width2 = location2["width"]
                        height2 = location2["height"]
                        if is_number(words2):
                            if left2 < left and (left - left2) < (width + width2) and (top - top2) < (height + height2):
                                box_number = [left2, top2, width2, height2]
                    pass
                else:
                    s = words.replace(ss, "")
                    if is_number(s):
                        box_number = None
                    else:
                        for item2 in words_result:
                            words2 = item2["words"]
                            location2 = item2["location"]
                            top2 = location2["top"]
                            left2 = location2["left"]
                            width2 = location2["width"]
                            height2 = location2["height"]
                            if is_number(words2):
                                if left2 < left and (left - left2) < (width + width2) and (top - top2) < (height + height2):
                                    box_number = [left2, top2, width2, height2]

    if box is not None:
        cv2.rectangle(image, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color=(0, 0, 255))
    
    if box_number is not None:
        cv2.rectangle(image, (box_number[0], box_number[1]), (box_number[0]+box_number[2], box_number[1]+box_number[3]), color=(0, 255, 255))

    cv2.imwrite(save_path, image)
        
                    
                    



