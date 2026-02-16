from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import base64
from io import BytesIO
from PIL import Image
import os
import shutil
import tempfile
import uuid

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import analysis scripts
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from particle_analysis_demo import analyze_sem_image_ukan_only, analyze_sem_image_yolo_only

# Global variable to store particle data
particle_data_cache = {}

# 工具函数：图片转base64
def imgfile2b64(path):
    with open(path, 'rb') as f:
        return 'data:image/png;base64,' + base64.b64encode(f.read()).decode()
# 工具函数：文本文件转字符串
def txtfile2str(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def parse_particle_details_from_report(report_path):
    """Parse particle details from analysis report"""
    particle_details = {}
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for detailed particle information section
        if "Detailed Particle Information:" in content:
            lines = content.split('\n')
            in_particle_section = False
            current_particle = None
            
            for line in lines:
                line = line.strip()
                if "Detailed Particle Information:" in line:
                    in_particle_section = True
                    continue
                
                if in_particle_section and line.startswith("Particle"):
                    # Extract particle number
                    import re
                    match = re.search(r'Particle\s*(\d+):', line)
                    if match:
                        current_particle = int(match.group(1))
                        particle_details[current_particle] = {}
                
                elif in_particle_section and current_particle and line.startswith("Area:"):
                    particle_details[current_particle]['area'] = line.split("Area:")[1].strip()
                elif in_particle_section and current_particle and line.startswith("Diameter:"):
                    particle_details[current_particle]['diameter'] = line.split("Diameter:")[1].strip()
                elif in_particle_section and current_particle and line.startswith("Circularity:"):
                    particle_details[current_particle]['circularity'] = line.split("Circularity:")[1].strip()
                
                # If we encounter empty line or other section, exit particle info parsing
                elif in_particle_section and (not line or line.startswith("Detection Results Statistics:")):
                    break
        
        # If no detailed particle information found, try to extract from statistics
        if not particle_details:
            # Extract unit information
            unit_match = re.search(r'Scale Unit:\s*(\w+)', content)
            unit = unit_match.group(1) if unit_match else "pixel"
            
            # Extract number of detected particles
            count_match = re.search(r'Number of Particles Detected:\s*(\d+)', content)
            if count_match:
                count = int(count_match.group(1))
                # Create basic info for each particle
                for i in range(1, count + 1):
                    particle_details[i] = {
                        'area': f"Average Area (Unit: {unit}²)",
                        'diameter': f"Average Diameter (Unit: {unit})",
                        'circularity': "Average Circularity"
                    }
    
    except Exception as e:
        print(f"Failed to parse particle details: {e}")
        particle_details = {}
    
    return particle_details

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), model: str = Form('MU-KAN')):
    print("Received analysis request")
    try:
        print("Saving image to temporary directory")
        temp_dir = tempfile.mkdtemp()
        img_ext = os.path.splitext(file.filename)[-1] or '.png'
        img_path = os.path.join(temp_dir, f"input_{uuid.uuid4().hex}{img_ext}")
        with open(img_path, 'wb') as f_out:
            shutil.copyfileobj(file.file, f_out)
            print(f"Image saved: {img_path}")

        print("Calling analysis function")
        if model == 'MU-KAN':
            output_dir, result_dict = analyze_sem_image_ukan_only(img_path)
        else:
            output_dir, result_dict = analyze_sem_image_yolo_only(img_path)
        print("Analysis function call completed")

        if not output_dir or not os.path.exists(output_dir):
            print("Analysis failed, no output directory generated")
            return {"error": "Analysis failed, no output directory generated"}
        
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        def find_file(patterns):
            for pat in patterns:
                for fname in os.listdir(output_dir):
                    if fname.startswith(pat):
                        return os.path.join(output_dir, fname)
            return None

        print("Collecting analysis results")
        result = {}
        result['original'] = imgfile2b64(img_path)
        prob_path = find_file([f"{base_name}_probability_map"])
        result['probability'] = imgfile2b64(prob_path) if prob_path else ''
        binary_path = find_file([f"{base_name}_binary_mask"])
        result['binary'] = imgfile2b64(binary_path) if binary_path else ''
        vis_path = find_file([f"{base_name}_visualization"])
        result['overlay'] = imgfile2b64(vis_path) if vis_path else ''
        dist_path = find_file([f"particle_distribution_{base_name}"])
        result['distribution'] = imgfile2b64(dist_path) if dist_path else ''
        contour_path = find_file([f"{base_name}_contour"])
        result['contour'] = imgfile2b64(contour_path) if contour_path else ''
        numbered_path = find_file([f"{base_name}_numbered"])
        result['numbered'] = imgfile2b64(numbered_path) if numbered_path else ''
        # Add scale detection visualization image
        scale_detection_path = find_file([f"{base_name}_improved_detection"])
        result['scale_detection'] = imgfile2b64(scale_detection_path) if scale_detection_path else ''
        report_path = find_file([f"analysis_report_{base_name}"])
        result['report'] = txtfile2str(report_path) if report_path else ''
        if report_path:
            with open(report_path, 'rb') as f:
                result['analysisFile'] = 'data:text/plain;base64,' + base64.b64encode(f.read()).decode()
        else:
            result['analysisFile'] = ''
        result['mask'] = result['binary']
        result['stats'] = result['distribution']
        # Merge scale information
        if result_dict and 'scale_info' in result_dict:
            result['scale_info'] = result_dict['scale_info']
        
        # Parse analysis report to extract particle details
        if report_path:
            particle_details = parse_particle_details_from_report(report_path)
            result['particle_details'] = particle_details
            # Cache particle data
            session_id = str(uuid.uuid4())
            particle_data_cache[session_id] = particle_details
            result['session_id'] = session_id
        
        print("Analysis results collection completed, preparing to clean up temporary files")
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            print("Failed to clean up temporary files")
            pass
        print("Analysis process completed, returning results")
        return result
    except Exception as e:
        import traceback
        print("Exception occurred during analysis:", e)
        traceback.print_exc()
        return {"error": f"Analysis failed: {str(e)}"}

@app.post("/api/particle-details")
async def get_particle_details(particle_id: int = Form(...), session_id: str = Form(...)):
    """Get detailed information for specified particle"""
    try:
        if session_id not in particle_data_cache:
            return {"error": "Session expired or does not exist"}
        
        particle_data = particle_data_cache[session_id]
        if particle_id not in particle_data:
            return {"error": f"Particle {particle_id} does not exist"}
        
        particle_info = particle_data[particle_id]
        return {
            "success": True,
            "particle": {
                "id": particle_id,
                "area": particle_info.get('area', 'Unknown'),
                "diameter": particle_info.get('diameter', 'Unknown'),
                "circularity": particle_info.get('circularity', 'Unknown')
            }
        }
    except Exception as e:
        print(f"Error occurred while getting particle details: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 
