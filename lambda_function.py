import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import json
import cv2  



CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
           'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
           'River', 'SeaLake']

# --- BUSINESS LOGIC LAYER ---
# This dictionary maps the model's raw prediction to actionable insights.

INSIGHTS = {
    'AnnualCrop': {
        "description": "Detected cultivated land used for seasonal farming (e.g., corn, wheat, vegetables).",
        "recommendation": "Monitor soil moisture levels for irrigation scheduling. Check for pest outbreaks common in seasonal monocultures."
    },
    'Forest': {
        "description": "Detected dense natural forest or woodland area.",
        "recommendation": "Eligible for Carbon Credit verification. Monitor for illegal logging activities or encroachments using change detection."
    },
    'HerbaceousVegetation': {
        "description": "Detected natural grasslands, savannas, or non-woody plant cover.",
        "recommendation": "Assess suitability for livestock grazing. Monitor fire risks during dry seasons as this terrain is highly combustible."
    },
    'Highway': {
        "description": "Detected major transportation infrastructure (paved roads).",
        "recommendation": "Analyze traffic flow efficiency. Check for surface degradation or need for maintenance in this sector."
    },
    'Industrial': {
        "description": "Detected industrial facilities, warehouses, or factories.",
        "recommendation": "Verify environmental compliance regarding emissions. cross-reference with zoning laws for urban expansion planning."
    },
    'Pasture': {
        "description": "Detected grazing land for livestock.",
        "recommendation": "Rotate grazing schedules to prevent soil erosion. Test soil nutrient levels to ensure quality fodder."
    },
    'PermanentCrop': {
        "description": "Detected long-term agricultural plantations (e.g., vineyards, orchards, coffee).",
        "recommendation": "Focus on long-term disease prevention. Inspect irrigation infrastructure as these crops require consistent water delivery."
    },
    'Residential': {
        "description": "Detected human housing and settlement areas.",
        "recommendation": "Analyze population density for utility planning (water/electricity). Monitor for unauthorized urban sprawl."
    },
    'River': {
        "description": "Detected flowing water body.",
        "recommendation": "Monitor water levels for flood early warning systems. Test for upstream pollution or sediment runoff."
    },
    'SeaLake': {
        "description": "Detected large standing water body.",
        "recommendation": "Monitor for algal blooms or water quality changes. Surveillance required for illegal fishing or unauthorized dumping."
    }
}

print('Loading model...')
session = ort.InferenceSession('eurosat.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def check_blur(image_bytes, threshold=100.0):

    try:
        # Convert raw bytes to a list of numbers (numpy array)
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decoding the numbers into an image structure OpenCV understands
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return True # If we can't read it, assume it's bad
            
        # Converting to Grayscale (We only need light/dark for sharpness, not color)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculating Laplacian Variance (The "Sharpness Score")
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        print(f"Blur Score: {score}") 
        return score < threshold
    except Exception as e:
        print(f"Error in blur check: {e}")
        return True 

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((224, 224), Image.BICUBIC)
    
    # Normalizing the image
    img_data = np.array(image).astype('float32') / 255
    
    # Standardize using ImageNet stats 
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    
    return img_data

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



def lambda_handler(event, context):
    try:
        if 'image_bytes' not in event:
             return {"statusCode": 400, "body": json.dumps({"error": "No image found"})}

        image_data = event['image_bytes']

        if check_blur(image_data):
            return {
                "statusCode": 400, 
                "body": json.dumps({"error": "Image is too blurry. Please upload a clearer photo."})
            }

        # PREPARE THE IMAGE
        img_tensor = preprocess_image(image_data)
        outputs = session.run(None, {input_name: img_tensor})
        
        #  PROCESS RESULTS
        raw_logits = outputs[0][0]     # Get the raw numbers
        probabilities = softmax(raw_logits) # Convert to percentages
        
    
        chart_data = {}
        for i in range(len(CLASSES)):
            chart_data[CLASSES[i]] = float(probabilities[i])
            
        # Find the Winner
        prediction_index = np.argmax(probabilities)
        confidence = float(probabilities[prediction_index])
        predicted_class = CLASSES[prediction_index]
        
        # BUSINESS LOGIC LAYER
        logic = INSIGHTS.get(predicted_class, {
            "description": "Terrain type not recognized.",
            "recommendation": "Manual review required."
        })

        #  RETURN SUCCESS
        return {
            'statusCode': 200,
            'body': json.dumps({
                "class": predicted_class,
                "confidence": confidence,
                "description": logic["description"],
                "recommendation": logic["recommendation"],
                "chart_data": chart_data
            })
        }
        
    except Exception as e:
        #  RETURN ERROR
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }