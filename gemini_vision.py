import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

def analyze_image(image_path: str, prompt: str = "Describe this image in a single paragraph. The entire response must be in Markdown.") -> str:
    """
    Analyze an image using Gemini Pro Vision model.
    
    Args:
        image_path (str): Path to the image file
        prompt (str): Custom prompt for image analysis (optional)
    
    Returns:
        str: Generated description of the image
    
    Example usage:
        # Basic usage
        result = analyze_image("path/to/image.jpg")
        
        # With custom prompt
        result = analyze_image(
            "path/to/image.jpg",
            "What objects can you identify in this image?"
        )
    """
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    try:
        image = Image.open(image_path)
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        raise Exception(f"Error analyzing image: {str(e)}")

if __name__ == "__main__":
    # Example usage
    try:
        # Single image analysis
        result = analyze_image("input_docs/prompt_royale.jpeg")
        print(f"Analysis Result:\n{result}")
        
        # Multiple images with different prompts
        images = [
            ("image1.jpg", "Describe the main subject"),
            ("image2.jpg", "List visible objects"),
        ]
        for img_path, prompt in images:
            result = analyze_image(img_path, prompt)
            print(f"\nAnalysis for {img_path}:\n{result}")
            
    except Exception as e:
        print(f"Error: {e}")