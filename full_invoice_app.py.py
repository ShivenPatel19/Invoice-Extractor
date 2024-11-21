import cv2
import numpy as np
import os
from dotenv import load_dotenv
import streamlit as st      
import io
from PIL import Image
import google.generativeai as genai
from pdf2image import convert_from_bytes
import json

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def straighten_and_save_image(image_data, save_folder='straightened_images'):
    """
    Straightens the image by detecting the largest contour and performs a perspective transform.
    Saves the straightened image in the specified folder.

    Parameters:
        image_data (bytes): The input image data in bytes.
        save_folder (str): Folder to save the straightened image.

    Returns:
        str: Path to the saved straightened image.
    """
    # Convert bytes data to a numpy array and then read it using OpenCV
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Image not found or unable to load.")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding for edge detection
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check for contours
    if not contours:
        raise ValueError("No contours found in the image.")

    # Sort contours by area and get the largest one (the receipt)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    # Get the bounding box of the largest contour
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # We found a rectangle
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")

            # Order points (top-left, top-right, bottom-right, bottom-left)
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  # top-left
            rect[2] = pts[np.argmax(s)]  # bottom-right

            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  # top-right
            rect[3] = pts[np.argmax(diff)]  # bottom-left

            # Compute width and height of new image
            widthA = np.linalg.norm(rect[1] - rect[0])
            widthB = np.linalg.norm(rect[2] - rect[3])
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.linalg.norm(rect[3] - rect[0])
            heightB = np.linalg.norm(rect[2] - rect[1])
            maxHeight = max(int(heightA), int(heightB))

            # Set destination points for perspective transform
            dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

            # Perform perspective transform
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

            # Ensure directory exists before saving
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            # Save the straightened image with a unique name based on timestamp or original name
            original_filename = "straightened_image.jpg"  # You can customize this as needed
            save_path = os.path.join(save_folder, original_filename)
            save_status = cv2.imwrite(save_path, warped)

            if save_status:
                return save_path
            else:
                raise IOError("Error saving the straightened image.")

    raise ValueError("Unable to straighten the image.")

# Function to load OpenAI model and get responses
def get_gemini_response(input_text, image_data, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        # Convert image data from bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Ensure that input_text and prompt are strings
        if not isinstance(input_text, str) or not isinstance(prompt, str):
            raise ValueError("Input text and prompt must be strings.")

        # Pass the input text, image object, and prompt to the model
        response = model.generate_content([input_text, image, prompt])
        
        # Check if the response is in a valid format
        if hasattr(response, 'text'):
            return response.text
        else:
            raise ValueError("Invalid response format from the model.")
    except Exception as e:
        return f"An error occurred: {str(e)}"

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        file_type = uploaded_file.type

        if file_type == "application/pdf":
            # Convert PDF to images
            images = convert_from_bytes(uploaded_file.getvalue())
            # Use the first page of the PDF for extraction or process all pages as needed
            image = images[0]  # Take the first page for processing (or iterate through all pages if needed)
            # Convert PIL image to bytes for OpenCV processing
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            bytes_data = buf.getvalue()
        else:
            # If it's not a PDF, assume it's an image
            bytes_data = uploaded_file.getvalue()

        return bytes_data
    else:
        raise FileNotFoundError("No file uploaded")

# Initialize Streamlit app
st.set_page_config(page_title="Invoice Extractor")
st.header("Invoice Extractor")

input_prompt = """
               You are an expert in understanding and extracting information from invoices.
               You will receive invoice images as input and need to provide accurate answers based on the extracted details.
               
               Scan the invoice completely and create a lookalike json structure that representation whole of the invoice in key:value pair.
               Create sub ditcionary for similar types of information. for example, shop details, customer details, bill details, items, taxes, amount, similarly other details if any.
               Be precise and perfect in matching the information from the invoice to generated structured copy of it.
               

                Additional instructions for Processing Invoices(Follow Strictly):

                1. Missing Fields: If any required fields or parameters are absent in the image, fill them in with the value "null".
                2. Item Consistency: Ensure consistency in the "items" section:
                    If the individual item total is missing, calculate it as quantity multiplied by price and provide the accurate result.
                3. Tax Calculations:
                    If multiple tax types are listed (e.g., CGST, SGST), sum them and confirm the accuracy of the total tax amount.
                    If tax percentages are provided, verify that they match the total tax amount.
                    If total taxes are absent, sum all listed taxes separately to determine the correct final tax amount. For example if  CGST is 14.26 and SGST is 14.26 then total tax is 28.52.
                    Ensure that the taxable amount (including any discounts) plus total taxes (including individual CGST and SGST) equals the final amount.
                4. Item Identification: Be aware that product or item names may be split across multiple lines in the same column. Treat each row as a continuation and consider it as a single item.
                5. Bill Generation Time: The time when the bill is generated or paid may appear anywhere on the bill. Identify it by recognizing common formats or searching for indicators like "am/AM" or "pm/PM".
                6. Data Verification: Whenever possible, verify that the extracted data aligns with expected values:
                    Check if the total amount matches the sum of item prices plus taxes, minus any applicable discounts.
                    Confirm that the total quantity matches the sum of quantities for each item if provided.
                    Cross-check all other values based on your expertise in understanding invoices and bills.
                7.  Subtotal calculation: Ignore extracting subtotal directly from the image.
                    Taxable value/amount in the bill is the actual subtotal. Subtotal can't be same as final amount, must check.
                    As subtotal plus taxes subtract discount(if given) gives the total final amount. Subtotal value should be added basis on this checks.
                    Calculate Subtotal: Ensure that subtotal is calculated as:
                    subtotal = final_total - tax_amount + discount_amount
                    Validation: Confirm that:
                    final_total = subtotal + tax_amount - discount_amount
                8.  Currency: Don't just use the currency sumbol in json as it is. Find out the name of that currency adn then add that name in currency.
                """ 

uploaded_file = st.file_uploader("Choose an invoice image or PDF...", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    try:
        # Extract the file name without extension
        file_name = uploaded_file.name.rsplit(".", 1)[0]  # Remove the file extension

        # Process the uploaded file
        image_data = input_image_setup(uploaded_file)
        straightened_image_path = straighten_and_save_image(image_data)
        st.image(straightened_image_path, use_column_width=True)

        initial_response = get_gemini_response("", image_data, input_prompt)
        flag = 1
        st.subheader("Extracted Invoice Information")
        st.write(initial_response)
        print("results:\n", initial_response)

        # Cleaning the initial_response to remove delimiters and language specifier
        json_content = initial_response.strip('```').replace('json\n', '', 1).strip()

        # Parsing the cleaned string to ensure it's valid JSON
        try:
            data = json.loads(json_content)
            print("Parsed JSON successfully!")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            data = None

        # If valid JSON, save it to a file with the name of the uploaded file
        if data:
            json_file_name = f"{file_name}.json"
            with open(json_file_name, "w") as json_file:
                json.dump(data, json_file, indent=4)
            print(f"JSON content saved to {json_file_name}")
            st.write(f"JSON content saved to {json_file_name}")

    except Exception as e:
        st.error(f"An error occurred during extraction: {str(e)}")

# # optional part!

# if flag == 0:
#     initial_response = ""

# st.subheader("Ask Specific Questions")

# query = st.text_input(
#     "Type your query based on the extracted information:", 
#     key="query",
#     disabled=not initial_response
# )

# if query:
#     try:
#         combined_input = f"{initial_response}\n\nUser Query: {query}"
#         imput_prompt_QNA = """You are given with the data and user query. You just need to answer based on the information provided. 
#                               Answer in a conversational manner, as if talking to a human. Any questions outside the information 
#                               in the invoice will be ignored and not answered. Thank you!"""

#         follow_up_response = get_gemini_response(combined_input, image_data, imput_prompt_QNA)
#         st.write(follow_up_response)

#     except Exception as e:
#         st.error(f"An error occurred during Q&A: {str(e)}")