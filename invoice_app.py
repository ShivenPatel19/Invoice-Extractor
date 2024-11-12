import cv2
import numpy as np
import os
from dotenv import load_dotenv
import streamlit as st
import io
from PIL import Image
import google.generativeai as genai
from pdf2image import convert_from_bytes

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
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Image not found or unable to load.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in the image.")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")

            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            widthA = np.linalg.norm(rect[1] - rect[0])
            widthB = np.linalg.norm(rect[2] - rect[3])
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.linalg.norm(rect[3] - rect[0])
            heightB = np.linalg.norm(rect[2] - rect[1])
            maxHeight = max(int(heightA), int(heightB))

            dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            original_filename = "straightened_image.jpg"
            save_path = os.path.join(save_folder, original_filename)
            save_status = cv2.imwrite(save_path, warped)

            if save_status:
                return save_path
            else:
                raise IOError("Error saving the straightened image.")

    raise ValueError("Unable to straighten the image.")

def get_gemini_response(input_text, image_data, prompt):
    """
    Loads the Gemini model and generates a response based on the provided input text, image data, and prompt.

    Parameters:
        input_text (str): The text input for the model.
        image_data (bytes): The image data in bytes format.
        prompt (str): The prompt to instruct the model's response behavior.

    Returns:
        str: Generated response text from the model.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        image = Image.open(io.BytesIO(image_data))
        
        if not isinstance(input_text, str) or not isinstance(prompt, str):
            raise ValueError("Input text and prompt must be strings.")

        response = model.generate_content([input_text, image, prompt])
        
        if hasattr(response, 'text'):
            return response.text
        else:
            raise ValueError("Invalid response format from the model.")
    except Exception as e:
        return f"An error occurred: {str(e)}"

def input_image_setup(uploaded_file):
    """
    Prepares the uploaded file for processing by converting it to bytes data. If the file is a PDF,
    converts the first page to a JPEG format.

    Parameters:
        uploaded_file (File): The file uploaded by the user.

    Returns:
        bytes: The prepared image data in bytes format.
    """
    if uploaded_file is not None:
        file_type = uploaded_file.type

        if file_type == "application/pdf":
            images = convert_from_bytes(uploaded_file.getvalue())
            image = images[0]
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            bytes_data = buf.getvalue()
        else:
            bytes_data = uploaded_file.getvalue()

        return bytes_data
    else:
        raise FileNotFoundError("No file uploaded")

# Initialize Streamlit app
st.set_page_config(page_title="Invoice Extractor")
st.header("Invoice Extractor")

flag = 0
input_prompt = """
               You are an expert in understanding and extracting information from invoices.
               You will receive invoice images as input and need to provide accurate answers based on the extracted details.
               For that you must carefully scan the document and return the results in the following structured format: 
                {
                    "merchant": {
                        "name": "Store Name",
                        "address": "123 Store St, City, ZIP",
                        "contact": "Phone number"
                    },
                    "receipt_details": {
                        "receipt_number": "XYZ123456",
                        "date": "YYYY-MM-DD",
                        "time": "HH:MM:SS",
                        "payment_method": "Credit Card / Cash / Other",
                        "currency": "USD",
                        "total_amount": "123.45",
                        "taxes": "3.45",
                        "discounts": "2.00"
                    },
                    "items": [
                        {
                            "name": "Item 1",
                            "quantity": "1",
                            "price": "10.00",
                            "total": "10.00"
                        },
                        {
                            "name": "Item 2",
                            "quantity": "2",
                            "price": "7.50",
                            "total": "15.00"
                        }
                    ],
                    "subtotal": "25.00",
                    "tax_amount": "3.45",
                    "discount_amount": "2.00",
                    "final_total": "26.45"
                }
                
                Additionally, Instructions for Processing Invoices(Follow Strictly):

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
                """ 

uploaded_file = st.file_uploader("Choose an invoice image or PDF...", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    try:
        image_data = input_image_setup(uploaded_file)
        straightened_image_path = straighten_and_save_image(image_data)
        st.image(straightened_image_path, use_column_width=True)

        initial_response = get_gemini_response("", image_data, input_prompt)
        flag = 1
        st.subheader("Extracted Invoice Information")
        st.write(initial_response)
        
    except Exception as e:
        st.error(f"An error occurred during extraction: {str(e)}")

if flag == 0:
    initial_response = ""

st.subheader("Ask Specific Questions")

query = st.text_input(
    "Type your query based on the extracted information:", 
    key="query",
    disabled=not initial_response
)

if query:
    try:
        combined_input = f"{initial_response}\n\nUser Query: {query}"
        imput_prompt_QNA = """You are given with the data and user query. You just need to answer based on the information provided. 
                              Answer in a conversational manner, as if talking to a human. Any questions outside the information 
                              in the invoice will be ignored and not answered. Thank you!"""

        follow_up_response = get_gemini_response(combined_input, image_data, imput_prompt_QNA)
        st.write(follow_up_response)

    except Exception as e:
        st.error(f"An error occurred during Q&A: {str(e)}")