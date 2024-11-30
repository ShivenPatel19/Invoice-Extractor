import cv2
import numpy as np
import os
from dotenv import load_dotenv
from PIL import Image
import io
import google.generativeai as genai
from pdf2image import convert_from_bytes
import json
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import logging
import uvicorn
from google.generativeai import GenerativeModel


# Initialize FastAPI app
app = FastAPI()

# Initialize Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY is missing from environment variables")
genai.configure(api_key=api_key)


# def straighten_and_save_image(image_data):
#     """
#     Straightens the image by detecting the largest contour and performs a perspective transform.
#     Returns the straightened image.
#     """
#     nparr = np.frombuffer(image_data, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     if img is None:
#         raise ValueError("Invalid image data or unable to load the image.")

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # cv2.imwrite("gray.png", gray)
#     # return gray

#     thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
#     # cv2.imwrite("straightened_image.png", thresh)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if not contours:
#         raise ValueError("No contours found in the image. The image might be too plain or improperly captured.")

#     # Process the largest contour
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

#     for contour in contours:
#         epsilon = 0.02 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)

#         if len(approx) == 4:
#             pts = approx.reshape(4, 2)
#             rect = np.zeros((4, 2), dtype="float32")

#             s = pts.sum(axis=1)
#             rect[0] = pts[np.argmin(s)]
#             rect[2] = pts[np.argmax(s)]
#             diff = np.diff(pts, axis=1)
#             rect[1] = pts[np.argmin(diff)]
#             rect[3] = pts[np.argmax(diff)]

#             widthA = np.linalg.norm(rect[1] - rect[0])
#             widthB = np.linalg.norm(rect[2] - rect[3])
#             maxWidth = max(int(widthA), int(widthB))

#             heightA = np.linalg.norm(rect[3] - rect[0])
#             heightB = np.linalg.norm(rect[2] - rect[1])
#             maxHeight = max(int(heightA), int(heightB))

#             dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
#             M = cv2.getPerspectiveTransform(rect, dst)
#             warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
#             return warped

#     raise ValueError("Unable to straighten the image. Ensure the image contains clear boundaries or edges.")


def get_gemini_response(input_text, image_data, prompt):
    """
    Loads the Gemini model and generates a response based on the provided input text, image data, and prompt.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        image = Image.open(io.BytesIO(image_data))
        response = model.generate_content([input_text, image, prompt])
        if hasattr(response, 'text'):
            return response.text
        else:
            raise ValueError("Invalid response format from the Gemini model.")
    except Exception as e:
        logging.error(f"Error with Gemini model: {str(e)}")
        raise ValueError("Failed to generate response from Gemini. Please try again later.")


def input_image_setup(uploaded_file):
    """
    Prepares the uploaded file for processing. Converts a PDF file to JPEG if necessary and processes all pages.
    """
    file_content = uploaded_file.file.read()
    file_type = uploaded_file.content_type

    if file_type == "application/pdf":
        # Convert the PDF to a list of images (one image per page)
        images = convert_from_bytes(file_content)
        if not images:
            raise HTTPException(status_code=400, detail="The uploaded PDF is empty or invalid.")
        # Return the list of image bytes (one for each page)
        image_bytes_list = []
        for image in images:
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            image_bytes_list.append(buf.getvalue())
        return image_bytes_list
    elif file_type in ["image/jpeg", "image/png"]:
        # If it's a single image, return the image byte data
        return [file_content]
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF, JPEG, and PNG are allowed.")


@app.post("/process-invoice/")
async def process_invoice(file: UploadFile):
    """
    Endpoint to process an uploaded invoice image or PDF and return extracted information in JSON format.
    """
    input_prompt = """
                You are an expert in understanding and extracting information from invoices.
                You will receive invoice images as input and need to provide accurate answers based on the extracted details.
                    
                Additional instructions for Processing Invoices(Follow Strictly):

                1. Missing Fields: If any required fields or parameters are absent in the image,
                    fill them in with the value "null" if the paramater accepts string,
                    else if it acepts number or integer of float just make its value as 0.0 in float.
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
    
    input_text = """
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
                        "total_amount": "26.45",
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
                """
    
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded.")

        # Prepare the uploaded file
        image_data_list = input_image_setup(file)

        all_page_data = []

        # Process each page's image
        for image_data in image_data_list:

            # Straighten the image
            # straightened_image = straighten_and_save_image(image_data)
            # _, buffer = cv2.imencode('.jpg', straightened_image)
            # image_bytes = buffer.tobytes()

            # Generate response using Gemini
            gemini_response = get_gemini_response(input_text, image_data, input_prompt)

            # Parse and clean the response
            json_content = gemini_response.strip('```').replace('json\n', '', 1).strip()
            page_data = json.loads(json_content)
            all_page_data.append(page_data)

        # Check if it's a single image or multi-page PDF
        if len(all_page_data) == 1:  # Single image case
            return JSONResponse(content=all_page_data[0])
        else:  # Multi-page PDF case
            # combined_data = combine_invoice_data(all_page_data)
            combined_data = process_merged_invoice(all_page_data)
            json_combined_data = combined_data.strip('```').replace('json\n', '', 1).strip()
            json_combined_data = json.loads(json_combined_data)
            print(json_combined_data)
            return JSONResponse(content=json_combined_data)

    except HTTPException as e:
        logging.error(f"HTTP Exception: {e.detail}")
        raise e
    except json.JSONDecodeError:
        logging.error("Failed to decode Gemini's response into JSON.")
        raise HTTPException(status_code=500, detail="Error decoding AI response. Please check the input.")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing invoice: {str(e)}")

# def combine_invoice_data(all_page_data):
#     """
#     Combines data from all pages into one unified result.
#     Assumes that the structure of each page's data is consistent.
#     """
#     def safe_float(value):
#         """Converts a value to float safely, defaulting to 0.0 if conversion fails."""
#         try:
#             return float(value)
#         except (TypeError, ValueError):
#             return 0.0

#     combined_data = {
#         "merchant": all_page_data[0].get("merchant", {}),
#         "receipt_details": all_page_data[0].get("receipt_details", {}),
#         "items": [],
#         "subtotal": 0.0,
#         "tax_amount": 0.0,
#         "discount_amount": 0.0,
#         "final_total": 0.0,
#     }

#     # Combine items and calculate totals
#     for page_data in all_page_data:
#         items = page_data.get("items", [])
#         combined_data["items"].extend(items)
        
#         # Safely add numeric values, converting them if necessary
#         combined_data["subtotal"] += safe_float(page_data.get("subtotal"))
#         combined_data["tax_amount"] += safe_float(page_data.get("tax_amount"))
#         combined_data["discount_amount"] += safe_float(page_data.get("discount_amount"))
#         combined_data["final_total"] += safe_float(page_data.get("final_total"))

#     return combined_data

def process_merged_invoice(all_page_texts):
    """
    Merges all invoice page texts and processes them using a text-based Gemini model.

    Args:
        all_page_texts (list): List of text content from all invoice pages.
    
    Returns:
        dict: Consolidated invoice data in structured format.
    """
    # Step 1: Merge all the pages into a single text
    merged_text_parts = []

    for page_dict in all_page_texts:
        # Convert the dictionary into a readable string format
        page_text = "\n".join(
            f"{key}: {value}" for key, value in page_dict.items()
        )
        merged_text_parts.append(page_text)

    # Combine all pages with double newline as separator
    merged_text = "\n\n".join(merged_text_parts)

    # Step 2: Define a detailed and accurate prompt for the LLM
    prompt = f"""
                You are an expert in understanding information from invoices.    

                Text from all invoice pages:
                {merged_text}
                """ + \
                """
                Provide the consolidated output in the following structured JSON format without any Butyfications.
                NOTE that i want pure json structure where there esist only keyvalue pair and nothing more.:
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
                        "total_amount": 26.45,
                        "taxes": 3.45,
                        "discounts": 2.00
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
                    "subtotal": 25.00,
                    "tax_amount": 3.45,
                    "discount_amount": 2.00,
                    "final_total": 26.45
                }
                Do necessary checksum validation after merging results from multiple pages, tp check data integrity.

                Additional instructions for Processing consolidated output json format(Follow Strictly):

                1. Missing Fields: If any required fields or parameters are absent in the image,
                    fill them in with the value "null" if the parameter accepts a string,
                    else if it accepts a number or integer or float just make its value as 0.0 in float.
                2. Item Consistency: Ensure consistency in the "items" section:
                    If the individual item total is missing, calculate it as quantity multiplied by price and provide the accurate result.
                3. Tax Calculations:
                    If multiple tax types are listed (e.g., CGST, SGST), sum them and confirm the accuracy of the total tax amount.
                    If tax percentages are provided, verify that they match the total tax amount.
                    If total taxes are absent, sum all listed taxes separately to determine the correct final tax amount. For example if CGST is 14.26 and SGST is 14.26 then total tax is 28.52.
                    Ensure that the taxable amount (including any discounts) plus total taxes (including individual CGST and SGST) equals the final amount.
                4. Item Identification: Be aware that product or item names may be split across multiple lines in the same column. Treat each row as a continuation and consider it as a single item.
                5. Bill Generation Time: The time when the bill is generated or paid may appear anywhere on the bill. Identify it by recognizing common formats or searching for indicators like "am/AM" or "pm/PM".
                6. Data Verification: Whenever possible, verify that the extracted data aligns with expected values:
                    Check if the total amount matches the sum of item prices plus taxes, minus any applicable discounts.
                    Confirm that the total quantity matches the sum of quantities for each item if provided.
                    Cross-check all other values based on your expertise in understanding invoices and bills.
                7. Subtotal calculation: Ignore extracting subtotal directly from the image.
                    Taxable value/amount in the bill is the actual subtotal. Subtotal can't be same as final amount, must check.
                    As subtotal plus taxes subtract discount(if given) gives the total final amount. Subtotal value should be added based on these checks.
                    NOTE: Final amount or Final total(final_total) is the amount the customer had paid.
                    NOTE: Calculate Subtotal: Ensure that subtotal is calculated as:(it should match)
                    subtotal = final_total - tax_amount + discount_amount
                    NOTE: Validation: Confirm that:(it should match)
                    final_total = subtotal + tax_amount - discount_amount
                8. Currency: Don't just use the currency symbol in JSON as it is. Find out the name of that currency and then add that name in currency.
                """
    
    try:
        gemini_model = GenerativeModel('gemini-1.5-flash')
        response = gemini_model.generate_content(contents=prompt)
        # print("response--------------",response.text)
        response = response.text.strip('```').replace('json\n', '', 1).strip()
        return response
    except Exception as e:
        logging.error(f"Error processing invoice: {e}")
        raise ValueError("Failed to process the invoice with Gemini model.")


# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
