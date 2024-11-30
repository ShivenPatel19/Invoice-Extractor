
# Invoice Extractor API

Welcome to the **Invoice Extractor API**! This API allows you to process PDF and image files (JPG, JPEG and PNG) and extract relevant information from invoices.

## Table of Contents
1. [Overview](#overview)
2. [API Endpoint](#api-endpoint)
3. [How to Use](#how-to-use)
   - [Using `curl`](#using-curl)
   - [Using Python](#using-python)
4. [Supported File Types](#supported-file-types)
5. [Error Handling](#error-handling)
6. [Example Requests](#example-requests)

## Overview
The Invoice Extractor API processes uploaded invoice files and extracts key data from them. It accepts PDF and image files, handling them as follows:
- **PDF files**: Converted to images for processing, with each page handled individually.
- **Image files**: Processed directly for data extraction.

## API Endpoint
```
POST https://invoice-extractor-api.onrender.com/process-invoice/
```

### Request Headers
- **Content-Type**: `multipart/form-data` (automatically set by `curl` or `requests` when uploading files)

## How to Use

### Using `curl`
To upload an invoice PDF using `curl`, run the following command in your terminal:

```bash
curl -X POST -F "file=@invoice.pdf" https://invoice-extractor-api.onrender.com/process-invoice/
```

Replace `invoice.pdf` with the path to your PDF or image file.

### Using Python
You can also use Python with the `requests` library to upload a file:

```python
import requests

# URL of the API
url = "https://invoice-extractor-api.onrender.com/process-invoice/"

# Open the file in binary mode
with open('D:\PaddleOCR\Gemini+OCR\images\image1.pdf', 'rb') as file:
    # Prepare the file for upload with the correct MIME type
    files = {'file': ('image1.pdf', file, 'application/pdf')}
    
    # Send a POST request to the API
    response = requests.post(url, files=files)

# Print the response
print(response.json())
```

**Note**: Ensure you have the `requests` library installed (`pip install requests`).

## Supported File Types
- **PDF**: `application/pdf`
- **JPEG**: `image/jpeg`
- **PNG**: `image/png`

**Note**: The API only processes PDF, JPEG, and PNG files. Other file types will result in an error.

## Error Handling
- **Unsupported File Type**: If a file type other than PDF, JPEG, or PNG is uploaded, the server will respond with:
  ```json
  {
    "detail": "Unsupported file type. Only PDF, JPEG, and PNG are allowed."
  }
  ```

- **Empty or Invalid PDF**: If a PDF file is empty or cannot be processed, the response will be:
  ```json
  {
    "detail": "The uploaded PDF is empty or invalid."
  }
  ```

## Example Requests

### Using `curl`
```bash
# Uploading a PDF file
curl -X POST -F "file=@path/to/your/file.pdf" https://invoice-extractor-api.onrender.com/process-invoice/

# Uploading a JPEG image
curl -X POST -F "file=@path/to/your/file.jpeg" https://invoice-extractor-api.onrender.com/process-invoice/
```

### Using Python
```python
# Python code snippet for PDF
files = {'file': ('file.pdf', open('path/to/your/file.pdf', 'rb'), 'application/pdf')}
response = requests.post(url, files=files)
print(response.json())

# Python code snippet for JPEG
files = {'file': ('file.jpeg', open('path/to/your/file.jpeg', 'rb'), 'image/jpeg')}
response = requests.post(url, files=files)
print(response.json())
```

## Conclusion
The Invoice Extractor API is a simple and effective way to automate the extraction of data from invoice files. Whether you are using `curl` or Python, the API is easy to integrate into your workflows.
