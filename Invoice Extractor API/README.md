
# Invoice Extractor API

Welcome to the **Invoice Extractor API**! This API allows you to process PDF and image files (JPG, JPEG, and PNG) and extract relevant information from invoices.

For a quick overview of how the **API works**, check out the [video demo](https://drive.google.com/file/d/1LdBAoG44ab5M5h7TZyJwxTEHXOJjGlBy/view?usp=drive_link).

## Table of Contents
1. [Overview](#overview)
2. [API Endpoints](#api-endpoints)
   - [Upload Endpoint](#upload-endpoint)
   - [Custom User Format Endpoint](#custom-user-format-endpoint)
   - [Demo Endpoint](#demo-endpoint)
3. [How to Use](#how-to-use)
   - [Using `curl`](#using-curl)
   - [Using Python](#using-python)
4. [Supported File Types](#supported-file-types)
5. [Error Handling](#error-handling)

## Overview
The Invoice Extractor API processes uploaded invoice files and extracts key data from them. It handles:
- **PDF files**: Converted to images for processing, with each page handled individually.
- **Image files**: Processed directly for data extraction.

The API ensures robust error handling for missing fields, item inconsistencies, tax calculations, and subtotal validations.

## API Endpoints

### Upload Endpoint
```
POST /
```
#### Description
Upload an invoice file (PDF or image) to extract structured data.

#### Request Headers
- **Content-Type**: `multipart/form-data`

---

### Custom User Format Endpoint
```
POST /user-format/
```
#### Description
Upload an invoice file with a custom JSON format to receive extracted data in the desired structure.

#### Request Headers
- **Content-Type**: `multipart/form-data`

---

### Demo Endpoint
```
GET /demo
```
#### Description
Serves a demo page for testing the API.

## How to Use

### Using `curl`
#### Upload Endpoint:
```bash
curl -X POST -F "file=@invoice.pdf" https://invoice-extractor-api.onrender.com/
```

#### Custom User Format Endpoint:
```bash
curl -X POST \
    -F "file=@invoice.pdf" \
    -F "user_format={\"merchant\":{\"name\":\"Store Name\"}}" \
    https://invoice-extractor-api.onrender.com/user-format/
```

### Using Python
#### Upload Endpoint:
```python
import requests

url = "https://invoice-extractor-api.onrender.com/"
with open('path/to/your/file.pdf', 'rb') as file:
    files = {'file': ('file.pdf', file)}
    response = requests.post(url, files=files)
    print(response.json())
```

#### Custom User Format Endpoint:
```python
import requests

url = "https://invoice-extractor-api.onrender.com/user-format/"
files = {'file': ('file.pdf', open('path/to/your/file.pdf', 'rb'))}
data = {'user_format': '{"merchant":{"name":"Store Name"}}'}
response = requests.post(url, files=files, data=data)
print(response.json())
```

## Supported File Types
- **PDF**: `application/pdf`
- **JPEG**: `image/jpeg`
- **JPG**: `image/jpg`
- **PNG**: `image/png`

Other file types will result in an error.

## Error Handling
- **Unsupported File Type**:
  ```json
  {
    "detail": "Unsupported file type. Only PDF, JPEG, and PNG are allowed."
  }
  ```
- **Empty or Invalid PDF**:
  ```json
  {
    "detail": "The uploaded PDF is empty or invalid."
  }
  ```
- **Internal Server Errors**:
  ```json
  {
    "detail": "Error processing invoice: <details>"
  }
  ```
  
## Note
**For missing fields in the invoice:**
- String fields will be assigned the value null.
- Numeric fields will be assigned the value 0.0.

## Conclusion
The Invoice Extractor API is a simple and effective tool to automate the extraction of data from invoices. Whether using `curl` or Python, the API is designed for ease of integration into your workflows. For advanced use cases, leverage the custom user format endpoint.
