
# Invoice Extractor

A Streamlit-based application that extracts structured information from invoice images or PDFs. It preprocesses uploaded documents by straightening the images and then leverages Google’s Generative AI API to identify and format key details, such as merchant information, item details, taxes, and total amounts, into an organized JSON structure. The app also provides a Q&A feature for querying extracted information conversationally.

- [Click here](https://invoice-extractor-api.onrender.com) to access API.
- [Read about](/Invoice%20Extractor%20API/README.md) how to use API.

## Features
- **Straightening & Preprocessing**: Automatically corrects image orientation.
- **Invoice Data Extraction**: Extracts key details such as merchant information, item list, tax details, and total amounts.
- **Q&A Interface**: Enables users to ask specific questions based on the extracted data, with responses generated conversationally.

## Demo

For a quick overview of how the application works, check out the [video demo](https://drive.google.com/file/d/18ZAj3EQ1Q5HzYqgdH-pKf3qRudWSS0a3/view?usp=drive_link).

## Installation

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### Step 2: Set up a Conda Environment
Ensure you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed. Then, create and activate a new environment:

```bash
conda create -n invoice_extractor_env python=3.10
conda activate invoice_extractor_env
```

### Step 3: Install Requirements
Install all dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 4: Set Up API Keys
This project uses Google’s Generative AI API. To use it:
1. Get your API key from Google and store it in a `.env` file in the project directory.
    Create your API key from [here](https://aistudio.google.com/app/apikey).
2. Add the following line in the `.env` file:
    ```bash
    GOOGLE_API_KEY=<Your-API-Key>
    ```

## Usage

### Step 1: Run the Application
Start the Streamlit app by running:

```bash
streamlit run invoice_app.py
```

or 

```bash
streamlit run full_invoice_app.py
```

### Step 2: Upload and Process an Invoice
1. Open the app in your browser (Streamlit will display the link in your terminal).
2. Upload an invoice image or PDF.
3. The app will straighten the image, extract details, and display the formatted data.

### Step 3: Query Extracted Data
After extraction, enter specific queries in the Q&A interface to get responses based on the extracted information.

## File Structure
- **invoice_app.py**: Main application code for processing invoices and handling Q&A interactions. Indicates a lightweight, focused application fetching only the required information.
- **full_invoice_app.py**: Main application code for processing invoices. Indicates a more comprehensive application fetching all available details.
- **requirements.txt**: List of dependencies for the project.

## Notes
- The Q&A function only responds to questions relevant to the extracted invoice data.
- Missing values in the invoice are marked as `"null"`.
