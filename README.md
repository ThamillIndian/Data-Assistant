# Retail Insights Assistant

A Streamlit dashboard for exploring retail sales data with automated KPIs, interactive charts, Gemini-powered executive summaries, and natural-language data questions.

## Features

- Upload and validate retail CSV files
- Normalize common sales-column names
- Calculate sales, transaction, daily-average, top-family, and top-store KPIs
- Display sales trends and performance charts with Plotly
- Generate an AI executive summary with Google Gemini
- Ask natural-language questions about the uploaded DataFrame
- Maintain chat history during the Streamlit session

## Project Structure

```text
.
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
├── plan.md                   # Project plan and implementation notes
└── utils/
    ├── data_loader.py        # CSV loading, normalization, and validation
    ├── summarizer.py         # KPIs, charts, and executive summary
    └── rag_agent.py          # Gemini/LangChain DataFrame assistant
```

## Requirements

- Python 3.11 or newer
- A Google Gemini API key

## Installation

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

The `.env` file is ignored by Git. The API key can also be entered in the application's sidebar.

## Run the Application

```powershell
streamlit run app.py
```

Then open the local URL shown by Streamlit, upload a CSV file, and select **Make Summary**.

## Expected CSV Schema

The application expects these columns:

| Column | Description |
| --- | --- |
| `date` | Sales record date |
| `store` | Store or sales-channel identifier |
| `family` | Product family or category |
| `unit_sales` | Number of units sold |
| `transactions` | Transaction-related numeric value |

The loader also recognizes these common source names:

| Source column | Normalized column |
| --- | --- |
| `Date` | `date` |
| `Qty` | `unit_sales` |
| `Amount` | `transactions` |
| `Category` | `family` |
| `Sales Channel` | `store` |

For the included Amazon report, review the column meaning before analysis. In that dataset, `Amount` represents monetary order value, while `Qty` represents units sold.

## Application Flow

1. Enter a Gemini API key or provide it through `.env`.
2. Upload a CSV file.
3. The application validates the required columns and data types.
4. Select **Make Summary** to calculate KPIs, charts, and the AI summary.
5. Ask questions about the uploaded data in the chat panel.

## Security Notes

- Never commit `.env` or API keys.
- The DataFrame agent uses generated Python operations to analyze data. Do not expose this application to untrusted users without adding appropriate sandboxing and access controls.
- The bundled dataset is ignored by Git through `.gitignore`.

## Validation

Compile the Python modules with:

```powershell
.\venv\Scripts\python.exe -m compileall -q app.py utils
```
