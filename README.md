# 🤖 Data Mind

An intelligent data analysis assistant powered by **Google Gemini**, **LangChain**, and **Streamlit**. This agent helps you analyze tabular data (CSV, SQL databases), generate interactive visualizations, and uncover insights using natural language.

## ✨ Key Features

*   **🗣️ Chat with Your Data**: Ask questions in plain English (e.g., "What is the sales trend for 2024?", "Compare profit by region").
*   **📊 Dynamic Visualizations**: Automatically generates interactive **Plotly** charts (Bar, Line, Pie, Scatter, Maps, etc.).
*   **🚀 Dashboard Generation**:
    *   **Guided Mode**: Step-by-step creation of custom dashboards based on your requirements.
    *   **Creative Mode**: Automatically generates a full "Power BI-style" dashboard with inferred KPIs and charts.
*   **📂 Multi-Source Support**:
    *   Upload **CSV** files directly.
    *   Connect to **SQLite** databases (default `snapshot.db` support).
*   **🧠 Diagnostic Analysis (RAG)**: Uses **Retrieval Augmented Generation (RAG)** to answer "Why" questions (e.g., "Why did sales drop in Q3?") by cross-referencing data with external documents (PDFs, reports).
*   **🔒 Safe Execution**: Runs LLM-generated Python code in a secure, sandboxed environment.

## 🛠️ Tech Stack

*   **Frontend**: Streamlit
*   **LLM**: Google Gemini (via `langchain-google-genai`)
*   **Orchestration**: LangChain, LangGraph
*   **Data Processing**: Pandas, NumPy
*   **Visualization**: Plotly Express, Plotly Graph Objects
*   **Vector Store**: FAISS (for RAG)
*   **Database**: SQLite

## 🚀 Getting Started

### Prerequisites

*   Python 3.10 or higher
*   A Google Cloud API Key (for Gemini)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/Analyst_AI_Agent.git
    cd Analyst_AI_Agent
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**:
    Create a `.env` file in the root directory and add your Google API Key:
    ```env
    GOOGLE_API_KEY=your_google_api_key_here
    ```

## 🏃‍♂️ Usage

Run the Streamlit application:

```bash
streamlit run ai_data_analyst/src/app.py
```

### How to Use:
1.  **Upload Data**: Drag and drop a CSV file, or let the app load the default `snapshot.db`.
2.  **Ask Questions**: Type queries like "Show me the top 5 products by revenue" or "Plot a heatmap of sales by day".
3.  **Create Dashboards**: Ask "Create a dashboard" and choose between **Guided** or **Creative** mode.
4.  **Diagnostic Analysis**: If you have indexed documents, ask "Why" questions to get context-aware answers.

## 📂 Project Structure

```
Analyst_AI_Agent/
├── ai_data_analyst/
│   ├── src/
│   │   ├── app.py              # Main Streamlit application
│   │   ├── code_executor.py    # Safe Python code execution logic
│   │   ├── data_analysis.py    # Data loading and summary generation
│   │   ├── llm_utils.py        # LLM interaction and prompt handling
│   │   ├── rag_agent.py        # RAG pipeline (Indexing & Retrieval)
│   │   └── db_connector.py     # Database connection utilities
│   ├── prompts/
│   │   └── system_prompt.txt   # Core instructions for the AI Agent
│   └── rag_output/             # Stored vector indices (FAISS)
├── requirements.txt            # Project dependencies
├── .env                        # Environment variables (API Keys)
└── README.md                   # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
