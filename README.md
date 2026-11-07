# DataPilot AI

### An Agentic AI-Powered Data Analytics Assistant

DataPilot AI is an AI-powered data analytics assistant that explores how Agentic AI can simplify data interaction through natural-language queries.

> **Project Status:** In progress — deployment and further improvements are ongoing.

## 🚀 Features

- Natural-language interaction for data-related tasks
- Agent workflow orchestration using LangGraph
- LLM integration using Groq
- SQL and data-processing capabilities
- CSV and Excel file upload and analysis
- FastAPI backend
- Streamlit interactive interface
- Docker-based containerization
- GitHub Actions for continuous integration

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Agent Framework | LangGraph, LangChain |
| LLM | Groq |
| Data Processing | Pandas |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Database | SQL Server |
| Containerization | Docker |
| CI/CD | GitHub Actions |

## 🏗️ Architecture

```text
                 User
                  |
                  v
            Streamlit UI
                  |
                  v
          Data Agent Workflow
              LangGraph
                  |
        +---------+---------+
        |         |         |
        v         v         v
    SQL Tasks  Data Analysis  LLM
        |         |         |
        +---------+---------+
                  |
                  v
       SQL Server / Uploaded Files
```

The project also includes a FastAPI backend for API-based interaction.

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API root endpoint |
| POST | `/chat` | Send chat requests to the AI agent |
| POST | `/chat/upload` | Upload CSV or Excel files for processing |

FastAPI provides interactive API documentation through Swagger UI.

## 💻 Getting Started

### Prerequisites

- Python 3.12
- Git
- uv package manager
- Groq API key
- Docker (optional)
- SQL Server configuration (for database functionality)

### 1. Clone the Repository

```bash
git clone https://github.com/PurpleCoder-23/DataPilot-AI.git
cd DataPilot-AI
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Configure Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

Add any required SQL Server connection settings.

**Important:** Never commit API keys, passwords, or database credentials to GitHub.

### 4. Run the FastAPI Backend

```bash
uv run uvicorn api:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### 5. Run the Streamlit Application

```bash
uv run streamlit run app.py
```

### 6. Run Tests

```bash
uv run pytest
```

## 🐳 Docker

Build the Docker image:

```bash
docker build -t datapilot-ai .
```

Run the container:

```bash
docker run --env-file .env -p 8000:8000 datapilot-ai
```

Check your Dockerfile and application entry point for configuration-specific adjustments.

## 🔄 Continuous Integration

GitHub Actions is configured to run automated checks when changes are pushed or pull requests are created for the `main` branch.

The workflow installs dependencies and executes tests using `pytest`.

## 📚 Key Learning Outcomes

- Developing Agentic AI workflows
- Integrating LLMs into data applications
- Building APIs with FastAPI
- Developing interactive applications with Streamlit
- Working with SQL Server and datasets
- Containerizing applications using Docker
- Implementing CI with GitHub Actions
- Debugging and resolving deployment issues

## 🗺️ Roadmap

- [ ] Complete and validate cloud deployment
- [ ] Improve agent routing and response reliability
- [ ] Expand data analysis capabilities
- [ ] Improve error handling and user experience
- [ ] Add more test cases
- [ ] Refine deployment and CI/CD workflow

## 🔐 Security

- Keep API keys and credentials out of source control.
- Use environment variables and platform secrets.
- Avoid uploading sensitive or confidential datasets.
- Validate uploaded files and database operations.

## 👩‍💻 Author

**Shiv Nandini**

M.Tech — Artificial Intelligence and Data Science

Interested in Generative AI, Agentic AI, Data Science, and AI Engineering.

## 🔗 Connect

- GitHub: [PurpleCoder-23](https://github.com/PurpleCoder-23)
- LinkedIn: [Shiv Nandini](https://www.linkedin.com/in/shiv-nandini-300058189/)

## 🏷️ Tags

`Agentic AI` `Generative AI` `LangGraph` `LangChain` `Groq` `Data Analytics` `Python` `FastAPI` `Streamlit` `Docker`
