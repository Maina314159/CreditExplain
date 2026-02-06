# 📘 CreditExplain RAG

**Evidence-backed, auditable explanations for credit decisions — with citations, PII-safety, and regulator-ready audit trails.**

CreditExplain RAG helps **compliance officers and analysts** quickly answer:

* *"Why was this loan declined?"*
* *"Which clause justifies KYC step X?"*

The system integrates advanced **RAG techniques** (SELF-RAG critic loop, reranking, provenance logging, PII redaction) to provide trustworthy, auditable explanations.

Try the system here: https://credit-explain.vercel.app

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
  - [Pre-requisites](#pre-requisites)
  - [Setup Instructions](#setup-instructions)
- [Basic Usage](#-basic-usage)
- [Repository Structure](#-repository-structure)
- [Known Issues](#-known-issues)
- [Future Development](#-future-development)
- [Acknowledgement](#-acknowledgement)
- [Contact Information](#-contact-information)
- [License](#-license)

## ✨ Project Overview

CreditExplain RAG addresses the critical need for **transparent, evidence-based explanations** in financial compliance and credit decisioning. Traditional AI systems often provide "black box" responses without verifiable sources, making them unsuitable for regulated environments.

Our solution provides:
- **Regulatory Compliance**: Every explanation cites specific clauses from authoritative documents
- **Audit Trail**: Complete provenance tracking with reflection token scoring
- **PII Protection**: Automatic redaction of sensitive personal information
- **Multi-jurisdictional Support**: Handling of Nigerian, Kenyan, and global financial regulations

## Some Background as to how this project promotes Responsible/Ethical use of AI

### AI in Credit Scoring & Loan Approval

**1. Description and Purpose**
Banks and fintechs use AI models to assess creditworthiness by analyzing transaction history, repayment behavior, mobile data, and alternative data sources. The goal is faster, more accurate lending decisions.

**2. Ethical Risks**
Bias & Financial Exclusion: Marginalized groups may be unfairly denied credit due to biased data.
Lack of Explainability: Applicants often don’t understand why loans are rejected.
Privacy Concerns: Use of alternative data (e.g., phone usage) can be intrusive.

**3. Responsible AI Mitigations**
This is where CreditExplain comes in!

-Fairness testing across demographics
-Explainable credit decisions (reason codes)
-Strict limits on data sources and informed consent.

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **LangChain** - RAG orchestration and tooling
- **ChromaDB** - Vector database for document storage
- **HuggingFace Transformers** - Embeddings and reranking models
- **Groq API** - LLM inference for critic and generator components
- **Pydantic** - Data validation and serialization

### Frontend
- **React 18** - User interface library
- **TypeScript** - Type-safe development
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **TanStack Query** - Server state management
- **Axios** - HTTP client for API communication

### Machine Learning
- **SELF-RAG Architecture** - Adaptive retrieval with reflection tokens
- **sentence-transformers/all-MiniLM-L6-v2** - Embedding model
- **cross-encoder/ms-marco-MiniLM-L-6-v2** - Reranker model
- **Llama 3** - LLMs for generation and critique

<img width="1251" height="576" alt="image" src="https://github.com/user-attachments/assets/7e393d06-cfb0-4189-a135-39943c1a3190" />


## 🚀 Installation & Setup

### Pre-requisites

- **Python 3.9+** installed on your system
- **Node.js 18+** and npm for frontend development
- **Groq API account** and API key for LLM access
- **Git** for version control

### Setup Instructions

#### 1. Clone & Install Backend Dependencies

```bash
git clone https://github.com/Maina314159/CreditExplain
cd CreditExplain

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

#### 2. Configure Environment Variables

```bash
# Copy and update the environment template
cp .env.example .env
```

Edit `.env` with your configuration:
```env
GROQ_API_KEY=your_groq_api_key_here
```

- For frontend, also create an `.env` with the backend server added to it
```bash
# Go to frontend and create .env file
cd frontend
cp .env.example .env

# Then add backend server
VITE_BACKEND_URL=http://localhost:8000
```

#### 3. Setup Frontend

```bash
cd frontend
npm install
```

#### 4. Ingest Sample Documents

```bash
cd ..
python -m ingest.index
```

#### 5. Run the Application

**Start Backend API:**
```bash
python -m api.app
# API server starts at http://localhost:8000
```

**Start Frontend (in new terminal):**
```bash
cd frontend
npm run dev
# Frontend starts at http://localhost:5173
```

**CLI Application**
- You can also choose to interact with the terminal app:
```bash
python -m core.self_rag
```

## 💡 Basic Usage

1. **Access the Web Interface**: Open http://localhost:5173 in your browser

2. **Upload Documents**: Navigate to the Upload page to add regulatory PDFs
   - Supported formats: PDF, text documents
   - Documents are automatically chunked and indexed

3. **Ask Questions**: Use the Query interface to ask compliance questions like:
   - "What are the capital requirements for banks in Nigeria?"
   - "What are the financial regulations in Kenya"
   - "What documents are required for KYC verification?"

4. **Review Results**: Each response includes:
   - Evidence-backed explanations with citations
   - Confidence scores (HIGH, MEDIUM, LOW)
   - Source document references with exact excerpts
   - Suggested follow-up questions

5. **Monitor Performance**: Check the Metrics dashboard for system performance and audit logs

## 📁 Repository Structure

```
.
├── .github/                 # GitHub templates and workflows
├── api/                     # FastAPI backend application
│   ├── app.py               # Main FastAPI application with CORS
│   ├── models.py            # Pydantic models for request/response
│   └── __init__.py
├── core/                    # RAG pipeline core components
│   ├── self_rag.py          # Main SELF-RAG orchestration logic
│   ├── critic.py            # Critic model for retrieval decisions
│   ├── generator.py         # Response generation component
│   ├── retrieval.py         # Vector retrieval functionality
│   ├── reranker.py          # Cross-encoder reranking
│   ├── prompts.py           # LLM prompt templates
│   ├── provenance.py        # Audit logging and provenance tracking
│   └── __init__.py
├── data/                    # Document storage
│   ├── raw/                 # Original PDF documents
│   └── interim/             # Processed data files
├── frontend/                # React TypeScript frontend
│   ├── src/
│   │   ├── api/             # API client and hooks
│   │   ├── components/      # Reusable UI components
│   │   ├── pages/           # Main application pages
│   │   ├── hooks/           # Custom React hooks
│   │   ├── types/           # TypeScript type definitions
│   │   └── utils/           # Utility functions
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
├── ingest/                  # Document ingestion pipeline
│   ├── loader.py            # PDF loading and parsing
│   ├── chunker.py           # Text chunking strategies
│   ├── index.py             # Vector indexing process
│   ├── normalize.py         # Text normalization
│   └── __init__.py
├── eval/                    # Evaluation and metrics
│   ├── metrics.py           # Performance metrics calculation
│   └── __init__.py
├── tests/                   # Test suites
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   ├── demo_data/           # Test data
│   └── scripts/             # Test scripts
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## ⚠️ Known Issues

### Current Limitations
1. **PDF Parsing Accuracy**: Complex PDF layouts with tables and multi-column formats may not parse perfectly
2. **Rate Limiting**: Groq API has rate limits that may affect performance during high usage
3. **Context Length**: Currently limited to ~1000 token chunks due to model constraints
4. **Metadata Extraction**: Some document metadata (section headers, page numbers) may not be fully preserved

### Performance Considerations
- Initial document ingestion can be slow for large PDF collections
- Real-time query processing typically takes 5-15 seconds depending on complexity
- Vector search performance degrades with very large document collections (>10,000 chunks)

### Browser Compatibility
- Best experienced in modern browsers (Chrome, Firefox, Safari, Edge)
- Mobile experience is functional but optimized for desktop use

## 🔮 Future Development

### Planned Features
- [ ] **Real-time Collaboration**: Multi-user support with shared workspaces
- [ ] **Advanced Document Types**: Support for Word documents, HTML, and scanned PDFs
- [ ] **Custom Model Support**: Integration with local LLMs and embedding models
- [ ] **Enhanced Analytics**: Advanced dashboard with trend analysis and compliance reporting
- [ ] **API Extensions**: Webhook support and third-party integrations

### Research Directions
- Improved chunking strategies for legal and regulatory documents
- Multi-hop reasoning across multiple documents
- Automated regulatory change detection and alerting
- Cross-jurisdictional compliance mapping

## 🙏 Acknowledgement

This project was developed for the **NSK.AI RAG Hackathon 2025** and builds upon several open-source technologies and research:

- **SELF-RAG Paper** ([Asai et al., 2023](https://arxiv.org/abs/2310.11511)) for the adaptive retrieval framework
- **LangChain** and **LangSmith** for RAG orchestration tools
- **HuggingFace** for transformer models and embeddings
- **FastAPI** for the high-performance backend framework
- **React Query** for efficient server state management

Special thanks to the regulatory bodies whose documents made this system possible:
- Central Bank of Nigeria (CBN)
- Central Bank of Kenya (CBK)
- Financial Action Task Force (FATF)

## 📞 Contact Information

For questions, issues, or contributions:

- **Project Maintainer**: CreditExplain Team
<!-- - **Email**: [your-email@example.com] -->
- **GitHub Issues**: [Create an issue](https://github.com/Maina314159/CreditExplain/issues)

We welcome bug reports, feature requests, and contributions from the community!

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

<div align="center">


[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18%2B-61dafb)](https://reactjs.org/)

</div>
