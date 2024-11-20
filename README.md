# Chat-with-PDF-using-API-model

A Streamlit-based application that allows users to chat with their PDF documents using Grok AI. The app supports both text-based PDFs and scanned documents through OCR capabilities.

## Description

This application bridges the gap between static PDF documents and interactive AI-powered conversations. Whether you have text-based PDFs or scanned documents, our application processes them and enables natural conversation about their contents using Grok AI's advanced language capabilities.

### Why This Project?

- **Accessibility**: Makes PDF content more interactive and accessible
- **Versatility**: Handles both digital and scanned documents
- **Efficiency**: Quick information retrieval through conversational interface
- **Intelligence**: Leverages Grok AI for context-aware responses

## Features

- PDF, DOCX, TXT text extraction (both native text and OCR support)
- Text chunking and vector storage for efficient retrieval
- Conversational interface powered by Grok AI
- Document context-aware responses
- Chat history management
- Support for multiple PDF uploads
- Code block detection and formatting
- Clean and intuitive user interface

## Prerequisites

Before you begin, ensure you have:
- Python 3.8+
- Grok API key
- Required Python packages (see requirements.txt)
- Sufficient storage space for document processing

## Installation

1. Clone the repository:```bash
git clone https://github.com/AnhQuan2004/Chat-with-PDF-using-API-model
```

2. Navigate to the project directory:
```bash
cd Chat-with-PDF-using-API-model
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export GROK_API_KEY="your_grok_api_key"
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run main.py
```

2. Upload your PDF document through the web interface
3. Start chatting with your document using natural language queries
4. View and manage chat history as needed

## Demo Video


[![Demo Video](https://img.youtube.com/vi/XSuvMuEx5EU/maxresdefault.jpg)](https://www.youtube.com/watch?v=XSuvMuEx5EU)
## Screenshots

Here are some screenshots showing the application in action:

![App Screenshot](image/Screenshot%202024-11-19%20214246.png)
![App Screenshot](image/Screenshot%202024-11-19%20214253.png)
![App Screenshot](image/Screenshot%202024-11-19%20214259.png)
![App Screenshot](image/Screenshot%202024-11-19%20215030.png)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

If you encounter any issues or have questions, please:
1. Check the existing issues or create a new one
2. Reach out through the repository's issue tracker

## Future Roadmap

- Support for additional document formats
- Enhanced OCR capabilities
- Multi-language support
- Improved context handling
- Custom training capabilities
