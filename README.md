# Medibot: A Streamlit-Powered Medical Assistant Chatbot

## Overview

Medibot is a medical assistant chatbot built using Streamlit, LangChain, and Hugging Face models. The chatbot leverages a pre-trained language model to provide accurate answers to medical queries by retrieving relevant information from a local FAISS vector store. The project supports persistent memory, allowing conversations to continue across sessions.

## Features

- **Medical Assistance:** Provides answers to medical queries using a pre-trained language model.
- **Persistent Memory:** Saves chat history locally to continue conversations across sessions.
- **Streamlit Interface:** Interactive user interface built with Streamlit.
- **Custom Prompt Templates:** Uses custom prompt templates to enhance response quality.
- **Local Vector Store:** Uses FAISS vector store for efficient information retrieval.

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Anaconda (recommended for managing environments)
- Hugging Face API token

### Installation

1. **Clone the Repository:**
   ```sh
   git clone <repository_url>
   cd <repository_directory>
2. **Create a Conda Environment:**
   conda create --name medibot python=3.11
   conda activate medibot
3. **Install Required Packages:**
   pip install -r requirements.txt
4. **Configuration**
Create a .env File:

Add your Hugging Face API token to the .env file:
HF_TOKEN=your_hugging_face_api_token

1. **Usage**
Load Data and Build Vector Store:

Run the memory_.py script to load PDF documents, create text chunks, and store embeddings in the FAISS vector store:
python memory_.py

2.**Run the Chatbot:**

Launch the Streamlit app:
streamlit run medibot.py

**Contributing**
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

**License**
This project is licensed under the MIT License.

**Acknowledgements**
Hugging Face for providing the pre-trained models and API.

LangChain for the retrieval and chaining framework.

Streamlit for the interactive user interface framework.


### Explanation
- **Overview:** Briefly describes what the project does.
- **Features:** Lists the main features of the project.
- **Setup Instructions:** Provides step-by-step instructions to set up the project.
- **Configuration:** Details how to configure the environment variables.
- **Usage:** Explains how to run the project.
- **Contributing:** Invites contributions and provides guidance on how to contribute.
- **License:** Mentions the project's license.
- **Acknowledgements:** Credits the tools and libraries used in the project.

Feel free to customize any part of this `README.md` file to better fit your project's needs! If you need further assistance, just let me know.








