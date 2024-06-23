# project_sem4
# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that accepts multiple PDF documents as input and allows users to query these documents. This approach helps the Language Model (LLM) reduce hallucinations and provide more accurate answers.

## Features

- Accepts multiple PDF documents as input.
- Allows querying across multiple documents.
- Uses Retrieval-Augmented Generation to improve answer accuracy.
- Reduces hallucinations from the language model by referencing actual documents.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/stevejoscm/project_sem4.git
    cd project_sem4
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Ensure your virtual environment is activated.
2. Place the PDF documents you want to query in the designated directory.
3. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```


## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.


