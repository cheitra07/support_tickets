# support_tickets

Certainly! Below is the updated code to integrate your **Support Ticket Classification** model with **Streamlit** as a web application. The app will allow users to input a new support ticket, and it will classify it based on the closest matches using FAISS and Sentence-Transformer.

### Steps:
1. **Install Streamlit**:
   If you don't have Streamlit installed, you can install it via pip:
   ```bash
   pip install streamlit
   ```

2. **Streamlit Web Application**:
   Below is the complete Streamlit code that sets up a web interface for support ticket classification.

### Streamlit Code (Save as `ticket_classifier_app.py`):
```python
import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Step 1: Load the support ticket data (using a CSV file)
data = pd.read_csv('support_tickets.csv')  # Your file path here

# Step 2: Initialize the SentenceTransformer model (local model, no API key required)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # A small, fast, and accurate model

# Step 3: Generate embeddings for all the ticket descriptions
ticket_descriptions = data['ticket_description'].tolist()  # assuming column is named 'ticket_description'
embeddings = np.array([embed_model.encode(ticket) for ticket in ticket_descriptions])

# Step 4: Create FAISS index for similarity search
embedding_dim = embeddings.shape[1]  # Dimension of embeddings (should match SentenceTransformer output)
faiss_index = faiss.IndexFlatL2(embedding_dim)  # FAISS index using L2 distance

# Add embeddings to the FAISS index
faiss_index.add(embeddings)

# Step 5: Function to classify a new support ticket
def classify_ticket(new_ticket):
    # Generate embedding for the new support ticket description
    new_ticket_embedding = embed_model.encode(new_ticket).reshape(1, -1)

    # Perform similarity search to find the closest match
    k = 3  # Number of closest matches to return
    distances, indices = faiss_index.search(new_ticket_embedding, k)

    # Output the top k closest tickets and their categories
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "ticket_description": data.iloc[idx]['ticket_description'],
            "category": data.iloc[idx]['category'],
            "distance": distances[0][i]
        })
    return results

# Step 6: Streamlit interface
st.title("Support Ticket Classifier")
st.write("Enter a description of the new support ticket below:")

# Step 7: Input field for the new support ticket
new_ticket = st.text_area("New Ticket Description")

if st.button('Classify'):
    if new_ticket:
        results = classify_ticket(new_ticket)
        st.write(f"Top 3 closest tickets for: '{new_ticket}'")
        
        # Display the results
        for i, result in enumerate(results):
            st.write(f"{i+1}:")
            st.write(f"Ticket Description: {result['ticket_description']}")
            st.write(f"Category: {result['category']}")
            st.write(f"Distance: {result['distance']:.4f}")
    else:
        st.warning("Please enter a ticket description.")

```

### Instructions to Run the Streamlit Application:
1. **Prepare your support ticket data**:
   - Make sure you have a CSV file called `support_tickets.csv` with at least two columns:
     - `ticket_description` (descriptions of the support tickets).
     - `category` (the predefined categories of the tickets).
   
   Example CSV (`support_tickets.csv`):
   ```csv
   ticket_description,category
   "Printer not working", "Hardware Issue"
   "Software crash when opening file", "Software Issue"
   "Account login failed", "Account Issue"
   "System is very slow", "Performance Issue"
   ```

2. **Run the Streamlit application**:
   - Save the Streamlit code above as `ticket_classifier_app.py`.
   - Open a terminal and navigate to the directory where `ticket_classifier_app.py` is saved.
   - Run the application by typing:
     ```bash
     streamlit run ticket_classifier_app.py
     ```

3. **Interact with the Streamlit app**:
   - The Streamlit app will open in your browser. You'll see a text input area where you can enter a description of a new support ticket.
   - After clicking the "Classify" button, the app will show the top 3 closest existing tickets and their categories along with the similarity distances.

### Key Components:
- **SentenceTransformer**: Used to generate embeddings for the support ticket descriptions. It's a local model, so no API key is needed.
- **FAISS**: Used to index the ticket descriptions and perform efficient similarity search.
- **Streamlit Interface**: Provides a user-friendly interface where users can input a new support ticket description and get classified based on the closest matches.

### Expected Output:
After entering a description of a new support ticket, the app will show something like:
```
Top 3 closest tickets for: 'Printer is showing an error message'
1:
Ticket Description: Printer not working
Category: Hardware Issue
Distance: 0.5423

2:
Ticket Description: System is very slow
Category: Performance Issue
Distance: 0.6789

3:
Ticket Description: Software crash when opening file
Category: Software Issue
Distance: 1.1345
```

This is a cost-effective solution as everything is done locally without requiring any external API keys. Streamlit makes it easy to deploy and interact with the model via a web interface.

Let me know if you need further adjustments or improvements!
