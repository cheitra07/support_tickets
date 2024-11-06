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

