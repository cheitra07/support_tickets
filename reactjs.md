To create a React.js UI for your support ticket classification application and run it, I'll guide you through the steps to integrate your backend (the machine learning model) with a React.js frontend.

Here’s how to proceed:

### Steps Overview:
1. **Set Up a Backend API** (to handle the model inference and search logic).
2. **Create a React.js Frontend** (to allow users to interact with the backend via a UI).
3. **Connect Backend and Frontend**.
4. **Run the Application**.

### 1. Backend API in Python (Using Flask)

To interact with your machine learning model, you'll first need to create a backend API in Python using Flask. This API will handle user input (support tickets) and return the classification result.

```python
# app.py (Flask backend)
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the sentence transformer model and FAISS index
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_excel('support_tickets.xlsx')  # Make sure you have an Excel file with 'ticket' and 'method' columns
documents = [row['ticket'] for _, row in df.iterrows()]
embeddings = np.array([embed_model.encode(doc) for doc in documents])
embedding_dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(embeddings)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the query from the user
    data = request.json
    query = data.get('ticket', '')
    
    # Generate query embedding and search the FAISS index
    query_embedding = embed_model.encode(query).reshape(1, -1)
    k = 3
    distances, indices = faiss_index.search(query_embedding, k)
    
    # Get the matching methods
    result = []
    for i, idx in enumerate(indices[0]):
        result.append({
            'method': df.iloc[idx]['method'],
            'distance': distances[0][i]
        })
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

### 2. React.js Frontend (UI for Input)

In the React app, you'll create a simple UI with an input box where users can type their support tickets, and it will display the matching methods.

To start, you need to create a React app:

```bash
npx create-react-app support-ticket-frontend
cd support-ticket-frontend
npm install axios
```

Create a component (`App.js`) for the UI:

```jsx
// src/App.js
import React, { useState } from 'react';
import axios from 'axios';

const App = () => {
  const [ticket, setTicket] = useState('');
  const [result, setResult] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!ticket) {
      alert('Please enter a support ticket!');
      return;
    }
    
    setLoading(true);
    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', { ticket });
      setResult(response.data);
    } catch (error) {
      console.error('Error fetching the result', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Support Ticket Classification</h1>
      <textarea 
        value={ticket}
        onChange={(e) => setTicket(e.target.value)}
        placeholder="Enter your support ticket"
        rows="5"
        cols="50"
      />
      <br />
      <button onClick={handleSubmit} disabled={loading}>
        {loading ? 'Processing...' : 'Get Methods'}
      </button>

      <div>
        {result.length > 0 && (
          <div>
            <h2>Suggested Methods:</h2>
            <ul>
              {result.map((item, index) => (
                <li key={index}>
                  Method: {item.method}, Distance: {item.distance.toFixed(4)}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
```

### 3. Running the Backend (Flask API)

Make sure you have the backend running with the Flask API that interacts with the machine learning model. To run the backend API (`app.py`), open a terminal and navigate to the directory where your `app.py` is located. Run the following:

```bash
python app.py
```

This will start the Flask server at `http://127.0.0.1:5000`. The API is now ready to handle requests from the React frontend.

### 4. Running the Frontend (React.js)

Now that the backend is running, you can run the frontend React app. Open a terminal, navigate to the `support-ticket-frontend` directory, and run:

```bash
npm start
```

This will start the React app on `http://localhost:3000`.

### 5. Integrating Flask and React

Now that the backend and frontend are set up, the React app sends a POST request to the Flask API when the user submits the ticket. The Flask API processes the input, searches the FAISS index, and returns the relevant results, which are then displayed in the React frontend.

### 6. Deploying the Application

For development:
- **Backend**: The Flask backend can run locally on `localhost:5000`.
- **Frontend**: The React frontend will run on `localhost:3000`.

For production deployment:
- Use a service like **Heroku**, **AWS**, or **Docker** to deploy both the backend and frontend.

### How to Run in Spyder

If you want to run this in **Spyder**, you can run the Flask API in one terminal and the React app in another terminal:

1. **Backend (Flask)**: Run `python app.py` in one terminal.
2. **Frontend (React)**: In another terminal, navigate to the `support-ticket-frontend` directory and run `npm start`.

This allows you to keep the backend and frontend running locally in parallel.

---

Let me know if you need further assistance with the setup or deployment!
