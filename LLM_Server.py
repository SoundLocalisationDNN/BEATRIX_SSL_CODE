from flask import Flask, request, jsonify
import ollama

client = ollama.Client()

app = Flask(__name__)

model = 'BEATRIX'

#Instantiate the Ollama client.
#Ollama is assumed to be running on localhost:11434.

@app.route('/send', methods=['POST'])
def send_message():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    message = data['message']
    print(message)
    try:
        #Send the message to Ollama using its custom chat method.
        response = client.generate(model= model, prompt=message)
        print("Response from Ollama:")
        print(response.response)
        #Return the response from Ollama.
        return jsonify({"response": response.response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # he server listens on all network interfaces on port 5000.
    app.run(host='0.0.0.0', port=5000)
