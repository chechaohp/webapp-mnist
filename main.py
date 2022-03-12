from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps
from utils import predict
import torch
from model import CNNClassifier
import io
import base64

app = Flask(__name__)

# load model
model = CNNClassifier(1,32,10,3)
model.load_state_dict(torch.load("savemodel.pth"))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(dev)
model.eval()

@app.route("/",methods=["POST","GET"])
def intial():
    prediction="?"
    return render_template("index.html",
                            prediction=prediction,
                            status="Ready")

@app.route("/predict_digit",methods=["POST"])
def predict_digit():
    # get image with transparent background
    # COMMENT THIS FOR DEV
    data = request.form['drawing_data']
    user_drawn_image = data.split(',')[1]
    buf = io.BytesIO(base64.b64decode(user_drawn_image))
    img = Image.open(buf).convert("RGBA")
    # create white background image, paste the transparent into white image
    white_img = Image.new("RGBA", img.size, "WHITE")
    white_img.paste(img, mask=img)
    img = white_img
    img = img.convert("RGB")
    img = img.resize([28,28])
    img = ImageOps.invert(img)
    img = ImageOps.grayscale(img)
    # return jsonify({'prediction':data})
    img.save('test.png')
    predicted, prob = predict(model,img,dev)
    # END DEV
    # predicted = [0,1,2,3,4]
    # prob = [0.90,0.80,0.70,0.60,0.50]
    predicted = predicted.tolist()
    prob = prob.tolist()
    prob = [round(x,4)*100 for x in prob]
    return jsonify({'prediction':predicted,
                    'prob':prob})
    
@app.route("/retrain",methods=["POST","GET"])
def retrain():
    pass

if __name__ == "__main__":
    app.run(debug=True)