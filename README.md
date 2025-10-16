FundusNext — Real-Time Glaucoma Detection with ConvNeXtTiny

Tech Stack: Python · PyTorch · Streamlit · Torchvision · PIL · NumPy

📘 Overview

FundusNext is a deep learning-powered web application designed for automated glaucoma detection from retinal fundus images.
It uses a ConvNeXtTiny model trained on the EYEPA CS dataset, achieving 89.61% accuracy and AUC of 0.9692.

The model is optimized and deployed using Streamlit, allowing real-time, interactive predictions directly in the browser.

🚀 Features

✅ Pre-trained ConvNeXtTiny model fine-tuned for glaucoma classification
✅ Live prediction through a Streamlit-based web interface
✅ Automatic image preprocessing (resizing, normalization)
✅ Confidence-based output with a clear Glaucoma/No Glaucoma result
✅ Fast CPU inference — no GPU required

🧠 Model Summary

The project uses ConvNeXtTiny as the backbone CNN with a custom classifier head for binary classification:

model.classifier = nn.Sequential(
    nn.Flatten(),
    nn.LayerNorm(768, eps=1e-06, elementwise_affine=True),
    nn.Linear(768, 64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 1),
    nn.Sigmoid()
)


The trained weights are hosted on Hugging Face, loaded dynamically during runtime:

state_dict = torch.hub.load_state_dict_from_url(
    "https://huggingface.co/dhundhun1111/FundusNext/resolve/main/ConvNeXtTiny_best.pth",
    map_location=torch.device("cpu")
)

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/<your-username>/FundusNext.git
cd FundusNext

2️⃣ Install dependencies
pip install -r requirements.txt


If you don’t have a requirements.txt, you can manually install the following:

pip install streamlit torch torchvision pillow numpy

3️⃣ Run the app
streamlit run app.py

4️⃣ Upload a fundus image

Once the app launches in your browser:

Click “Choose an image…”

Upload your retinal fundus image (.jpg/.png/.jpeg)

View the prediction and confidence instantly.

🧩 How It Works

The uploaded image is preprocessed using:

Resizing → 256×256

Normalization using ImageNet mean & std

The image is passed through the ConvNeXtTiny model.

The model outputs a confidence score (0–100%).

A confidence > 50% indicates Glaucoma Detected, otherwise No Glaucoma.

🧮 Example Output

Input:
A fundus image uploaded by the user

Output:

Prediction: Glaucoma Detected
Confidence: 91.24%

📁 Project Structure
FundusNext/
│
├── app.py                 # Streamlit app (main script)
├── requirements.txt        # Project dependencies
└── README.md               # Documentation

🧑‍💻 Author

Arpit Sharma
📧 arpitsharma2511@gmail.com

https://www.linkedin.com/in/arpitsharma71/
 

🧭 Future Enhancements

Add Grad-CAM visualizations for explainability

Integrate support for multi-disease classification (e.g., DR, AMD)

Extend deployment via Hugging Face Spaces or Streamlit Cloud

📝 License

This project is licensed under the MIT License — see the LICENSE file for details.
