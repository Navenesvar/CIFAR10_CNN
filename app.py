import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

from model import AdvancedCNN


# CIFAR10 Classes
classes = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load Model
model = AdvancedCNN().to(device)

model.load_state_dict(
    torch.load(
        "advanced_cifar10_cnn.pth",
        map_location=device
    )
)

model.eval()


# Image Transform
transform = transforms.Compose([

    transforms.Resize((32, 32)),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# Streamlit UI
st.title("CIFAR-10 Image Classifier")

st.write("Upload an image to classify")


uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "png", "jpeg"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert('RGB')

    st.image(
        image,
        caption='Uploaded Image',
        width="stretch"
    )

    # Preprocess
    img = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():

        outputs = model(img)

        probabilities = F.softmax(outputs, dim=1)

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    predicted_class = classes[predicted.item()]

    st.success(
        f"Prediction: {predicted_class}"
    )

    st.write(
        f"Confidence: {confidence.item()*100:.2f}%"
    )

    # Show all probabilities
    st.subheader("Class Probabilities")

    for i, cls in enumerate(classes):

        st.write(
            f"{cls}: "
            f"{probabilities[0][i].item()*100:.2f}%"
        )