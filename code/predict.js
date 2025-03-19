function displayFile(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            document.getElementById("selectedImage").src = e.target.result; // Show selected image
            document.getElementById("submitButton").disabled = false; // Enable submit button
        };
        reader.readAsDataURL(file);
    }
}

function submitImage() {
    const fileInput = document.getElementById("imageInput");
    const resultDisplay = document.getElementById("result");

    if (!fileInput.files.length) {
        alert("Please select an image file.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
    })
        .then(response => response.json())
        .then(data => {
            console.log("Prediction result:", data);
            resultDisplay.innerHTML = "Prediction: " + (data.result || "Unknown");
        })
        .catch(error => {
            console.error("Error:", error);
            alert("Failed to fetch the prediction. Please try again.");
        });
}
