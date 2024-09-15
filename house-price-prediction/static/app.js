document.getElementById('predictionForm').onsubmit = async function(event) {
    event.preventDefault();
    
    const feature1 = document.getElementById('feature1').value;
    const feature2 = document.getElementById('feature2').value;
    // Gather other features as necessary
    
    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            feature1: feature1,
            feature2: feature2
            // Add other features here
        })
    });
    
    const result = await response.json();
    document.getElementById('result').textContent = `Predicted House Price: ${result.prediction}`;
};
