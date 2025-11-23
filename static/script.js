// Prediction form handler
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Get form data
    const formData = new FormData(e.target);
    const data = {};
    
    formData.forEach((value, key) => {
        // Convert numeric fields
        if (['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'].includes(key)) {
            data[key] = parseInt(value);
        } else {
            data[key] = value;
        }
    });
    
    // Show loading state
    const submitBtn = e.target.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = '<span>Predicting...</span>';
    submitBtn.disabled = true;
    
    try {
        // Make prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const result = await response.json();
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        alert('Prediction failed: ' + error.message);
    } finally {
        // Restore button
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    }
});

function displayResults(result) {
    // Show results section
    const resultsDiv = document.getElementById('results');
    resultsDiv.classList.remove('hidden');
    
    // Update prediction
    const predictionValue = document.getElementById('predictionValue');
    predictionValue.textContent = result.prediction.toUpperCase();
    predictionValue.style.color = result.prediction === 'yes' ? 'var(--success-green)' : 'var(--danger-red)';
    
    // Update probability
    const probabilityValue = document.getElementById('probabilityValue');
    probabilityValue.textContent = (result.probability * 100).toFixed(2) + '%';
    
    // Update confidence
    const confidenceValue = document.getElementById('confidenceValue');
    confidenceValue.textContent = result.confidence;
    confidenceValue.className = 'confidence-badge ' + result.confidence.toLowerCase();
    
    // Update recommendation
    const recommendationValue = document.getElementById('recommendationValue');
    recommendationValue.textContent = result.recommendation;
    
    // Scroll to results
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Check API health on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/health');
        const health = await response.json();
        
        if (health.status !== 'healthy') {
            console.warn('API health check failed:', health);
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
});
