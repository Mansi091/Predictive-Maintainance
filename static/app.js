const inputs = {
    air_temp: document.getElementById('air_temp'),
    process_temp: document.getElementById('process_temp'),
    rotational_speed: document.getElementById('rotational_speed'),
    torque: document.getElementById('torque'),
    tool_wear: document.getElementById('tool_wear')
};

const displays = {
    air_temp: document.getElementById('air_temp_val'),
    process_temp: document.getElementById('process_temp_val'),
    rotational_speed: document.getElementById('rotational_speed_val'),
    torque: document.getElementById('torque_val'),
    tool_wear: document.getElementById('tool_wear_val')
};

const riskLevel = document.getElementById('risk-level');
const failureProb = document.getElementById('failure-probability');
const recommendationText = document.getElementById('recommendation-text');
const shapBarsContainer = document.getElementById('shap-bars');

const csvFileInput = document.getElementById('csv-file');
const fileNameDisplay = document.getElementById('file-name-display');
const uploadBtn = document.getElementById('upload-btn');
const batchResultsSummary = document.getElementById('batch-results-summary');
const batchTotal = document.getElementById('batch-total');
const batchFailures = document.getElementById('batch-failures');

Object.keys(inputs).forEach(key => {
    inputs[key].addEventListener('input', (e) => {
        displays[key].textContent = e.target.value;
    });
    inputs[key].addEventListener('change', runPrediction);
});

async function runPrediction() {
    const payload = {
        air_temp: parseFloat(inputs.air_temp.value),
        process_temp: parseFloat(inputs.process_temp.value),
        rotational_speed: parseFloat(inputs.rotational_speed.value),
        torque: parseFloat(inputs.torque.value),
        tool_wear: parseFloat(inputs.tool_wear.value)
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`Server returned status ${response.status}`);
        }

        const data = await response.json();
        updateUI(data);
    } catch (error) {
        console.error('Error fetching prediction:', error);
        recommendationText.textContent = `Diagnostic evaluation failed: ${error.message}`;
    }
}

function updateUI(data) {
    const probPct = (data.failure_probability * 100).toFixed(1);
    failureProb.textContent = `${probPct}%`;
    
    riskLevel.textContent = `${data.risk_level} Risk`;
    riskLevel.className = 'risk-badge';
    if (data.risk_level === 'Medium') {
        riskLevel.classList.add('medium');
    } else if (data.risk_level === 'High') {
        riskLevel.classList.add('high');
    }

    recommendationText.textContent = data.recommendations.join(' ');
    renderShapBars(data.feature_contributions);
}

function renderShapBars(contributions) {
    shapBarsContainer.innerHTML = '';
    
    const values = Object.values(contributions);
    const maxAbs = Math.max(...values.map(Math.abs), 0.0001);

    Object.entries(contributions).forEach(([feature, val]) => {
        const row = document.createElement('div');
        row.className = 'shap-row';

        const label = document.createElement('div');
        label.className = 'shap-label';
        label.textContent = feature;

        const barWrapper = document.createElement('div');
        barWrapper.className = 'shap-bar-wrapper';

        const bar = document.createElement('div');
        const percentage = Math.min(100, (Math.abs(val) / maxAbs) * 100);
        bar.className = `shap-bar ${val >= 0 ? 'positive' : 'negative'}`;
        bar.style.width = `${percentage}%`;
        barWrapper.appendChild(bar);

        const valText = document.createElement('div');
        valText.className = 'shap-value-text';
        valText.textContent = (val >= 0 ? '+' : '') + val.toFixed(3);

        row.appendChild(label);
        row.appendChild(barWrapper);
        row.appendChild(valText);
        shapBarsContainer.appendChild(row);
    });
}

csvFileInput.addEventListener('change', () => {
    if (csvFileInput.files.length > 0) {
        const file = csvFileInput.files[0];
        fileNameDisplay.textContent = file.name;
        uploadBtn.disabled = false;
    } else {
        fileNameDisplay.textContent = 'No file selected';
        uploadBtn.disabled = true;
    }
});

uploadBtn.addEventListener('click', async () => {
    const file = csvFileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
        uploadBtn.textContent = 'Processing...';
        uploadBtn.disabled = true;

        const response = await fetch('/predict-batch', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errDetails = await response.json();
            throw new Error(errDetails.detail || `Server error: ${response.status}`);
        }

        const data = await response.json();
        
        batchTotal.textContent = data.total_records;
        batchFailures.textContent = data.failures_detected;
        batchResultsSummary.style.display = 'block';
    } catch (error) {
        alert(`Batch processing failed: ${error.message}`);
    } finally {
        uploadBtn.textContent = 'Process Batch';
        uploadBtn.disabled = false;
    }
});

window.addEventListener('DOMContentLoaded', runPrediction);
