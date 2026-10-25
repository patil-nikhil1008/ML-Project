/**
 * Frontend Interactive Controller for CIFAR-10 Pure NumPy Classifier
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const dropzoneContent = document.getElementById('dropzoneContent');
  const previewContainer = document.getElementById('previewContainer');
  const imagePreview = document.getElementById('imagePreview');
  const clearBtn = document.getElementById('clearBtn');
  const classifyBtn = document.getElementById('classifyBtn');
  const btnText = document.getElementById('btnText');
  const btnSpinner = document.getElementById('btnSpinner');

  // Result Elements
  const resultsPlaceholder = document.getElementById('resultsPlaceholder');
  const resultsContent = document.getElementById('resultsContent');
  const predictionBanner = document.getElementById('predictionBanner');
  const predictedClass = document.getElementById('predictedClass');
  const confidenceValue = document.getElementById('confidenceValue');
  const latencyValue = document.getElementById('latencyValue');

  // OOD Alert Elements
  const oodWarningBanner = document.getElementById('oodWarningBanner');
  const oodWarningTitle = document.getElementById('oodWarningTitle');
  const oodWarningText = document.getElementById('oodWarningText');

  // Probability Bars
  const probAirplane = document.getElementById('probAirplane');
  const barAirplane = document.getElementById('barAirplane');
  const probAutomobile = document.getElementById('probAutomobile');
  const barAutomobile = document.getElementById('barAutomobile');
  const probBird = document.getElementById('probBird');
  const barBird = document.getElementById('barBird');

  // Tabs
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  // Sample Buttons
  const sampleBtns = document.querySelectorAll('.sample-btn');

  let currentFile = null;

  // 1. File Upload / Dropzone Handlers
  dropzone.addEventListener('click', (e) => {
    if (e.target !== clearBtn) {
      fileInput.click();
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files[0]) {
      handleSelectedFile(dt.files[0]);
    }
  });

  clearBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
  });

  function handleSelectedFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file (PNG, JPG, WebP).');
      return;
    }
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      dropzoneContent.style.display = 'none';
      previewContainer.style.display = 'flex';
      classifyBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function resetUpload() {
    currentFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    previewContainer.style.display = 'none';
    dropzoneContent.style.display = 'block';
    classifyBtn.disabled = true;
    resultsContent.style.display = 'none';
    resultsPlaceholder.style.display = 'flex';
    oodWarningBanner.style.display = 'none';
  }

  // 2. Sample Selector
  sampleBtns.forEach(btn => {
    btn.addEventListener('click', async () => {
      const className = btn.getAttribute('data-class');
      try {
        btn.disabled = true;
        btnText.textContent = `Loading ${className}...`;
        const res = await fetch(`/api/samples/${className.toLowerCase()}`);
        if (!res.ok) throw new Error('Failed to load sample image');
        const blob = await res.blob();
        const file = new File([blob], `sample_${className.toLowerCase()}.png`, { type: 'image/png' });
        handleSelectedFile(file);
        // Automatically trigger classification
        await runClassification(file);
      } catch (err) {
        console.error(err);
        alert(`Error loading sample: ${err.message}`);
      } finally {
        btn.disabled = false;
        btnText.textContent = 'Classify Image';
      }
    });
  });

  // 3. Classification API Call
  classifyBtn.addEventListener('click', async () => {
    if (!currentFile) return;
    await runClassification(currentFile);
  });

  async function runClassification(file) {
    classifyBtn.disabled = true;
    btnSpinner.style.display = 'inline-block';
    btnText.textContent = 'Running Inference...';

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Inference request failed');
      }

      const data = await response.json();
      displayResults(data);
    } catch (err) {
      console.error(err);
      alert(`Classification error: ${err.message}`);
    } finally {
      classifyBtn.disabled = false;
      btnSpinner.style.display = 'none';
      btnText.textContent = 'Classify Image';
    }
  }

  function displayResults(data) {
    resultsPlaceholder.style.display = 'none';
    resultsContent.style.display = 'block';

    latencyValue.textContent = `${data.latency_ms} ms`;

    // Check Out-of-Distribution Warning
    if (data.is_out_of_distribution) {
      oodWarningBanner.style.display = 'flex';
      oodWarningTitle.textContent = '⚠️ Unsupported Subject Detected';
      oodWarningText.textContent = data.warning_message || 'This image does not appear to be an Airplane, Automobile, or Bird.';
      predictionBanner.classList.add('is-ood');
      predictedClass.textContent = 'Unsupported Class (OOD)';
      confidenceValue.textContent = '0.00%';
    } else {
      oodWarningBanner.style.display = 'none';
      predictionBanner.classList.remove('is-ood');
      predictedClass.textContent = data.predicted_class;
      confidenceValue.textContent = data.confidence_percentage;
    }

    // Probability Bars
    const probs = data.probabilities || {};
    const airVal = ((probs['Airplane'] || 0) * 100).toFixed(1);
    const autoVal = ((probs['Automobile'] || 0) * 100).toFixed(1);
    const birdVal = ((probs['Bird'] || 0) * 100).toFixed(1);

    probAirplane.textContent = `${airVal}%`;
    barAirplane.style.width = `${airVal}%`;

    probAutomobile.textContent = `${autoVal}%`;
    barAutomobile.style.width = `${autoVal}%`;

    probBird.textContent = `${birdVal}%`;
    barBird.style.width = `${birdVal}%`;
  }

  // 4. Tab Navigation
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetTab = document.getElementById(`tab-${btn.getAttribute('data-tab')}`);
      if (targetTab) {
        targetTab.classList.add('active');
      }
    });
  });
});
