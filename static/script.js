const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const uploadPrompt = document.getElementById('uploadPrompt');
const previewImg = document.getElementById('previewImg');
const resultCard = document.getElementById('resultCard');
const errorCard = document.getElementById('errorCard');
const errorText = document.getElementById('errorText');
const resetBtn = document.getElementById('resetBtn');
const pixelGrid = document.getElementById('pixelGrid');
const verdictDigit = document.getElementById('verdictDigit');
const confidenceFill = document.getElementById('confidenceFill');
const confidenceValue = document.getElementById('confidenceValue');
const top3 = document.getElementById('top3');

dropzone.addEventListener('click', () => {
  if (!dropzone.classList.contains('has-image')) fileInput.click();
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

['dragover', 'dragenter'].forEach(evt =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  })
);
['dragleave', 'drop'].forEach(evt =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
  })
);
dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

resetBtn.addEventListener('click', () => {
  fileInput.value = '';
  previewImg.hidden = true;
  uploadPrompt.hidden = false;
  dropzone.classList.remove('has-image');
  resultCard.hidden = true;
  errorCard.hidden = true;
  resetBtn.hidden = true;
});

function handleFile(file) {
  errorCard.hidden = true;
  resultCard.hidden = true;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.hidden = false;
    uploadPrompt.hidden = true;
    dropzone.classList.add('has-image');
  };
  reader.readAsDataURL(file);

  const formData = new FormData();
  formData.append('image', file);

  fetch('/predict', { method: 'POST', body: formData })
    .then(res => res.json().then(data => ({ ok: res.ok, data })))
    .then(({ ok, data }) => {
      if (!ok) {
        showError(data.error || 'Something went wrong.');
        return;
      }
      showResult(data);
    })
    .catch(() => showError('Could not reach the server. Is it running?'));
}

function showError(message) {
  errorText.textContent = message;
  errorCard.hidden = false;
  resultCard.hidden = true;
  resetBtn.hidden = false;
}

function showResult(data) {
  errorCard.hidden = true;
  resultCard.hidden = false;
  resetBtn.hidden = false;

  // render the 8x8 grid the model actually saw
  pixelGrid.innerHTML = '';
  const flat = data.grid.flat();
  flat.forEach((val, i) => {
    const cell = document.createElement('div');
    cell.className = 'pixel-cell';
    cell.style.setProperty('--cell-opacity', Math.min(val / 16, 1));
    cell.style.animationDelay = `${i * 6}ms`;
    pixelGrid.appendChild(cell);
  });

  verdictDigit.textContent = data.prediction;

  if (data.confidence !== undefined) {
    confidenceFill.style.width = data.confidence + '%';
    confidenceValue.textContent = data.confidence + '%';
  }

  top3.innerHTML = '';
  if (data.top3) {
    data.top3.forEach((item, idx) => {
      const chip = document.createElement('span');
      chip.className = 'chip' + (idx === 0 ? ' top' : '');
      chip.textContent = `${item.digit} · ${item.confidence}%`;
      top3.appendChild(chip);
    });
  }
}
