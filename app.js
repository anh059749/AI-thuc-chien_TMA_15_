async function readApiResponse(response) {
  let data;
  try {
    data = JSON.parse(await response.text());
  } catch {
    const message = [504, 524].includes(response.status)
      ? 'The connection timed out while the AI was working. Keep the server PC running and try again.'
      : 'The website could not reach the essay server. Check that the Python server and your sharing link are running, then refresh this page.';
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  if (!response.ok || !data || typeof data !== 'object') {
    const error = new Error(data?.error || 'The essay server could not complete this request. Please try again.');
    error.status = response.status;
    throw error;
  }
  return data;
}

const essayService = {
  async improveEssay(files, onProgress = () => {}) {
    const formData = new FormData();
    files.forEach((file) => formData.append('images', file));
    const response = await fetch('/api/improve', { method: 'POST', body: formData });
    const data = await readApiResponse(response);
    // Accept an older server's direct result while the server is being restarted.
    if (!data.jobId && data.category) return data;
    if (!data.jobId) throw new Error('Please restart the Python server and refresh this page.');

    onProgress({ title: 'Your pages are uploaded', message: 'The AI will read your pages in order. Keep this page open.' });
    const deadline = Date.now() + 30 * 60 * 1000;
    let connectionFailures = 0;
    while (Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 1500));
      let job;
      try {
        const statusResponse = await fetch(`/api/jobs/${encodeURIComponent(data.jobId)}`, { cache: 'no-store' });
        job = await readApiResponse(statusResponse);
        connectionFailures = 0;
      } catch (requestError) {
        if (requestError.status === 404 || ++connectionFailures >= 5) throw requestError;
        onProgress({ title: 'Reconnecting', message: 'Checking the connection again. Your essay may still be processing.' });
        continue;
      }
      onProgress(job);
      if (job.status === 'completed') return job.result;
      if (job.status === 'failed') throw new Error(job.error || 'The AI could not finish your essay. Please try again.');
    }
    throw new Error('This essay is taking longer than expected. Check the server PC before trying again.');
  },
  async updateFeedback(original, category) {
    const response = await fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ original, category }),
    });
    return readApiResponse(response);
  }
};

const fileInput = document.querySelector('#file-input');
const dropZone = document.querySelector('#drop-zone');
const previewPanel = document.querySelector('#preview-panel');
const previewList = document.querySelector('#preview-list');
const photoCount = document.querySelector('#photo-count');
const error = document.querySelector('#upload-error');
const improveButton = document.querySelector('#improve-button');
const readyHint = document.querySelector('#ready-hint');
const cloudConsent = document.querySelector('#cloud-consent');
const processing = document.querySelector('#processing');
const results = document.querySelector('#results');
const transcribedText = document.querySelector('#transcribed-text');
const updateFeedbackButton = document.querySelector('#update-feedback-button');
const transcriptionStatus = document.querySelector('#transcription-status');
const MAX_PHOTOS = 5;
const MAX_PHOTO_SIZE = 10 * 1024 * 1024;
let currentFiles = [];
const previewUrls = new Map();
let recognizedText = '';
let needsFeedbackRetry = false;
let processingTimer = null;
let processingProgress = null;
const processingMessages = [
  ['Reading your handwriting', 'Reading your page photos in order.'],
  ['Checking your writing', 'Finding grammar, clarity, and organization ideas.'],
  ['Preparing feedback', 'Writing helpful suggestions just for this essay.'],
];

function startProcessingUi() {
  processingProgress = null;
  const preview = document.querySelector('#loading-preview');
  preview.src = previewUrls.get(currentFiles[0]);
  let messageIndex = 0;
  const updateMessage = () => {
    const message = processingProgress || processingMessages[messageIndex];
    document.querySelector('#processing-title').childNodes[0].textContent = message[0];
    document.querySelector('#processing-copy').textContent = message[1];
    messageIndex = (messageIndex + 1) % processingMessages.length;
  };
  updateMessage();
  clearInterval(processingTimer);
  processingTimer = setInterval(updateMessage, 1800);
  processing.hidden = false;
}

function setProcessingProgress(progress) {
  if (!progress.title || !progress.message) return;
  processingProgress = [progress.title, progress.message];
  document.querySelector('#processing-title').childNodes[0].textContent = progress.title;
  document.querySelector('#processing-copy').textContent = progress.message;
}

function stopProcessingUi() {
  clearInterval(processingTimer);
  processingTimer = null;
  processing.hidden = true;
}

function showError(message) { error.textContent = message; error.hidden = false; }
function clearError() { error.hidden = true; error.textContent = ''; }
function renderSuggestions(suggestions) {
  const guide = document.querySelector('#improvement-guide');
  if (!suggestions.length) {
    const empty = document.createElement('li');
    empty.textContent = 'No specific revision questions are available yet. Check the converted text below.';
    guide.replaceChildren(empty);
    return;
  }
  guide.replaceChildren(...suggestions.map((suggestion) => {
    const item = document.createElement('li');
    const title = document.createElement('strong');
    title.textContent = `${suggestion.title}: `;
    const quote = document.createElement('blockquote');
    quote.textContent = `“${suggestion.quote}”`;
    const advice = document.createElement('p');
    advice.className = 'suggestion-advice';
    advice.textContent = suggestion.advice;
    const disagreeButton = document.createElement('button');
    disagreeButton.className = 'feedback-button';
    disagreeButton.type = 'button';
    disagreeButton.textContent = 'I disagree';
    disagreeButton.setAttribute('aria-pressed', 'false');
    disagreeButton.setAttribute('aria-expanded', 'false');
    const feedbackPanel = document.createElement('div');
    feedbackPanel.className = 'feedback-panel';
    feedbackPanel.hidden = true;
    const feedbackPrompt = document.createElement('p');
    feedbackPrompt.className = 'feedback-note';
    feedbackPrompt.textContent = 'AI can make mistakes. Which areas would you like to focus on instead?';
    const options = document.createElement('fieldset');
    options.className = 'feedback-options';
    const legend = document.createElement('legend');
    legend.textContent = 'Choose any areas';
    options.append(legend);
    const selectedNote = document.createElement('p');
    selectedNote.className = 'feedback-selection';
    selectedNote.setAttribute('aria-live', 'polite');
    selectedNote.hidden = true;
    for (const category of ['Grammar', 'Clarity', 'Ideas', 'Organization']) {
      const label = document.createElement('label');
      label.className = 'feedback-option';
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.value = category;
      const categoryName = document.createElement('span');
      categoryName.textContent = category;
      label.append(checkbox, categoryName);
      options.append(label);
    }
    options.addEventListener('change', () => {
      const selected = [...options.querySelectorAll('input:checked')].map((input) => input.value);
      selectedNote.hidden = selected.length === 0;
      selectedNote.textContent = selected.length
        ? `Focus areas: ${selected.join(', ')}. Use these as a checklist for your own revision.`
        : '';
    });
    const privacyNote = document.createElement('p');
    privacyNote.className = 'feedback-privacy-note';
    privacyNote.textContent = 'Your choices only appear here; they do not change or resend your essay.';
    feedbackPanel.append(feedbackPrompt, options, selectedNote, privacyNote);
    disagreeButton.addEventListener('click', () => {
      const disagreed = disagreeButton.getAttribute('aria-pressed') !== 'true';
      disagreeButton.setAttribute('aria-pressed', String(disagreed));
      disagreeButton.setAttribute('aria-expanded', String(disagreed));
      feedbackPanel.hidden = !disagreed;
    });
    item.append(title, quote, advice, disagreeButton, feedbackPanel);
    return item;
  }));
}
function updateSubmitState() {
  improveButton.disabled = !currentFiles.length || !cloudConsent.checked;
  readyHint.textContent = !currentFiles.length
    ? 'Add page photos and check the box to continue.'
    : cloudConsent.checked
      ? `${currentFiles.length} ${currentFiles.length === 1 ? 'page is' : 'pages are'} ready. Press the button when you are ready.`
      : 'Check the box to continue.';
}
function renderPreview() {
  previewPanel.hidden = currentFiles.length === 0;
  dropZone.hidden = currentFiles.length > 0;
  photoCount.textContent = `${currentFiles.length} ${currentFiles.length === 1 ? 'photo' : 'photos'} ready`;
  const rows = currentFiles.map((file, index) => {
    const row = document.createElement('div');
    row.className = 'preview-page';
    const thumbnail = document.createElement('img');
    thumbnail.src = previewUrls.get(file);
    thumbnail.alt = `Preview of essay page ${index + 1}`;
    const details = document.createElement('div');
    details.className = 'preview-page-details';
    const title = document.createElement('strong');
    title.textContent = `Page ${index + 1}`;
    const name = document.createElement('span');
    name.className = 'file-name';
    name.textContent = file.name;
    details.append(title, name);
    const actions = document.createElement('div');
    actions.className = 'page-actions';
    const moveEarlier = document.createElement('button');
    moveEarlier.className = 'text-button order-button';
    moveEarlier.type = 'button';
    moveEarlier.textContent = '↑';
    moveEarlier.setAttribute('aria-label', `Move page ${index + 1} earlier`);
    moveEarlier.disabled = index === 0;
    moveEarlier.addEventListener('click', () => movePage(index, -1));
    const moveLater = document.createElement('button');
    moveLater.className = 'text-button order-button';
    moveLater.type = 'button';
    moveLater.textContent = '↓';
    moveLater.setAttribute('aria-label', `Move page ${index + 1} later`);
    moveLater.disabled = index === currentFiles.length - 1;
    moveLater.addEventListener('click', () => movePage(index, 1));
    const remove = document.createElement('button');
    remove.className = 'text-button danger';
    remove.type = 'button';
    remove.textContent = 'Remove';
    remove.setAttribute('aria-label', `Remove page ${index + 1}`);
    remove.addEventListener('click', () => removePage(index));
    actions.append(moveEarlier, moveLater, remove);
    row.append(thumbnail, details, actions);
    return row;
  });
  previewList.replaceChildren(...rows);
  updateSubmitState();
}
function setFiles(fileList) {
  clearError();
  if (!fileList || !fileList.length) return;
  const supported = ['image/jpeg', 'image/png', 'image/webp'];
  let problem = '';
  for (const file of Array.from(fileList)) {
    if (!supported.includes(file.type)) {
      problem ||= 'Please choose JPG, PNG, or WEBP photos.';
      continue;
    }
    if (file.size > MAX_PHOTO_SIZE) {
      problem ||= 'Each photo must be smaller than 10 MB.';
      continue;
    }
    if (currentFiles.length >= MAX_PHOTOS) {
      problem ||= `You can add up to ${MAX_PHOTOS} page photos.`;
      break;
    }
    currentFiles.push(file);
    previewUrls.set(file, URL.createObjectURL(file));
  }
  renderPreview();
  if (problem) showError(problem);
}
function movePage(index, offset) {
  const newIndex = index + offset;
  if (newIndex < 0 || newIndex >= currentFiles.length) return;
  [currentFiles[index], currentFiles[newIndex]] = [currentFiles[newIndex], currentFiles[index]];
  renderPreview();
}
function removePage(index) {
  const [file] = currentFiles.splice(index, 1);
  URL.revokeObjectURL(previewUrls.get(file));
  previewUrls.delete(file);
  renderPreview();
}
function resetUpload() {
  currentFiles = [];
  for (const url of previewUrls.values()) URL.revokeObjectURL(url);
  previewUrls.clear();
  fileInput.value = '';
  renderPreview();
  cloudConsent.checked = false;
  clearError();
}
// Force the empty state when the page first loads or is restored from browser history.
resetUpload();
window.addEventListener('pageshow', (event) => { if (event.persisted) resetUpload(); });
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); fileInput.click(); } });
fileInput.addEventListener('change', () => { setFiles(fileInput.files); fileInput.value = ''; });
transcribedText.addEventListener('input', () => {
  const changed = transcribedText.value.trim() !== recognizedText.trim();
  updateFeedbackButton.disabled = !(changed || needsFeedbackRetry) || !transcribedText.value.trim();
  transcriptionStatus.textContent = changed
    ? 'You changed the text. Update the suggestions to use your corrections.'
    : needsFeedbackRetry
      ? 'Your photos have been read. You can retry feedback using this text.'
      : 'You can edit the text above.';
});
updateFeedbackButton.addEventListener('click', async () => {
  const correctedText = transcribedText.value.trim();
  if (!correctedText || (!needsFeedbackRetry && correctedText === recognizedText.trim())) return;
  updateFeedbackButton.disabled = true;
  transcribedText.readOnly = true;
  updateFeedbackButton.textContent = 'Updating suggestions…';
  transcriptionStatus.textContent = 'Using your corrected text to prepare new feedback…';
  try {
    const result = await essayService.updateFeedback(correctedText, document.querySelector('.category-card h3').textContent);
    document.querySelector('#ai-report').textContent = result.report;
    renderSuggestions(result.suggestions);
    recognizedText = correctedText;
    needsFeedbackRetry = false;
    transcriptionStatus.textContent = 'Updated. The suggestions now use your corrected text.';
  } catch (requestError) {
    transcriptionStatus.textContent = requestError.message || 'Could not update suggestions. Please try again.';
    updateFeedbackButton.disabled = false;
  } finally {
    transcribedText.readOnly = false;
    updateFeedbackButton.textContent = needsFeedbackRetry ? 'Retry feedback using this text' : 'Use corrected text for feedback';
  }
});
cloudConsent.addEventListener('change', updateSubmitState);
['dragenter', 'dragover'].forEach((type) => dropZone.addEventListener(type, (event) => { event.preventDefault(); dropZone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach((type) => dropZone.addEventListener(type, (event) => { event.preventDefault(); dropZone.classList.remove('dragging'); }));
dropZone.addEventListener('drop', (event) => setFiles(event.dataTransfer.files));
['dragenter', 'dragover'].forEach((type) => previewPanel.addEventListener(type, (event) => { event.preventDefault(); previewPanel.classList.add('dragging'); }));
['dragleave', 'drop'].forEach((type) => previewPanel.addEventListener(type, (event) => { event.preventDefault(); previewPanel.classList.remove('dragging'); }));
previewPanel.addEventListener('drop', (event) => setFiles(event.dataTransfer.files));
document.querySelector('#add-photos-button').addEventListener('click', () => fileInput.click());
document.querySelector('#remove-all-button').addEventListener('click', resetUpload);
improveButton.addEventListener('click', async () => {
  if (!currentFiles.length || !cloudConsent.checked) return;
  document.querySelector('#upload-section').hidden = true;
  document.querySelector('.how-it-works').hidden = true;
  startProcessingUi(); processing.scrollIntoView({ behavior: 'smooth', block: 'center' });
  try {
    const result = await essayService.improveEssay(currentFiles, setProcessingProgress);
    document.querySelector('.category-card h3').textContent = result.category;
    document.querySelector('.category-copy').textContent = result.categoryDescription;
    const resultImages = document.querySelector('#result-images');
    resultImages.replaceChildren(...currentFiles.map((file, index) => {
      const image = document.createElement('img');
      image.className = 'essay-photo';
      image.src = previewUrls.get(file);
      image.alt = `Your uploaded handwritten essay, page ${index + 1}`;
      return image;
    }));
    recognizedText = result.original || '';
    transcribedText.value = recognizedText;
    needsFeedbackRetry = Boolean(result.feedbackError);
    updateFeedbackButton.disabled = !needsFeedbackRetry;
    updateFeedbackButton.textContent = needsFeedbackRetry ? 'Retry feedback using this text' : 'Use corrected text for feedback';
    transcriptionStatus.textContent = result.feedbackError || 'You can edit the text above.';
    document.querySelector('#ai-report').textContent = result.report;
    renderSuggestions(result.suggestions);
    stopProcessingUi(); results.hidden = false; results.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (requestError) {
    stopProcessingUi(); document.querySelector('#upload-section').hidden = false; document.querySelector('.how-it-works').hidden = false;
    showError(requestError.message || 'We could not improve this photo. Please try again with a clearer image.');
  }
});
function startOver() { results.hidden = true; stopProcessingUi(); document.querySelector('#upload-section').hidden = false; document.querySelector('.how-it-works').hidden = false; resetUpload(); document.querySelector('#upload-section').scrollIntoView({ behavior: 'smooth', block: 'start' }); }
document.querySelector('#start-over-top').addEventListener('click', startOver);
document.querySelector('#start-over-bottom').addEventListener('click', startOver);
