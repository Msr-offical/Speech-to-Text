document.addEventListener('DOMContentLoaded', () => {
    // Variables
    let mediaRecorder = null;
    let audioChunks = [];
    let audioFile = null;
    let isRecording = false;
    const downloadTranscriptionBtn = document.getElementById('downloadTranscriptionBtn');
    const downloadTranscriptionContainer = document.getElementById('downloadTranscriptionContainer');
    let lastTranscriptionText = '';

    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const audioFileInput = document.getElementById('audioFile');
    const recordButton = document.getElementById('recordButton');
    const transcribeButton = document.getElementById('transcribeButton');
    const transcribeButtonContainer = document.getElementById('transcribeButtonContainer');
    const fileStatus = document.getElementById('fileStatus');
    const fileStatusText = document.getElementById('fileStatusText');
    const recordStatus = document.getElementById('recordStatus');
    const recordStatusText = document.getElementById('recordStatusText');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const transcriptionText = document.getElementById('transcriptionText');

    // WER Section logic (not modal)
    const werButton = document.getElementById('werButton');
    const werSection = document.getElementById('werSection');
    const calculateWerButton = document.getElementById('calculateWerButton');
    const groundTruthInput = document.getElementById('groundTruthInput');
    const werResults = document.getElementById('werResults');
    const transcriptionTextDiv = document.getElementById('transcriptionText');
    const toggleViewButton = document.getElementById('toggleViewButton');
    const textDiffView = document.getElementById('textDiffView');
    let isDiffView = false;

    // File Upload Handling
    uploadArea.addEventListener('click', () => audioFileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--accent-color)';
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = 'var(--primary-color)';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--primary-color)';
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('audio/')) {
            handleFileUpload(file);
        } else {
            alert('Please drop a valid audio file');
        }
    });

    audioFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file && file.type.startsWith('audio/')) {
            handleFileUpload(file);
        } else {
            alert('Please select a valid audio file');
        }
    });

    const stepsContainer = document.getElementById('stepsContainer');

    function handleFileUpload(file) {
        audioFile = file;
        fileStatus.classList.add('active');
        fileStatusText.textContent = 'File selected: ' + file.name;
        transcribeButtonContainer.style.display = 'flex';
        if (stepsContainer) stepsContainer.style.display = 'none';
    }

    // Recording Handling
    recordButton.addEventListener('click', async function() {
        const recordButton = this;
        const recordStatus = document.getElementById('recordStatus');
        const recordStatusText = document.getElementById('recordStatusText');
        const transcribeButtonContainer = document.getElementById('transcribeButtonContainer');

        try {
            if (!isRecording) {
                // Hide transcribe button when starting recording
                transcribeButtonContainer.style.display = 'none';
                
                // Request microphone access
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                
                // Initialize MediaRecorder
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                // Handle data available event
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };

                // Handle recording stop
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    
                    // Update UI
                    recordStatus.className = 'status-indicator status-success';
                    recordStatusText.textContent = 'Processing recording...';
                    
                    // Store the audio blob for transcription
                    window.recordedAudio = audioBlob;
                    
                    // Clean up
                    stream.getTracks().forEach(track => track.stop());
                    
                    // Automatically transcribe the recording
                    try {
                        await transcribeAudio(audioBlob, 'recorded');
                        recordStatusText.textContent = 'Transcription completed';
                    } catch (error) {
                        console.error('Error during transcription:', error);
                        recordStatus.className = 'status-indicator status-error';
                        recordStatusText.textContent = 'Error in transcription';
                    }
                };

                // Start recording
                mediaRecorder.start();
                isRecording = true;
                
                // Update UI
                recordButton.innerHTML = '<i class="fas fa-stop"></i><span>Stop Recording</span>';
                recordStatus.className = 'status-indicator status-recording';
                recordStatusText.textContent = 'Recording...';
                transcribeButtonContainer.style.display = 'none';
                if (stepsContainer) stepsContainer.style.display = 'none';
                
            } else {
                // Stop recording
                mediaRecorder.stop();
                isRecording = false;
                
                // Update UI
                recordButton.innerHTML = '<i class="fas fa-microphone"></i><span>Start Recording</span>';
                // Don't show transcribe button as it will auto-transcribe
                transcribeButtonContainer.style.display = 'none';
            }
        } catch (error) {
            console.error('Error accessing microphone:', error);
            
            // Update UI with error
            recordStatus.className = 'status-indicator status-error';
            recordStatusText.textContent = 'Error accessing microphone. Please ensure you have granted microphone permissions.';
            
            // Show detailed error message to user
            let errorMessage = 'Error accessing microphone. ';
            if (error.name === 'NotAllowedError') {
                errorMessage += 'Please grant microphone permissions in your browser settings.';
            } else if (error.name === 'NotFoundError') {
                errorMessage += 'No microphone found. Please connect a microphone and try again.';
            } else {
                errorMessage += 'Please try again or use a different browser.';
            }
            
            // Create and show error popup
            const errorPopup = document.createElement('div');
            errorPopup.className = 'error-popup';
            errorPopup.innerHTML = `
                <div class="error-content">
                    <i class="fas fa-exclamation-circle"></i>
                    <p>${errorMessage}</p>
                    <button onclick="this.parentElement.parentElement.remove()">OK</button>
                </div>
            `;
            document.body.appendChild(errorPopup);
        }
    });

    // Transcription Handling
    transcribeButton.addEventListener('click', async () => {
        if (audioFile) {
            await transcribeAudio(audioFile, 'upload');
        }
    });

    async function transcribeAudio(file, source) {
        try {
            showLoading();
            
            const formData = new FormData();
            formData.append('audio', file);
            
            // Use the correct endpoint based on the source
            const endpoint = source === 'upload' ? '/transcribe_upload' : '/transcribe_recorded';
            console.log('Sending request to:', endpoint);
            
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Transcription failed');
            }
            
            const result = await response.json();
            if (result.error) {
                throw new Error(result.error);
            }
            
            showTranscription(result.text);
            // Update status text after successful transcription
            if (source === 'recorded') {
                recordStatusText.textContent = 'Transcription complete';
            }
        } catch (error) {
            console.error('Error during transcription:', error);
            alert(`Error during transcription: ${error.message}`);
            // Update status text on error
            if (source === 'recorded') {
                recordStatusText.textContent = 'Error in transcription';
            }
        } finally {
            hideLoading();
        }
    }

    function showLoading() {
        loadingSpinner.style.display = 'flex';
        loadingSpinner.style.justifyContent = 'center';
        loadingSpinner.style.alignItems = 'center';
        loadingSpinner.style.height = '100%';
        transcriptionText.style.display = 'none';
    }

    function hideLoading() {
        loadingSpinner.style.display = 'none';
        transcriptionText.style.display = 'block';
    }

    function showTranscription(text) {
        const MAX_VISIBLE_CHARS = 500; // Show first 500 characters initially
        lastTranscriptionText = text;
        if (downloadTranscriptionContainer) downloadTranscriptionContainer.style.display = 'flex';
        if (text.length > MAX_VISIBLE_CHARS) {
            const truncatedText = text.substring(0, MAX_VISIBLE_CHARS);
            const remainingText = text.substring(MAX_VISIBLE_CHARS);
            
            transcriptionText.innerHTML = `
                <div class="transcription-content">
                    <span class="visible-text">${truncatedText}</span>
                    <span class="hidden-text" style="display: none;">${remainingText}</span>
                </div>
                <button class="see-more-btn">See More</button>
            `;
            
            const seeMoreBtn = transcriptionText.querySelector('.see-more-btn');
            const hiddenText = transcriptionText.querySelector('.hidden-text');
            const visibleText = transcriptionText.querySelector('.visible-text');
            
            seeMoreBtn.addEventListener('click', () => {
                if (hiddenText.style.display === 'none') {
                    hiddenText.style.display = 'inline';
                    seeMoreBtn.textContent = 'See Less';
                    visibleText.style.maxHeight = 'none';
                } else {
                    hiddenText.style.display = 'none';
                    seeMoreBtn.textContent = 'See More';
                    visibleText.style.maxHeight = 'initial';
                }
            });
        } else {
            transcriptionText.textContent = text;
        }
    }

    // Show WER section when button is clicked
    werButton.addEventListener('click', () => {
        if (werSection.style.display === 'block') {
            werSection.style.display = 'none';
        } else {
            werSection.style.display = 'block';
            werResults.innerHTML = '';
            textDiffView.innerHTML = '';
            groundTruthInput.value = '';
            textDiffView.style.display = 'none';
            werResults.style.display = 'block';
            toggleViewButton.textContent = 'Switch to Text Differences';
            isDiffView = false;
        }
    });

    // File upload logic for ground truth
    const groundTruthFileInput = document.getElementById('groundTruthFile');
    const groundTruthFileName = document.getElementById('groundTruthFileName');
    let uploadedGroundTruth = '';

    groundTruthFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file && file.type === 'text/plain') {
            const reader = new FileReader();
            reader.onload = function(evt) {
                uploadedGroundTruth = evt.target.result.trim();
                groundTruthFileName.textContent = file.name;
            };
            reader.readAsText(file);
        } else {
            uploadedGroundTruth = '';
            groundTruthFileName.textContent = '';
        }
    });

    // Toggle view logic
    toggleViewButton.addEventListener('click', () => {
        isDiffView = !isDiffView;
        if (isDiffView) {
            toggleViewButton.textContent = 'Switch to Comparison';
            werResults.style.display = 'none';
            textDiffView.style.display = 'block';
            // Re-render text differences when switching to diff view
            const groundTruth = getGroundTruthValue();
            const transcription = getTranscriptionTextOnly();
            if (groundTruth && transcription) {
                const diffHtml = renderTextDiff(transcription, groundTruth);
                textDiffView.innerHTML = diffHtml;
            }
        } else {
            toggleViewButton.textContent = 'Switch to Text Differences';
            werResults.style.display = 'block';
            textDiffView.style.display = 'none';
        }
    });

    // Helper function to get ground truth value
    function getGroundTruthValue() {
        // First check if there's an uploaded file
        if (uploadedGroundTruth) {
            return uploadedGroundTruth;
        }
        // Then check the textarea input
        const groundTruthInput = document.getElementById('groundTruthInput');
        return groundTruthInput ? groundTruthInput.value.trim() : '';
    }

    // Helper function to get transcription text without button labels
    function getTranscriptionTextOnly() {
        // If the transcriptionTextDiv contains .visible-text and .hidden-text, concatenate them
        const visible = transcriptionTextDiv.querySelector('.visible-text');
        const hidden = transcriptionTextDiv.querySelector('.hidden-text');
        if (visible) {
            let text = visible.textContent || '';
            if (hidden && hidden.style.display !== 'none') {
                text += hidden.textContent || '';
            }
            return text.trim();
        }
        // Otherwise, just use the textContent (no buttons)
        return transcriptionTextDiv.textContent.trim();
    }

    // Add Download Diff as HTML button
    const downloadDiffHtmlBtn = document.createElement('button');
    downloadDiffHtmlBtn.id = 'downloadDiffHtmlBtn';
    downloadDiffHtmlBtn.textContent = 'Download Difference';
    // Use the exact same classes as the Compare & Calculate WER button
    downloadDiffHtmlBtn.className = 'btn btn-primary';
    // Place the button in the same parent as Compare & Calculate WER
    const compareBtn = document.querySelector('.btn.btn-primary');
    if (compareBtn && compareBtn.parentElement) {
        compareBtn.parentElement.appendChild(downloadDiffHtmlBtn);
    } else if (toggleViewButton && toggleViewButton.parentElement) {
        toggleViewButton.parentElement.appendChild(downloadDiffHtmlBtn);
    }

    // Download Diff as HTML logic
    downloadDiffHtmlBtn.addEventListener('click', async () => {
        const groundTruth = getGroundTruthValue();
        const transcription = getTranscriptionTextOnly();
        // Render the full diff HTML (all tokens, not just loaded chunk)
        const fullDiffHtml = renderTextDiff(transcription, groundTruth, true); // pass true for full diff
        let fileName = 'text-differences.html';
        if (audioFile && audioFile.name) {
            const base = audioFile.name.replace(/\.[^/.]+$/, "");
            fileName = base + '-diff.html';
        }

        // --- Gather WER/model stats ---
        const modelResults = window.modelResults || [];
        let werStatsHtml = '';
        if (modelResults.length > 0) {
            werStatsHtml += `<section style="background:#fff;border-radius:14px;box-shadow:0 2px 12px rgba(30,58,138,0.06);border:1.5px solid #e5e7eb;padding:2rem 1.5rem 1.5rem 1.5rem;margin-bottom:2.5rem;max-width:900px;margin:auto;">
                <h2 style="color:#1E3A8A;font-family:'Yanone Kaffeesatz',sans-serif;font-size:2.2rem;margin-bottom:1.2rem;text-align:center;border-bottom:2px solid #D4AF37;padding-bottom:0.5rem;">WER & Model Statistics</h2>
                <div style="overflow-x:auto;">
                <table style="width:100%;border-collapse:collapse;font-size:1.1rem;">
                    <thead style="background:#f4f6fa;">
                        <tr>
                            <th style="padding:8px 12px;border-bottom:2px solid #D4AF37;">Model/File</th>
                            <th style="padding:8px 12px;border-bottom:2px solid #D4AF37;">WER (%)</th>
                            <th style="padding:8px 12px;border-bottom:2px solid #D4AF37;">Words</th>
                            <th style="padding:8px 12px;border-bottom:2px solid #D4AF37;">Subs</th>
                            <th style="padding:8px 12px;border-bottom:2px solid #D4AF37;">Ins</th>
                            <th style="padding:8px 12px;border-bottom:2px solid #D4AF37;">Dels</th>
                            <th style="padding:8px 12px;border-bottom:2px solid #D4AF37;">Precision</th>
                            <th style="padding:8px 12px;border-bottom:2px solid #D4AF37;">Recall</th>
                            <th style="padding:8px 12px;border-bottom:2px solid #D4AF37;">F1</th>
                        </tr>
                    </thead>
                    <tbody>`;
            modelResults.forEach(m => {
                werStatsHtml += `<tr style="text-align:center;">
                    <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;">${m.identifier}</td>
                    <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;color:#1E3A8A;font-weight:bold;">${m.wer}</td>
                    <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;">${m.totalWords}</td>
                    <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;color:#D97706;">${m.substitutions}</td>
                    <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;color:#059669;">${m.insertions}</td>
                    <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;color:#DC2626;">${m.deletions}</td>
                    <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;">${(m.precision*100).toFixed(1)}%</td>
                    <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;">${(m.recall*100).toFixed(1)}%</td>
                    <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;">${(m.f1*100).toFixed(1)}%</td>
                </tr>`;
            });
            werStatsHtml += `</tbody></table></div></section>`;
        }

        // --- Gather all graph canvases as images ---
        let graphsHtml = '';
        const graphSection = document.getElementById('werGraphsImages');
        if (graphSection) {
            const canvases = graphSection.querySelectorAll('canvas');
            if (canvases.length > 0) {
                graphsHtml += `<section style="background:#fff;border-radius:14px;box-shadow:0 2px 12px rgba(30,58,138,0.06);border:1.5px solid #e5e7eb;padding:2rem 1.5rem 1.5rem 1.5rem;margin-bottom:2.5rem;max-width:900px;margin:auto;">
                    <h2 style="color:#1E3A8A;font-family:'Yanone Kaffeesatz',sans-serif;font-size:2.2rem;margin-bottom:1.2rem;text-align:center;border-bottom:2px solid #D4AF37;padding-bottom:0.5rem;">WER Graphs</h2>
                    <div style="display:flex;flex-wrap:wrap;gap:2rem;justify-content:center;">`;
                canvases.forEach((canvas, idx) => {
                    const imgData = canvas.toDataURL('image/png');
                    // Try to get the title from the previous sibling (the label)
                    let title = '';
                    if (canvas.previousSibling && canvas.previousSibling.textContent) {
                        title = canvas.previousSibling.textContent;
                    }
                    graphsHtml += `<div style="flex:1 1 320px;max-width:420px;text-align:center;background:#f8fafc;border-radius:10px;padding:1.2rem 1rem 1.5rem 1rem;box-shadow:0 2px 8px rgba(30,58,138,0.10);margin-bottom:1.5rem;border:1.5px solid #e5e7eb;">
                        <div style="font-weight:bold;color:#1E3A8A;font-size:1.1rem;margin-bottom:0.7rem;">${title}</div>
                        <img src="${imgData}" alt="Graph ${idx+1}" style="max-width:100%;border-radius:8px;box-shadow:0 1px 6px rgba(30,58,138,0.07);background:#fff;"/>
                    </div>`;
                });
                graphsHtml += `</div></section>`;
            }
        }

        // --- Compose the final HTML ---
        const htmlContent = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>WER & Text Differences</title>
<link href='https://fonts.googleapis.com/css2?family=Yanone+Kaffeesatz:wght@200;300;400;500;600;700&display=swap' rel='stylesheet'>
<style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f7f8fa; color: #1E293B; padding: 2.5rem 0; }
    h2 { color: #1E3A8A; font-family: 'Yanone Kaffeesatz', sans-serif; font-size: 2.2rem; margin-bottom: 1.2rem; text-align: center; border-bottom: 2px solid #D4AF37; padding-bottom: 0.5rem; }
    .diff-eq { color: #222; }
    .diff-sub { background: #ffe082; color: #b26a00; border-radius: 4px; padding: 0.1em 0.3em; }
    .diff-ins { background: #e8f5e9; color: #388e3c; border-radius: 4px; padding: 0.1em 0.3em; }
    .diff-del { background: #ffebee; color: #c62828; text-decoration: line-through; border-radius: 4px; padding: 0.1em 0.3em; }
    .diff-container { background: #fff; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); padding: 2rem; margin-top: 2rem; max-width:900px; margin-left:auto; margin-right:auto; }
</style>
</head>
<body>
${werStatsHtml}
${graphsHtml}
<section class="diff-container">
    <h2 style="margin-top:0;">Text Differences</h2>
    <div>${fullDiffHtml}</div>
</section>
</body>
</html>`;
        const blob = new Blob([htmlContent], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 100);
    });

    // Refactor renderTextDiff to support full export mode (no controls, all tokens)
    function renderTextDiff(hypothesis, reference, exportAll) {
        const h = hypothesis.split(/\s+/).filter(word => word.length > 0);
        const r = reference.split(/\s+/).filter(word => word.length > 0);
        // Alignment logic
        const d = Array(r.length + 1).fill(null).map(() => Array(h.length + 1).fill(0));
        const op = Array(r.length + 1).fill(null).map(() => Array(h.length + 1).fill(''));
        for (let i = 0; i <= r.length; i++) { d[i][0] = i; op[i][0] = 'del'; }
        for (let j = 0; j <= h.length; j++) { d[0][j] = j; op[0][j] = 'ins'; }
        op[0][0] = '';
        for (let i = 1; i <= r.length; i++) {
            for (let j = 1; j <= h.length; j++) {
                if (r[i - 1] === h[j - 1]) {
                    d[i][j] = d[i - 1][j - 1];
                    op[i][j] = 'eq';
                } else {
                    const del = d[i - 1][j] + 1;
                    const ins = d[i][j - 1] + 1;
                    const sub = d[i - 1][j - 1] + 1;
                    const min = Math.min(del, ins, sub);
                    d[i][j] = min;
                    if (min === sub) op[i][j] = 'sub';
                    else if (min === ins) op[i][j] = 'ins';
                    else op[i][j] = 'del';
                }
            }
        }
        // Reconstruct the alignment path
        let i = r.length, j = h.length;
        const diffTokens = [];
        while (i > 0 || j > 0) {
            if (i > 0 && j > 0 && op[i][j] === 'eq') {
                diffTokens.unshift({ type: 'eq', word: r[i - 1] });
                i--; j--;
            } else if (i > 0 && j > 0 && op[i][j] === 'sub') {
                diffTokens.unshift({ type: 'sub', word: h[j - 1] });
                i--; j--;
            } else if (j > 0 && op[i][j] === 'ins') {
                diffTokens.unshift({ type: 'ins', word: h[j - 1] });
                j--;
            } else if (i > 0 && op[i][j] === 'del') {
                diffTokens.unshift({ type: 'del', word: r[i - 1] });
                i--;
            } else {
                break;
            }
        }
        // If exporting, return all tokens as a single HTML string (no controls)
        if (exportAll) {
            let html = '';
            diffTokens.forEach(token => {
                if (token.type === 'eq') {
                    html += `<span class="diff-eq">${token.word}</span> `;
                } else if (token.type === 'sub') {
                    html += `<span class="diff-sub">${token.word}</span> `;
                } else if (token.type === 'ins') {
                    html += `<span class="diff-ins">${token.word}</span> `;
                } else if (token.type === 'del') {
                    html += `<span class="diff-del">${token.word}</span> `;
                }
            });
            return html;
        }
        // Otherwise, render the paginated/chunked UI as before
        // Dynamically set chunk size so there are at most 3 pages
        const MAX_CHUNKS = 3;
        const totalTokens = diffTokens.length;
        const CHUNK_SIZE = Math.ceil(totalTokens / MAX_CHUNKS);
        let loadedChunks = 1;
        const totalChunks = Math.ceil(totalTokens / CHUNK_SIZE);
        // Helper to build HTML for a chunk
        function buildChunkHtml(chunkIdx) {
            const start = chunkIdx * CHUNK_SIZE;
            const end = Math.min(diffTokens.length, (chunkIdx + 1) * CHUNK_SIZE);
            let html = '';
            for (let k = start; k < end; k++) {
                const token = diffTokens[k];
                if (token.type === 'eq') {
                    html += `<span class="diff-eq">${token.word}</span> `;
                } else if (token.type === 'sub') {
                    html += `<span class="diff-sub">${token.word}</span> `;
                } else if (token.type === 'ins') {
                    html += `<span class="diff-ins">${token.word}</span> `;
                } else if (token.type === 'del') {
                    html += `<span class="diff-del">${token.word}</span> `;
                }
            }
            return html;
        }
        // Initial HTML structure with improved styling
        let initialHtml = `
            <div style="font-size:1.1rem;line-height:2;word-break:break-word;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h3 style="margin: 0;">Text Differences</h3>
                    <div>
                        <button id="closeDiffView" class="button" style="padding: 0.5rem 1rem; margin-right: 0.5rem;">
                            <i class="fas fa-times"></i> Close
                        </button>
                        <button id="seeMoreBtn" class="button" style="padding: 0.5rem 1rem;">
                            Load More (${loadedChunks}/${totalChunks})
                        </button>
                    </div>
                </div>
                <div id="diffContent" style="max-height: 400px; overflow-y: auto; padding: 1.2rem; background: #f7f8fa; border-radius: 10px; border: 1px solid #e0e0e0; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                    ${buildChunkHtml(0)}
                </div>
            </div>
            <style>
                .diff-eq { color: #222; }
                .diff-sub { background: #ffe082; color: #b26a00; border-radius: 4px; padding: 0.1em 0.3em; }
                .diff-ins { background: #e8f5e9; color: #388e3c; border-radius: 4px; padding: 0.1em 0.3em; }
                .diff-del { background: #ffebee; color: #c62828; text-decoration: line-through; border-radius: 4px; padding: 0.1em 0.3em; }
            </style>
        `;
        setTimeout(() => {
            const seeMoreBtn = document.getElementById('seeMoreBtn');
            const closeDiffView = document.getElementById('closeDiffView');
            const diffContent = document.getElementById('diffContent');
            if (seeMoreBtn) {
                seeMoreBtn.addEventListener('click', () => {
                    if (loadedChunks < totalChunks) {
                        const prevScrollHeight = diffContent.scrollHeight;
                        diffContent.innerHTML += buildChunkHtml(loadedChunks);
                        loadedChunks++;
                        seeMoreBtn.textContent = `Load More (${loadedChunks}/${totalChunks})`;
                        diffContent.scrollTop = prevScrollHeight;
                        if (loadedChunks >= totalChunks) {
                            seeMoreBtn.textContent = 'All Content Loaded';
                            seeMoreBtn.disabled = true;
                        }
                    }
                });
            }
            if (closeDiffView) {
                closeDiffView.addEventListener('click', () => {
                    textDiffView.style.display = 'none';
                    werResults.style.display = 'block';
                    toggleViewButton.textContent = 'Switch to Text Differences';
                    isDiffView = false;
                });
            }
        }, 0);
        return initialHtml;
    }

    // WER calculation function
    function calculateWER(hypothesis, reference) {
        // Split into words and normalize
        const h = hypothesis.split(/\s+/).filter(word => word.length > 0);
        const r = reference.split(/\s+/).filter(word => word.length > 0);
        
        // Initialize dynamic programming table
        const d = Array(r.length + 1).fill(null).map(() => Array(h.length + 1).fill(0));
        const op = Array(r.length + 1).fill(null).map(() => Array(h.length + 1).fill(''));
        
        // Initialize first row and column
        for (let i = 0; i <= r.length; i++) {
            d[i][0] = i;
            op[i][0] = 'del';
        }
        for (let j = 0; j <= h.length; j++) {
            d[0][j] = j;
            op[0][j] = 'ins';
        }
        op[0][0] = '';
        
        // Fill the table
        for (let i = 1; i <= r.length; i++) {
            for (let j = 1; j <= h.length; j++) {
                if (r[i - 1] === h[j - 1]) {
                    d[i][j] = d[i - 1][j - 1];
                    op[i][j] = 'eq';
                } else {
                    const del = d[i - 1][j] + 1;
                    const ins = d[i][j - 1] + 1;
                    const sub = d[i - 1][j - 1] + 1;
                    const min = Math.min(del, ins, sub);
                    d[i][j] = min;
                    if (min === sub) op[i][j] = 'sub';
                    else if (min === ins) op[i][j] = 'ins';
                    else op[i][j] = 'del';
                }
            }
        }
        
        // Count operations
        let i = r.length, j = h.length;
        let subs = 0, ins = 0, dels = 0;
        while (i > 0 || j > 0) {
            if (i > 0 && j > 0 && op[i][j] === 'eq') {
                i--; j--;
            } else if (i > 0 && j > 0 && op[i][j] === 'sub') {
                subs++; i--; j--;
            } else if (j > 0 && op[i][j] === 'ins') {
                ins++; j--;
            } else if (i > 0 && op[i][j] === 'del') {
                dels++; i--;
            } else {
                break;
            }
        }
        
        // Calculate WER
        const totalErrors = subs + dels + ins;
        const wer = (totalErrors / r.length) * 100;
        
        return {
            wer: wer.toFixed(2),
            subs: subs,
            dels: dels,
            ins: ins,
            totalWords: r.length
        };
    }

    // Ensure global modelResults array exists
    if (!window.modelResults) window.modelResults = [];

    // Update calculateWerButton event listener
    calculateWerButton.addEventListener('click', () => {
        // Get ground truth and transcription
        const groundTruth = getGroundTruthValue();
        const transcription = getTranscriptionTextOnly();
        let identifier = '';
        // Try to get file name or ask user for model name
        if (audioFile && audioFile.name) {
            identifier = audioFile.name;
        } else {
            identifier = prompt('Enter a name for this model or file:', 'Model ' + (window.modelResults.length + 1));
            if (!identifier) identifier = 'Model ' + (window.modelResults.length + 1);
        }

        // Validate inputs
        if (!groundTruth || !transcription) {
            const errorMessage = '<span style="color:var(--error-color);font-family:var(--title-font);">Both ground truth and transcription are required.</span>';
            werResults.innerHTML = errorMessage;
            textDiffView.innerHTML = errorMessage;
            return;
        }

        try {
            // Calculate WER
            const werData = calculateWER(transcription, groundTruth);

            // Compute ML metrics
            const TP = werData.totalWords - werData.subs - werData.dels;
            const FP = werData.ins;
            const FN = werData.dels;
            const TN = 0; // Not well-defined for text
            const precision = TP + FP > 0 ? TP / (TP + FP) : 0;
            const recall = TP + FN > 0 ? TP / (TP + FN) : 0;
            const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

            // Update or add to modelResults
            const idx = window.modelResults.findIndex(m => m.identifier === identifier);
            const newResult = {
                identifier,
                wer: parseFloat(werData.wer),
                totalWords: werData.totalWords,
                substitutions: werData.subs,
                insertions: werData.ins,
                deletions: werData.dels,
                tp: TP,
                tn: TN,
                fp: FP,
                fn: FN,
                precision: precision,
                recall: recall,
                f1: f1
            };
            if (idx >= 0) {
                window.modelResults[idx] = newResult;
            } else {
                window.modelResults.push(newResult);
            }

            // Update WER results display
            werResults.innerHTML = `
                <div style="font-size:1.3rem;"><strong>WER Score:</strong> ${werData.wer}%</div>
                <div style="margin-top:1rem;font-size:1.1rem;"><strong>Legend:</strong></div>
                <div style="margin-top:0.5rem;">
                    <span style="color:var(--warning-color);">Substitutions</span> (amber),
                    <span style="color:var(--error-color);">Deletions</span> (red),
                    <span style="color:var(--success-color);">Insertions</span> (green)
                </div>
                <div style="margin-top:1rem;">
                    <div><strong>Substitutions:</strong> <span style="color:var(--warning-color);">${werData.subs}</span></div>
                    <div><strong>Deletions:</strong> <span style="color:var(--error-color);">${werData.dels}</span></div>
                    <div><strong>Insertions:</strong> <span style="color:var(--success-color);">${werData.ins}</span></div>
                    <div><strong>Total Words:</strong> ${werData.totalWords}</div>
                </div>
                <div style="margin-top:1rem;"><strong>Saved as:</strong> <span style="color:var(--primary-color);">${identifier}</span></div>
            `;

            // Update text differences view
            const diffHtml = renderTextDiff(transcription, groundTruth);
            textDiffView.innerHTML = diffHtml;

            // Show appropriate view based on current state
            if (isDiffView) {
                werResults.style.display = 'none';
                textDiffView.style.display = 'block';
            } else {
                werResults.style.display = 'block';
                textDiffView.style.display = 'none';
            }
        } catch (error) {
            console.error('Error calculating WER:', error);
            const errorMessage = '<span style="color:var(--error-color);font-family:var(--title-font);">Error calculating WER. Please try again.</span>';
            werResults.innerHTML = errorMessage;
            textDiffView.innerHTML = errorMessage;
        }
    });

    // Add Show Graphs button logic with multiple ML model graphs
    const showGraphsBtn = document.getElementById('showGraphsBtn');
    const werGraphsSection = document.getElementById('werGraphsSection');
    const werGraphsImages = document.getElementById('werGraphsImages');

    showGraphsBtn.addEventListener('click', () => {
        if (werGraphsSection.style.display === 'block') {
            werGraphsSection.style.display = 'none';
            werGraphsSection.classList.remove('fade-slide-in');
        } else {
            werGraphsSection.style.display = 'block';
            setTimeout(() => werGraphsSection.classList.add('fade-slide-in'), 10);
            werGraphsImages.innerHTML = '';

            // === DYNAMIC DATA: Replace or populate this array with real model/file results ===
            const modelResults = window.modelResults || [
              {
                identifier: "Whisper (sample1.wav)",
                wer: 23.5,
                totalWords: 1500,
                substitutions: 200,
                insertions: 30,
                deletions: 20,
                tp: 1200,
                tn: 200,
                fp: 50,
                fn: 50,
                precision: 0.96,
                recall: 0.95,
                f1: 0.955
              },
              {
                identifier: "Google STT (test_audio.mp3)",
                wer: 18.2,
                totalWords: 1600,
                substitutions: 150,
                insertions: 20,
                deletions: 10,
                tp: 1300,
                tn: 220,
                fp: 40,
                fn: 40,
                precision: 0.97,
                recall: 0.97,
                f1: 0.97
              },
              {
                identifier: "DeepSpeech (meeting.wav)",
                wer: 30.1,
                totalWords: 1400,
                substitutions: 250,
                insertions: 40,
                deletions: 30,
                tp: 1100,
                tn: 180,
                fp: 60,
                fn: 60,
                precision: 0.92,
                recall: 0.90,
                f1: 0.91
              }
            ];
            // === END DYNAMIC DATA ===

            // Helper to create and append a canvas
            function addCanvas(title) {
              const wrapper = document.createElement('div');
              wrapper.style.marginBottom = '2rem';
              const label = document.createElement('span');
              label.textContent = title;
              label.style.display = 'block';
              label.style.textAlign = 'center';
              label.style.fontWeight = 'bold';
              label.style.marginBottom = '0.5rem';
              wrapper.appendChild(label);
              const canvas = document.createElement('canvas');
              wrapper.appendChild(canvas);
              werGraphsImages.appendChild(wrapper);
              return canvas;
            }

            if (window.Chart) {
              // 1. WER Percentage by Model
              const werCanvas = addCanvas('WER Percentage by Model/File');
              new Chart(werCanvas, {
                type: 'bar',
                data: {
                  labels: modelResults.map(m => m.identifier),
                  datasets: [{
                    label: 'WER (%)',
                    data: modelResults.map(m => m.wer),
                    backgroundColor: 'rgba(54, 162, 235, 0.7)'
                  }]
                },
                options: { responsive: false, plugins: { legend: { display: false }, tooltip: { enabled: true } } }
              });

              // 2. Total Words Count by Model
              const wordsCanvas = addCanvas('Total Words by Model/File');
              new Chart(wordsCanvas, {
                type: 'bar',
                data: {
                  labels: modelResults.map(m => m.identifier),
                  datasets: [{
                    label: 'Total Words',
                    data: modelResults.map(m => m.totalWords),
                    backgroundColor: 'rgba(255, 206, 86, 0.7)'
                  }]
                },
                options: { responsive: false, plugins: { legend: { display: false }, tooltip: { enabled: true } } }
              });

              // 3. Error Types by Model (Stacked Bar)
              const errorCanvas = addCanvas('Error Types by Model/File');
              new Chart(errorCanvas, {
                type: 'bar',
                data: {
                  labels: modelResults.map(m => m.identifier),
                  datasets: [
                    {
                      label: 'Substitutions',
                      data: modelResults.map(m => m.substitutions),
                      backgroundColor: 'rgba(255, 193, 7, 0.7)'
                    },
                    {
                      label: 'Insertions',
                      data: modelResults.map(m => m.insertions),
                      backgroundColor: 'rgba(40, 167, 69, 0.7)'
                    },
                    {
                      label: 'Deletions',
                      data: modelResults.map(m => m.deletions),
                      backgroundColor: 'rgba(220, 53, 69, 0.7)'
                    }
                  ]
                },
                options: {
                  responsive: false,
                  plugins: { legend: { position: 'top' }, tooltip: { enabled: true } },
                  scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } }
                }
              });

              // 4. Confusion Matrix by Model (Grouped Bar)
              const cmCanvas = addCanvas('Confusion Matrix (TP, FP, FN, TN) by Model/File');
              new Chart(cmCanvas, {
                type: 'bar',
                data: {
                  labels: modelResults.map(m => m.identifier),
                  datasets: [
                    { label: 'TP (Correct)', data: modelResults.map(m => m.tp), backgroundColor: 'rgba(40, 167, 69, 0.7)' },
                    { label: 'FP (Insertions)', data: modelResults.map(m => m.fp), backgroundColor: 'rgba(255, 193, 7, 0.7)' },
                    { label: 'FN (Deletions)', data: modelResults.map(m => m.fn), backgroundColor: 'rgba(220, 53, 69, 0.7)' },
                    { label: 'TN (N/A)', data: modelResults.map(m => m.tn), backgroundColor: 'rgba(54, 162, 235, 0.7)' }
                  ]
                },
                options: {
                  responsive: false,
                  plugins: {
                    legend: { position: 'top' },
                    tooltip: { enabled: true },
                    title: {
                      display: true,
                      text: 'Mapping: Insertions=FP, Deletions=FN, Correct=TP, TN=N/A',
                      font: { size: 12 },
                      color: '#888'
                    }
                  },
                  scales: { y: { beginAtZero: true } }
                }
              });

              // 5. Radar/Spider Chart for Error Profile
              const radarCanvas = addCanvas('Error Profile (Radar/Spider) by Model/File');
              new Chart(radarCanvas, {
                type: 'radar',
                data: {
                  labels: ['Correct', 'Substitutions', 'Deletions', 'Insertions'],
                  datasets: modelResults.map((m, i) => ({
                    label: m.identifier,
                    data: [m.tp, m.substitutions, m.deletions, m.insertions],
                    fill: true,
                    backgroundColor: `rgba(${54 + i*60}, ${162 - i*40}, ${235 - i*60}, 0.2)`,
                    borderColor: `rgba(${54 + i*60}, ${162 - i*40}, ${235 - i*60}, 1)`,
                    pointBackgroundColor: `rgba(${54 + i*60}, ${162 - i*40}, ${235 - i*60}, 1)`
                  }))
                },
                options: {
                  responsive: false,
                  plugins: { legend: { position: 'top' }, tooltip: { enabled: true } }
                }
              });

              // 6. Precision, Recall, F1-score Bar Chart
              const prfCanvas = addCanvas('Precision, Recall, F1-score by Model/File');
              new Chart(prfCanvas, {
                type: 'bar',
                data: {
                  labels: modelResults.map(m => m.identifier),
                  datasets: [
                    {
                      label: 'Precision',
                      data: modelResults.map(m => m.precision),
                      backgroundColor: 'rgba(54, 162, 235, 0.7)'
                    },
                    {
                      label: 'Recall',
                      data: modelResults.map(m => m.recall),
                      backgroundColor: 'rgba(40, 167, 69, 0.7)'
                    },
                    {
                      label: 'F1-score',
                      data: modelResults.map(m => m.f1),
                      backgroundColor: 'rgba(255, 193, 7, 0.7)'
                    }
                  ]
                },
                options: {
                  responsive: false,
                  plugins: { legend: { position: 'top' }, tooltip: { enabled: true } },
                  scales: { y: { beginAtZero: true, max: 1 } }
                }
              });

              // 7. WER Breakdown (Pie/Donut) for first model
              const pieCanvas = addCanvas(`WER Breakdown for ${modelResults[0].identifier}`);
              new Chart(pieCanvas, {
                type: 'doughnut',
                data: {
                  labels: ['Substitutions', 'Insertions', 'Deletions'],
                  datasets: [{
                    data: [
                      modelResults[0].substitutions,
                      modelResults[0].insertions,
                      modelResults[0].deletions
                    ],
                    backgroundColor: [
                      'rgba(255, 193, 7, 0.7)',
                      'rgba(40, 167, 69, 0.7)',
                      'rgba(220, 53, 69, 0.7)'
                    ]
                  }]
                },
                options: { responsive: false, plugins: { tooltip: { enabled: true } } }
              });

              // 8. WER vs. Total Words (Scatter Plot)
              const scatterCanvas = addCanvas('WER vs. Total Words');
              new Chart(scatterCanvas, {
                type: 'scatter',
                data: {
                  datasets: [{
                    label: 'Model/File',
                    data: modelResults.map(m => ({ x: m.totalWords, y: m.wer })),
                    backgroundColor: 'rgba(54, 162, 235, 0.7)'
                  }]
                },
                options: {
                  responsive: false,
                  plugins: { legend: { display: false }, tooltip: { enabled: true } },
                  scales: {
                    x: { title: { display: true, text: 'Total Words' } },
                    y: { title: { display: true, text: 'WER (%)' } }
                  }
                }
              });
            } else {
              werGraphsImages.innerHTML = '<span style="color:var(--error-color);font-size:1.1rem;">Chart.js not loaded. Please include Chart.js for dynamic graphs.</span>';
            }
        }
    });

    // Add styles for error popup
    const style = document.createElement('style');
    style.textContent = `
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
            display: inline-block;
        }
        
        .status-indicator.status-success {
            background-color: #4CAF50;
        }
        
        .status-indicator.status-error {
            background-color: #f44336;
        }
        
        .status-indicator.status-recording {
            background-color: #ff9800;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% {
                transform: scale(1);
                box-shadow: 0 0 0 0 rgba(255, 152, 0, 0.7);
            }
            70% {
                transform: scale(1.2);
                box-shadow: 0 0 0 10px rgba(255, 152, 0, 0);
            }
            100% {
                transform: scale(1);
                box-shadow: 0 0 0 0 rgba(255, 152, 0, 0);
            }
        }
        
        .record-button.recording {
            background-color: #f44336;
        }
        
        .status-container {
            display: flex;
            align-items: center;
            margin-top: 10px;
        }
        
        .error-popup {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        
        .error-content {
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            max-width: 400px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        .error-content i {
            color: #ff4444;
            font-size: 48px;
            margin-bottom: 15px;
        }
        
        .error-content p {
            margin: 15px 0;
            color: #333;
        }
        
        .error-content button {
            background: var(--primary-color);
            color: white;
            border: none;
            padding: 8px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .error-content button:hover {
            background: var(--accent-color);
        }
    `;
    document.head.appendChild(style);

    // Fade-in animation for cards and blocks on scroll
    function fadeInOnScroll() {
        const fadeEls = document.querySelectorAll('.card, .steps-card, .text-block');
        const observer = new window.IntersectionObserver((entries, obs) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                    obs.unobserve(entry.target);
                }
            });
        }, { threshold: 0.15 });
        fadeEls.forEach(el => {
            observer.observe(el);
        });
    }
    fadeInOnScroll();

    if (downloadTranscriptionBtn) {
        downloadTranscriptionBtn.addEventListener('click', function() {
            if (!lastTranscriptionText) return;
            const blob = new Blob([lastTranscriptionText], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'transcription.txt';
            document.body.appendChild(a);
            a.click();
            setTimeout(() => {
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }, 100);
        });
    }

    const copyTranscriptionBtn = document.getElementById('copyTranscriptionBtn');
    if (copyTranscriptionBtn) {
        copyTranscriptionBtn.addEventListener('click', async function() {
            if (!lastTranscriptionText) return;
            try {
                await navigator.clipboard.writeText(lastTranscriptionText);
                const original = copyTranscriptionBtn.innerHTML;
                copyTranscriptionBtn.innerHTML = '<i class="fas fa-check button-icon"></i>Copied!';
                copyTranscriptionBtn.disabled = true;
                setTimeout(() => {
                    copyTranscriptionBtn.innerHTML = original;
                    copyTranscriptionBtn.disabled = false;
                }, 1200);
            } catch (e) {
                copyTranscriptionBtn.innerHTML = '<i class="fas fa-times button-icon"></i>Error';
                setTimeout(() => {
                    copyTranscriptionBtn.innerHTML = '<i class="fas fa-copy button-icon"></i>Copy to Clipboard';
                }, 1200);
            }
        });
    }
}); 