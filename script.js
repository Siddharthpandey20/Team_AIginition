// DOM elements
const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-button');
const chatMessages = document.getElementById('chat-messages');
const fileInput = document.getElementById('file-input');
const urlInput = document.getElementById('url-input');
const uploadButton = document.getElementById('upload-button');
const uploadStatus = document.getElementById('upload-status');
const documentItemsContainer = document.getElementById('document-items-container');

// Load previously uploaded documents from localStorage
let uploadedDocuments = [];
try {
    const savedDocs = localStorage.getItem('uploadedDocuments');
    if (savedDocs) {
        uploadedDocuments = JSON.parse(savedDocs);
    }
} catch (e) {
    console.error('Error loading saved documents:', e);
}

// Backend URL (replace with your actual backend URL)
const backendUrl = 'http://localhost:8000';

// Document array is already initialized above

// Add this constant at the top with other constants
const UPLOADS_DIR = 'uploads/';

// Auto-resize the textarea
chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
});

// Send message when Enter key is pressed (without Shift)
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Send message when send button is clicked
sendButton.addEventListener('click', sendMessage);

// Add event listeners after DOM elements are defined
uploadButton.addEventListener('click', async () => {
    const file = fileInput.files[0];
    const url = urlInput.value.trim();
    const selectedFileName = document.getElementById('selected-file-name');

    if (!file && !url) {
        showUploadStatus('Please select a file or enter a URL', false);
        return;
    }

    try {
        if (file) {
            showUploadStatus('📂 File selected: ' + file.name, true);
            selectedFileName.textContent = file.name;
            selectedFileName.classList.add('active');

            const formData = new FormData();
            formData.append('file', file);

            // Add document to list before upload completes
            addDocument(file.name, 'File');

            const response = await fetch(`${backendUrl}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Upload failed');
            const result = await response.json();

            // Start checking processing status
            if (result.filename) {
                await checkProcessingStatus(result.filename);
            }

            // Clear input fields
            fileInput.value = "";
            selectedFileName.textContent = "";
            selectedFileName.classList.remove('active');
        } else if (url) {
            showUploadStatus('⏳ Processing URL...', true);
            const response = await fetch(`${backendUrl}/upload-url`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url })
            });

            if (!response.ok) throw new Error('URL processing failed');
            result = await response.json();

            showUploadStatus('✅ URL processed successfully!', true);
            addDocument(url, 'URL');
            urlInput.value = "";
        }
    } catch (error) {
        console.error('Error:', error);
        // Remove failed upload from document list
        uploadedDocuments = uploadedDocuments.filter(d => d.name !== file.name);
        updateDocumentList();
        showUploadStatus(`❌ Error: ${error.message}`, false);
        isProcessingDocument = false;
        sendButton.disabled = false;
        chatInput.disabled = false;
        chatInput.placeholder = "Ask a question...";
    }
});

// Add a global variable for tracking document processing
let isProcessingDocument = false;

// Chat functionality
sendButton.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Add file input change handler
fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    const selectedFileName = document.getElementById('selected-file-name');
    if (file) {
        selectedFileName.textContent = file.name;
        selectedFileName.classList.add('active');
    } else {
        selectedFileName.textContent = '';
        selectedFileName.classList.remove('active');
    }
});

// Helper functions
function showUploadStatus(message, isSuccess) {
    const status = document.getElementById('upload-status');
    status.textContent = message;
    status.style.display = 'block';
    status.className = `upload-status ${isSuccess ? 'upload-success' : 'upload-error'}`;
    
    // Hide after 5 seconds
    setTimeout(() => {
        status.style.display = 'none';
    }, 5000);
}

function addDocument(name, type) {
    // Create timestamp when document is actually added
    const doc = {
        name: name,
        type: type,
        timestamp: new Date().toLocaleString(),
        // Use proper path joining
        filepath: type === 'File' ? `${UPLOADS_DIR}${name}` : name
    };
    
    // Always add document immediately after upload starts
    uploadedDocuments = uploadedDocuments.filter(d => d.name !== name);
    uploadedDocuments.unshift(doc);
    updateDocumentList();
    localStorage.setItem('uploadedDocuments', JSON.stringify(uploadedDocuments));
}

// Update the sendMessage function for better error handling
async function sendMessage() {
    if (isProcessingDocument) {
        showUploadStatus('Please wait while document processing completes...', false);
        return;
    }

    const message = chatInput.value.trim();
    if (!message) return;

    try {
        // Show user message before clearing input
        addUserMessage(message);
        
        // Clear input and adjust height after showing message
        chatInput.value = '';
        chatInput.style.height = 'auto';

        // Show loading indicator
        const loadingIndicator = addLoadingIndicator();

        // Send message to backend
        const response = await fetch(`${backendUrl}/chat/${encodeURIComponent(message)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });

        // Remove loading indicator
        loadingIndicator.remove();

        if (!response.ok) {
            throw new Error('Failed to get response from server');
        }

        const result = await response.json();

        // Display bot response only if we have an answer
        if (result.answer && result.answer.trim()) {
            addBotMessage(result.answer.trim());
        } else {
            addBotMessage('I apologize, but I was unable to find a relevant answer. Please try rephrasing your question.');
        }

    } catch (error) {
        console.error('Chat error:', error);
        addBotMessage('Sorry, I encountered an error processing your message. Please try again.');
    } finally {
        scrollToBottom();
    }
}

// Add user message to chat
function addUserMessage(message) {
    if (!message) return;
    
    const messageElement = document.createElement('div');
    messageElement.className = 'message user-message';
    messageElement.innerHTML = `
        <div class="avatar user-avatar">You</div>
        <div class="message-content">
            <div class="message-text">${escapeHtml(message)}</div>
        </div>
    `;
    chatMessages.appendChild(messageElement);
    scrollToBottom();
    saveChatHistory();
}

// Update addBotMessage for better error handling
function addBotMessage(message) {
    if (!message) return;
    
    const messageElement = document.createElement('div');
    messageElement.className = 'message bot-message';
    
    const formattedMessage = formatMessage(message);
    
    messageElement.innerHTML = `
        <div class="avatar bot-avatar">AI</div>
        <div class="message-content">
            <div class="message-text">${formattedMessage}</div>
        </div>
    `;
    chatMessages.appendChild(messageElement);
    scrollToBottom();
    saveChatHistory();
}

// Add system message to chat
function addSystemMessage(message) {
    const messageElement = document.createElement('div');
    messageElement.className = 'message system-message';
    messageElement.innerHTML = `
        <div class="avatar system-avatar">S</div>
        <div class="message-content">
            <div class="message-text sources-text">${formatMessage(message)}</div>
        </div>
    `;
    chatMessages.appendChild(messageElement);
    scrollToBottom();
}

// Add loading indicator
function addLoadingIndicator() {
    const loadingElement = document.createElement('div');
    loadingElement.className = 'loading-indicator';
    loadingElement.innerHTML = `
        <div class="loading-dots">
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
        </div>
    `;
    chatMessages.appendChild(loadingElement);
    scrollToBottom();
    return loadingElement;
}

// Scroll to bottom of chat
function scrollToBottom() {
    const lastMessage = chatMessages.lastElementChild;
    if (lastMessage) {
        lastMessage.scrollIntoView({ behavior: 'smooth' });
    }
}

// Show upload status
function showUploadStatus(message, isSuccess) {
    uploadStatus.textContent = message;
    uploadStatus.style.display = 'block';
    
    if (isSuccess) {
        uploadStatus.className = 'upload-status upload-success';
    } else {
        uploadStatus.className = 'upload-status upload-error';
    }
    
    setTimeout(() => {
        uploadStatus.style.display = 'none';
    }, 5000);
}

// Update document list
function updateDocumentList() {
    const container = document.getElementById('document-items-container');
    container.innerHTML = '';
    
    if (uploadedDocuments.length === 0) {
        const noDocsElement = document.createElement('div');
        noDocsElement.className = 'document-item';
        noDocsElement.textContent = 'No documents uploaded yet.';
        container.appendChild(noDocsElement);
        return;
    }
    
    uploadedDocuments.forEach((doc) => {
        const docElement = document.createElement('div');
        docElement.className = 'document-item';
        docElement.innerHTML = `
            <div class="doc-title">${escapeHtml(doc.name)}</div>
            <div class="doc-type">
                <span class="doc-icon">${doc.type === 'File' ? '📄' : '🔗'}</span>
                ${escapeHtml(doc.type)} • ${escapeHtml(doc.timestamp)}
            </div>
        `;
        container.appendChild(docElement);
    });
    
    // Save to localStorage for persistence
    localStorage.setItem('uploadedDocuments', JSON.stringify(uploadedDocuments));
}

// Update formatMessage function to handle empty messages
function formatMessage(message) {
    if (!message) return '';
    let formattedMessage = escapeHtml(message);
    // Add line breaks for better readability
    formattedMessage = formattedMessage.replace(/\n/g, '<br>');
    return formattedMessage;
}

// Fix the escapeHtml function
function escapeHtml(unsafe) {
    if (!unsafe) return '';
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")  // Fixed regex here
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// Initialize document list
updateDocumentList();

async function uploadFile(file) {
    try {
        showUploadStatus('Uploading file...', true);
        
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${backendUrl}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }
        
        const result = await response.json();
        console.log('Upload response:', result); // Debug log
        
        if (result.status === 'success') {
            showUploadStatus('File uploaded successfully!', true);
            
            // Add document to list with more details
            const docInfo = {
                name: result.filename,
                type: 'File',
                timestamp: new Date().toLocaleString(),
                count: result.document_count
            };
            
            // Add to documents array and update UI
            uploadedDocuments.unshift(docInfo);
            updateDocumentList();
            
            // Save to localStorage
            localStorage.setItem('uploadedDocuments', JSON.stringify(uploadedDocuments));
        } else {
            throw new Error(result.message || 'Upload failed');
        }
        
        return result;
        
    } catch (error) {
        console.error('Error uploading file:', error);
        showUploadStatus(`Upload failed: ${error.message}`, false);
        throw error;
    }
}

// Check the processing status of the uploaded file every 3 seconds.
async function checkProcessingStatus(filename) {
    isProcessingDocument = true;
    showUploadStatus('🛠 Extracting data... Please wait.', true);
    sendButton.disabled = true;
    chatInput.disabled = true;
    chatInput.placeholder = "Please wait while processing document...";
    
    let isProcessingComplete = false;
    while (!isProcessingComplete) {
        await new Promise(resolve => setTimeout(resolve, 3000));  // Wait 3 seconds
        
        try {
            const response = await fetch(`${backendUrl}/processing-status?filename=${encodeURIComponent(filename)}`);
            const result = await response.json();
            
            if (result.status === 'completed') {
                isProcessingComplete = true;
                isProcessingDocument = false;
                sendButton.disabled = false;
                chatInput.disabled = false;
                chatInput.placeholder = "Ask a question...";
                showUploadStatus('✅ Data extracted! You may now ask queries.', true);
            } else if (result.status === 'failed') {
                isProcessingComplete = true;
                isProcessingDocument = false;
                sendButton.disabled = false;
                chatInput.disabled = false;
                chatInput.placeholder = "Ask a question...";
                showUploadStatus('❌ File processing failed.', false);
            } else {
                showUploadStatus('⏳ Still processing...', true);
            }
        } catch (error) {
            console.error('Error checking status:', error);
            isProcessingComplete = true;
            isProcessingDocument = false;
        }
    }
}

// Update document list initialization
window.addEventListener('load', () => {
    try {
        // Clear any stale data
        localStorage.removeItem('uploadedDocuments');
        
        // Scan uploads directory for existing files
        fetch(`${backendUrl}/list-uploads`)
            .then(response => response.json())
            .then(files => {
                files.forEach(file => {
                    addDocument(file, 'File');
                });
            })
            .catch(error => console.error('Error loading existing files:', error));
    } catch (e) {
        console.error('Error initializing documents:', e);
        uploadedDocuments = [];
    }
    
    // Reset processing state
    isProcessingDocument = false;
    sendButton.disabled = false;
    chatInput.disabled = false;
    chatInput.placeholder = "Ask a question...";
    loadChatHistory();
    chatInput.focus();
});

// Add clear chat functionality
const clearChatButton = document.getElementById('clear-chat');
clearChatButton.addEventListener('click', () => {
    const confirmClear = confirm('Are you sure you want to clear the chat history?');
    if (confirmClear) {
        while (chatMessages.firstChild) {
            chatMessages.removeChild(chatMessages.firstChild);
        }
        localStorage.removeItem('chatHistory');
        
        // Add welcome message after clearing
        addBotMessage('Chat cleared. How can I help you today?');
    }
});

// Enable/disable send button based on input
chatInput.addEventListener('input', () => {
    sendButton.disabled = !chatInput.value.trim();
    
    // Auto resize input
    chatInput.style.height = 'auto';
    const newHeight = Math.min(chatInput.scrollHeight, 120);
    chatInput.style.height = newHeight + 'px';
});

// Save chat history to localStorage
function saveChatHistory() {
    const messages = Array.from(chatMessages.children).map(msg => ({
        type: msg.classList.contains('user-message') ? 'user' : 'bot',
        text: msg.querySelector('.message-text').textContent
    }));
    localStorage.setItem('chatHistory', JSON.stringify(messages));
}

// Load chat history
function loadChatHistory() {
    const history = localStorage.getItem('chatHistory');
    if (history) {
        const messages = JSON.parse(history);
        messages.forEach(msg => {
            if (msg.type === 'user') {
                addUserMessage(msg.text);
            } else {
                addBotMessage(msg.text);
            }
        });
    }
}
