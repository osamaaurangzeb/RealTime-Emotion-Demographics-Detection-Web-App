// Webcam functionality (original)
let startButton = document.getElementById("startButton");
let stopButton = document.getElementById("stopButton");
let video = document.getElementById("video");
let emotionResult = document.getElementById("emotionResult");
let ageResult = document.getElementById("ageResult");
let fetchInterval;

// Mode switching functionality
document.addEventListener('DOMContentLoaded', function() {
    const modeButtons = document.querySelectorAll('.mode-btn');
    const webcamSection = document.querySelector('.webcam-section');
    const uploadSection = document.querySelector('.upload-section');

    modeButtons.forEach(button => {
        button.addEventListener('click', function() {
            const mode = this.dataset.mode;
            
            // Update active button
            modeButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');

            // Switch sections
            if (mode === 'webcam') {
                webcamSection.classList.add('active');
                uploadSection.classList.remove('active');
            } else {
                webcamSection.classList.remove('active');
                uploadSection.classList.add('active');
                
                // Stop webcam if running
                if (!stopButton.disabled) {
                    stopButton.click();
                }
            }
        });
    });

    // Initialize upload functionality
    initializeUpload();
});

// Original webcam functionality
startButton.addEventListener('click', function() {
    video.src = "/expression-age/video_feed";
    startButton.disabled = true;
    stopButton.disabled = false;
    fetchInterval = setInterval(fetchLatestData, 1000);
});

stopButton.addEventListener('click', function() {
    video.src = "";
    startButton.disabled = false;
    stopButton.disabled = true;
    clearInterval(fetchInterval);
    emotionResult.textContent = "N/A";
    ageResult.textContent = "N/A";
});

function fetchLatestData() {
    // Fixed: Properly handle JSON responses
    fetch('/expression-age/latest_emotion')
        .then(response => response.json())
        .then(data => {
            emotionResult.textContent = data.emotion;
        })
        .catch(error => {
            console.error('Error fetching emotion:', error);
            emotionResult.textContent = "Error";
        });
        
    fetch('/expression-age/latest_age_gender')
        .then(response => response.json())
        .then(data => {
            ageResult.textContent = data.age_gender;
        })
        .catch(error => {
            console.error('Error fetching age_gender:', error);
            ageResult.textContent = "Error";
        });
}

// Image upload functionality - IMPROVED VERSION
function initializeUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const resultsContainer = document.getElementById('resultsContainer');
    const loadingMsg = document.getElementById('loadingMsg');
    const uploadedImage = document.getElementById('uploadedImage');
    const faceResults = document.getElementById('faceResults');
    const errorMsg = document.getElementById('errorMsg');

    // Global upload state to prevent multiple uploads
    let isUploading = false;

    // File input change handler - FIXED
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file && !isUploading) {
            handleFileUpload(file);
        }
        // Always reset file input to allow re-uploading the same file
        setTimeout(() => {
            this.value = '';
        }, 100);
    });

    // Drag and drop functionality - IMPROVED
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('dragover');
        
        if (isUploading) {
            showError('Please wait for the current upload to complete.');
            return;
        }
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                handleFileUpload(file);
            } else {
                showError('Please upload an image file.');
            }
        }
    });

    // Click to upload - IMPROVED
    uploadArea.addEventListener('click', function(e) {
        if (isUploading) {
            showError('Please wait for the current upload to complete.');
            return;
        }
        fileInput.click();
    });

    function handleFileUpload(file) {
        // Prevent multiple uploads
        if (isUploading) {
            showError('Please wait for the current upload to complete.');
            return;
        }

        // Validate file type
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
        if (!allowedTypes.includes(file.type)) {
            showError('Please upload a valid image file (JPG, PNG, GIF, BMP).');
            return;
        }

        // Validate file size (max 16MB to match backend)
        if (file.size > 16 * 1024 * 1024) {
            showError('File size should be less than 16MB.');
            return;
        }

        // Validate file is not empty
        if (file.size === 0) {
            showError('The selected file is empty. Please choose a valid image.');
            return;
        }

        // Set upload state
        isUploading = true;

        // Show loading
        hideError();
        showLoading();

        // Create FormData and upload
        const formData = new FormData();
        formData.append('file', file);

        // Add timeout to prevent hanging requests
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            controller.abort();
            isUploading = false;
        }, 45000); // 45 second timeout

        fetch('/expression-age/upload_image', {
            method: 'POST',
            body: formData,
            signal: controller.signal
        })
        .then(response => {
            clearTimeout(timeoutId);
            
            // Check if response is ok
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return response.json();
        })
        .then(data => {
            isUploading = false;
            hideLoading();
            
            if (data.success) {
                displayResults(data);
            } else {
                showError(data.error || 'Failed to process image.');
            }
        })
        .catch(error => {
            clearTimeout(timeoutId);
            isUploading = false;
            hideLoading();
            console.error('Error uploading image:', error);
            
            if (error.name === 'AbortError') {
                showError('Request timeout. Please try again with a smaller image.');
            } else if (error.message.includes('HTTP error')) {
                showError('Server error. Please try again.');
            } else {
                showError('Network error. Please check your connection and try again.');
            }
        });
    }

    function displayResults(data) {
        // Clear previous results
        faceResults.innerHTML = '';
        
        // Display processed image
        if (data.processed_image) {
            uploadedImage.src = 'data:image/jpeg;base64,' + data.processed_image;
            uploadedImage.style.display = 'block';
            
            // Add image load error handling
            uploadedImage.onerror = function() {
                showError('Failed to load processed image.');
            };
        }

        // Display face results
        if (data.results && data.results.length > 0) {
            // Create summary
            const summaryDiv = document.createElement('div');
            summaryDiv.className = 'face-summary';
            summaryDiv.innerHTML = `
                <h6 style="color: white; text-align: center; margin-bottom: 15px;">
                    Found ${data.results.length} face${data.results.length > 1 ? 's' : ''}
                </h6>
            `;
            faceResults.appendChild(summaryDiv);
            
            // Display each face result
            data.results.forEach((result, index) => {
                const faceDiv = document.createElement('div');
                faceDiv.className = 'face-result';
                
                faceDiv.innerHTML = `
                    <h6>Face ${result.face_id}</h6>
                    <p>Emotion: ${result.emotion}</p>
                    <p>${result.age_gender}</p>
                `;
                
                faceResults.appendChild(faceDiv);
            });
        } else {
            faceResults.innerHTML = '<p class="text-center" style="color: white;">No faces detected in the image.</p>';
        }

        resultsContainer.classList.add('show');
    }

    function showLoading() {
        loadingMsg.style.display = 'block';
        uploadedImage.style.display = 'none';
        faceResults.innerHTML = '';
        resultsContainer.classList.add('show');
        
        // Disable upload area during loading
        uploadArea.style.pointerEvents = 'none';
        uploadArea.style.opacity = '0.6';
    }

    function hideLoading() {
        loadingMsg.style.display = 'none';
        
        // Re-enable upload area
        uploadArea.style.pointerEvents = 'auto';
        uploadArea.style.opacity = '1';
    }

    function showError(message) {
        errorMsg.textContent = message;
        errorMsg.style.display = 'block';
        resultsContainer.classList.remove('show');
        
        // Auto-hide error after 7 seconds
        setTimeout(() => {
            hideError();
        }, 7000);
    }

    function hideError() {
        errorMsg.style.display = 'none';
    }
}