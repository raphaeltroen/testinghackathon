// Show popup when clicking the button
const typicalViewerBtn = document.getElementById('typical-viewer-btn');
if (typicalViewerBtn) {
    typicalViewerBtn.addEventListener('click', function() {
        document.getElementById('popup-overlay').style.display = 'flex';
    });
}

// Close popup when clicking the close button or outside the popup
const closeButton = document.querySelector('.close-button');
if (closeButton) {
    closeButton.addEventListener('click', function() {
        document.getElementById('popup-overlay').style.display = 'none';
    });
}

const popupOverlay = document.getElementById('popup-overlay');
if (popupOverlay) {
    popupOverlay.addEventListener('click', function(e) {
        if (e.target === this) {
            this.style.display = 'none';
        }
    });
}

// Suggestions tab click handler
const suggestionsTab = document.getElementById('suggestions-tab');
if (suggestionsTab) {
    suggestionsTab.addEventListener('click', function(e) {
        e.preventDefault();
        window.location.href = suggestionsTab.getAttribute('href');
    });
}
