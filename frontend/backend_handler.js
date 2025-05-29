class YouTubeAnalyzer {
    constructor() {
        this.initialized = false;
        this.videoData = null;
    }

    async initializeVideo(videoUrl) {
        try {
            const response = await fetch('/api/initialize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ video_url: videoUrl })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            this.videoData = await response.json();
            this.initialized = true;

            // Redirect to video page with the data
            window.location.href = `video_page/video_page_index.html?videoId=${encodeURIComponent(videoUrl)}`;

        } catch (error) {
            console.error('Error initializing video:', error);
            throw error;
        }
    }

    isInitialized() {
        return this.initialized;
    }

    getVideoData() {
        return this.videoData;
    }
}

// Create a global instance
window.youtubeAnalyzer = new YouTubeAnalyzer(); 