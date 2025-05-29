document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    const sortSelect = document.getElementById('sort-select');
    const filters = {
        positive: document.getElementById('positive-filter'),
        negative: document.getElementById('negative-filter'),
        questions: document.getElementById('questions-filter'),
        suggestions: document.getElementById('suggestions-filter')
    };

    // Example comments data (in a real application, this would come from an API)
    let comments = [
        {
            id: 1,
            username: 'John Doe',
            avatar: 'JD',
            timestamp: '2 hours ago',
            type: 'positive',
            text: 'This video was incredibly helpful! I especially loved the detailed explanations and practical examples. Keep up the great work! 👍',
            likes: 24,
            replies: 3
        },
        {
            id: 2,
            username: 'Alice Smith',
            avatar: 'AS',
            timestamp: '5 hours ago',
            type: 'suggestion',
            text: 'Could you make a tutorial about advanced techniques? I think many of us would benefit from that kind of content!',
            likes: 18,
            replies: 2
        },
        {
            id: 3,
            username: 'Mike Brown',
            avatar: 'MB',
            timestamp: '1 day ago',
            type: 'question',
            text: 'At 5:23, you mentioned a special technique. Could you explain that part in more detail? I\'m having trouble understanding it.',
            likes: 12,
            replies: 4
        }
    ];

    // Search functionality
    function handleSearch() {
        const searchTerm = searchInput.value.toLowerCase();
        const filteredComments = comments.filter(comment => {
            const matchesSearch = comment.text.toLowerCase().includes(searchTerm) ||
                                comment.username.toLowerCase().includes(searchTerm);
            
            // Check if any filter is active
            const activeFilters = Object.entries(filters)
                .filter(([_, element]) => element.checked)
                .map(([key]) => key);

            // If no filters are active, show all comments that match the search
            if (activeFilters.length === 0) return matchesSearch;

            // If filters are active, check if the comment type matches any active filter
            return matchesSearch && activeFilters.includes(comment.type);
        });

        displayComments(filteredComments);
    }

    // Sort functionality
    function handleSort() {
        const sortBy = sortSelect.value;
        const sortedComments = [...comments].sort((a, b) => {
            switch (sortBy) {
                case 'recent':
                    // This is a simplified sort by timestamp
                    return b.id - a.id;
                case 'likes':
                    return b.likes - a.likes;
                case 'replies':
                    return b.replies - a.replies;
                default:
                    return 0;
            }
        });

        displayComments(sortedComments);
    }

    // Display comments
    function displayComments(commentsToShow) {
        const container = document.querySelector('.comments-container');
        container.innerHTML = commentsToShow.map(comment => `
            <div class="comment-card">
                <div class="comment-header">
                    <img src="https://ui-avatars.com/api/?name=${comment.avatar}&background=fff7f7&color=FF0000" alt="User Avatar" class="user-avatar">
                    <div class="comment-info">
                        <h3 class="username">${comment.username}</h3>
                        <span class="timestamp">${comment.timestamp}</span>
                    </div>
                    <div class="comment-type ${comment.type}">${comment.type.charAt(0).toUpperCase() + comment.type.slice(1)}</div>
                </div>
                <p class="comment-text">${comment.text}</p>
                <div class="comment-stats">
                    <span>❤️ ${comment.likes} likes</span>
                    <span>💬 ${comment.replies} replies</span>
                </div>
            </div>
        `).join('');
    }

    // Event listeners
    searchBtn.addEventListener('click', handleSearch);
    searchInput.addEventListener('keyup', (e) => {
        if (e.key === 'Enter') handleSearch();
    });
    sortSelect.addEventListener('change', handleSort);
    
    // Add event listeners for filters
    Object.values(filters).forEach(filter => {
        filter.addEventListener('change', handleSearch);
    });

    // Initial display
    displayComments(comments);
}); 