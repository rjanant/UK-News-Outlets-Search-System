
const BASE_URL = 'http://127.0.0.1:8000'; // Update with your backend URL

export const fetchSearchResults = async (query, year, page = 1, limit = 10) => {
    // Remove empty parameters from the request
    const params = new URLSearchParams();
    params.append('q', query);
    if (year) params.append('year', year);
    if (page) params.append('page', page);
    if (limit) params.append('limit', limit);

    const url = `${BASE_URL}/search?${params.toString()}`;

    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error('Failed to fetch search results');
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching search results:', error);
        throw error;
    }
};


export const postSearchTest = async (data) => {
    try {
        const response = await fetch(`${BASE_URL}/search/test`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        if (!response.ok) {
            throw new Error('Failed to post search test');
        }
        return await response.json();
    } catch (error) {
        console.error('Error posting search test:', error);
        throw error;
    }
};

export const fetchSearchBoolean = async (query, year, page = 1, limit = 10) => {
    const params = new URLSearchParams();
    params.append('q', query);
    if (year) params.append('year', year);
    if (page) params.append('page', page);
    if (limit) params.append('limit', limit);

    const url = `${BASE_URL}/search/boolean?${params.toString()}`;

    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error('Failed to fetch boolean search results');
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching boolean search results:', error);
        throw error;
    }
};

export const fetchSearchTfidf = async () => {
    try {
        const response = await fetch(`${BASE_URL}/search/tfidf`);
        if (!response.ok) {
            throw new Error('Failed to fetch tfidf search results');
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching tfidf search results:', error);
        throw error;
    }
};
