'use server'

interface SearchResult {
  doc_number: string;
  score: number;
  url: string;
  snippet: string;
}

interface SearchResponse {
  query: string;
  tokens: Record<string, number>;
  results_count: number;
  top_results: SearchResult[];
}

export async function searchDocuments(
  query: string,
  queryScheme: string,
  docScheme: string
): Promise<SearchResponse> {
  // Replace this URL with your actual API endpoint
  const apiUrl = `http://127.0.0.1:8000/search?query=${encodeURIComponent(
    query
  )}&query_scheme=${encodeURIComponent(queryScheme)}&doc_scheme=${encodeURIComponent(docScheme)}`
  console.log('searchDocuments called with query:', query)
  try {
    const response = await fetch(apiUrl)
    console.log('Response status:', response.status) 
    if (!response.ok) {
      throw new Error('Failed to fetch search results')
    }
    const data: SearchResponse = await response.json()
    return data
  } catch (error) {
    console.error('Error fetching search results:', error)
    return {
      query,
      tokens: {},
      results_count: 0,
      top_results: []
    }
  }
}

