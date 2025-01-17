import Link from 'next/link'
import { searchDocuments } from '../actions'

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

export default async function SearchResults({ searchParams }: { searchParams: { q?: string } }) {
  const query = searchParams.q
  const queryScheme = searchParams.query_scheme || 'ntc'
  const docScheme = searchParams.doc_scheme || 'ntc'
  
  if (!query) return null

  const results: SearchResponse = await searchDocuments(query, queryScheme, docScheme)

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-600 mb-4">
        Found {results.results_count} results for "{query}" 
        using query_scheme="{queryScheme.toUpperCase()}" 
        and doc_scheme="{docScheme.toUpperCase()}" 
      </p>
      {results.top_results.map((doc) => (
        <Link
          key={doc.doc_number}
          href={doc.url}
          target="_blank"
          rel="noopener noreferrer"
          className="block p-4 rounded-lg border border-gray-200 hover:border-gray-300 transition-colors"
        >
          <h2 className="text-lg font-semibold mb-2">Document {doc.doc_number}</h2>
          <p className="text-sm text-gray-600 mb-2">{doc.snippet}</p>
          <p className="text-xs text-blue-600 truncate">{doc.url}</p>
        </Link>
      ))}
      {results.top_results.length === 0 && (
        <p className="text-center text-gray-600">No results found.</p>
      )}
    </div>
  )
}

